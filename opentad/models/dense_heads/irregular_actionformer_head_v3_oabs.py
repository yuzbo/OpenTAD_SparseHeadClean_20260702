import torch
from torch.nn import functional as F

from ..builder import HEADS
from .irregular_actionformer_head_v3 import IrregularActionFormerHeadV3


@HEADS.register_module()
class IrregularActionFormerHeadV3OABS(IrregularActionFormerHeadV3):
    def __init__(
        self,
        *args,
        oabs_mode="visible",
        oabs_boundary_gamma=1.0,
        oabs_delta_max=10000.0,
        oabs_singleton_expand=1.0,
        oabs_singleton_mode="expanded_span",
        oabs_observed_only=True,
        oaa_enabled=False,
        oaa_min_overlap_ratio=0.05,
        **kwargs,
    ):
        if oabs_mode not in {"visible", "full"}:
            raise ValueError(f"Unsupported oabs_mode={oabs_mode}. Expected one of ['visible', 'full'].")
        if oabs_singleton_mode not in {"expanded_span", "legacy_zero_span"}:
            raise ValueError(
                "Unsupported oabs_singleton_mode="
                f"{oabs_singleton_mode}. Expected one of ['expanded_span', 'legacy_zero_span']."
            )

        self.oabs_mode = oabs_mode
        self.oabs_boundary_gamma = float(oabs_boundary_gamma)
        self.oabs_delta_max = float(oabs_delta_max)
        self.oabs_singleton_expand = float(oabs_singleton_expand)
        self.oabs_singleton_mode = oabs_singleton_mode
        self.oabs_observed_only = bool(oabs_observed_only)
        self.oaa_enabled = bool(oaa_enabled)
        self.oaa_min_overlap_ratio = float(oaa_min_overlap_ratio)
        super().__init__(*args, **kwargs)

    def forward_train(self, feat_list, mask_list, temporal_grid_list, gt_segments, gt_labels, **kwargs):
        cls_pred = []
        reg_pred = []
        boundary_pred = []
        for level_idx, (feat, mask, temporal_grid) in enumerate(zip(feat_list, mask_list, temporal_grid_list)):
            cls_out, reg_out, boundary_out = self._forward_single_level(feat, mask, level_idx, temporal_grid)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)
            boundary_pred.append(boundary_out)

        points = self.prior_generator(feat_list, temporal_grid_list)
        return self.losses(
            cls_pred,
            reg_pred,
            boundary_pred,
            mask_list,
            points,
            temporal_grid_list,
            gt_segments,
            gt_labels,
        )

    def _clamp_oabs_delta(self, delta):
        return delta.clamp(min=self.reg_denom_floor, max=self.oabs_delta_max)

    @torch.no_grad()
    def _build_oabs_observation_info(self, points, temporal_grid_list, gt_segments):
        fine_points = points[0]
        fine_grid = temporal_grid_list[0]
        fine_valid_masks = fine_grid["valid_mask"].bool()
        fine_fresh_masks = fine_grid.get("fresh_mask", fine_valid_masks).bool() & fine_valid_masks
        obs_infos = []

        for point, fresh_mask, gt_segment in zip(fine_points, fine_fresh_masks, gt_segments):
            num_gts = int(gt_segment.shape[0])
            info = dict(
                obs_count=gt_segment.new_zeros((num_gts,), dtype=torch.long),
                cls_valid=gt_segment.new_zeros((num_gts,), dtype=torch.bool),
                reg_valid=gt_segment.new_zeros((num_gts,), dtype=torch.bool),
                cls_segments=gt_segment.new_zeros((num_gts, 2)),
                reg_segments=gt_segment.new_zeros((num_gts, 2)),
                gt_lengths=gt_segment.new_zeros((num_gts,)),
                visible_lengths=gt_segment.new_zeros((num_gts,)),
                visible_ratio=gt_segment.new_zeros((num_gts,)),
                c_start=gt_segment.new_zeros((num_gts,)),
                c_end=gt_segment.new_zeros((num_gts,)),
                delta_start=gt_segment.new_zeros((num_gts,)),
                delta_end=gt_segment.new_zeros((num_gts,)),
                support_intervals=[],
            )
            if num_gts == 0:
                obs_infos.append(info)
                continue

            observed_points = point[fresh_mask]
            if observed_points.shape[0] == 0:
                obs_infos.append(info)
                continue

            observed_time = observed_points[:, 0]
            observed_left = observed_points[:, 3].clamp_min(self.reg_denom_floor)
            observed_right = observed_points[:, 4].clamp_min(self.reg_denom_floor)
            observed_delta = self._clamp_oabs_delta(0.5 * (observed_left + observed_right))

            for gt_idx, seg in enumerate(gt_segment):
                start = seg[0]
                end = seg[1]
                gt_len = (end - start).clamp_min(self.reg_denom_floor)
                info["gt_lengths"][gt_idx] = gt_len
                inside_mask = torch.logical_and(observed_time >= start, observed_time <= end)
                inside_idx = torch.nonzero(inside_mask, as_tuple=False).flatten()
                obs_count = int(inside_idx.numel())
                info["obs_count"][gt_idx] = obs_count
                support = gt_segment.new_zeros((0, 2))

                start_nearest_idx = (observed_time - start).abs().argmin()
                end_nearest_idx = (observed_time - end).abs().argmin()
                start_delta = observed_delta[start_nearest_idx]
                end_delta = observed_delta[end_nearest_idx]
                info["delta_start"][gt_idx] = start_delta
                info["delta_end"][gt_idx] = end_delta
                info["c_start"][gt_idx] = (
                    1.0 - (start - observed_time[start_nearest_idx]).abs() / start_delta
                ).clamp_(min=0.0, max=1.0)
                info["c_end"][gt_idx] = (
                    1.0 - (end - observed_time[end_nearest_idx]).abs() / end_delta
                ).clamp_(min=0.0, max=1.0)

                if obs_count > 0:
                    # Approximate the locally observed support around each fresh point
                    # using its asymmetric fine-grid cell footprint.
                    support_time = observed_time[inside_idx]
                    support_left = observed_left[inside_idx]
                    support_right = observed_right[inside_idx]
                    support_start = torch.maximum(support_time - 0.5 * support_left, start.expand_as(support_time))
                    support_end = torch.minimum(support_time + 0.5 * support_right, end.expand_as(support_time))
                    support = torch.stack([support_start, support_end], dim=-1)

                if obs_count >= 2:
                    visible = observed_time[inside_idx]
                    visible_span = torch.stack([visible.min(), visible.max()])
                    visible_len = (visible_span[1] - visible_span[0]).clamp_min(0.0)
                    info["cls_valid"][gt_idx] = True
                    info["reg_valid"][gt_idx] = True
                    info["cls_segments"][gt_idx] = visible_span
                    info["reg_segments"][gt_idx] = visible_span
                    info["visible_lengths"][gt_idx] = visible_len
                    info["visible_ratio"][gt_idx] = (visible_len / gt_len).clamp_(min=0.0, max=1.0)
                elif obs_count == 1:
                    singleton_idx = inside_idx[0]
                    singleton_time = observed_time[singleton_idx]
                    if self.oabs_singleton_mode == "legacy_zero_span":
                        singleton_start = singleton_time
                        singleton_end = singleton_time
                    else:
                        singleton_delta = observed_delta[singleton_idx] * self.oabs_singleton_expand
                        singleton_start = torch.maximum(singleton_time - singleton_delta, start)
                        singleton_end = torch.minimum(singleton_time + singleton_delta, end)
                    singleton_span = torch.stack([singleton_start, singleton_end])
                    singleton_len = (singleton_end - singleton_start).clamp_min(0.0)
                    info["cls_valid"][gt_idx] = True
                    info["reg_valid"][gt_idx] = True
                    info["cls_segments"][gt_idx] = singleton_span
                    info["reg_segments"][gt_idx] = singleton_span
                    info["visible_lengths"][gt_idx] = singleton_len
                    info["visible_ratio"][gt_idx] = (singleton_len / gt_len).clamp_(min=0.0, max=1.0)

                info["support_intervals"].append(support)

            obs_infos.append(info)
        return obs_infos

    @torch.no_grad()
    def _build_oaa_overlap_matrix(self, point, obs_info):
        cls_valid = obs_info["cls_valid"]
        cls_valid_idx = torch.nonzero(cls_valid, as_tuple=False).flatten()
        num_pts = int(point.shape[0])
        overlap = point.new_zeros((num_pts, int(cls_valid_idx.numel())))
        if cls_valid_idx.numel() == 0:
            return overlap

        point_left = point[:, 3].clamp_min(self.reg_denom_floor)
        point_right = point[:, 4].clamp_min(self.reg_denom_floor)
        point_start = point[:, 0] - 0.5 * point_left
        point_end = point[:, 0] + 0.5 * point_right
        point_len = (point_end - point_start).clamp_min(self.reg_denom_floor)

        support_intervals = obs_info.get("support_intervals", [])
        for col_idx, gt_idx in enumerate(cls_valid_idx.tolist()):
            if gt_idx >= len(support_intervals):
                continue
            support = support_intervals[gt_idx]
            if support.numel() == 0:
                continue
            inter_start = torch.maximum(point_start[:, None], support[None, :, 0])
            inter_end = torch.minimum(point_end[:, None], support[None, :, 1])
            inter = (inter_end - inter_start).clamp_min(0.0)
            overlap[:, col_idx] = (inter / point_len[:, None]).max(dim=1).values
        return overlap

    @torch.no_grad()
    def prepare_targets(self, points, temporal_grid_list, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=1)
        gt_cls = []
        gt_reg = []
        reg_weight_list = []
        debug_state = {
            "oabs_mode": self.oabs_mode,
            "oabs_singleton_mode": self.oabs_singleton_mode,
            "oaa_enabled": self.oaa_enabled,
            "oaa_min_overlap_ratio": self.oaa_min_overlap_ratio,
        }
        obs_infos = self._build_oabs_observation_info(points, temporal_grid_list, gt_segments)

        level_lengths = [int(level.shape[1]) for level in points]
        level_offsets = []
        offset = 0
        for length in level_lengths:
            level_offsets.append((offset, offset + length))
            offset += length

        for point, gt_segment, gt_label, obs_info in zip(concat_points, gt_segments, gt_labels, obs_infos):
            num_pts = int(point.shape[0])
            cls_target = point.new_zeros((num_pts, self.num_classes))
            reg_target = point.new_zeros((num_pts, 2))
            reg_weight = point.new_zeros((num_pts,))
            total_cost = None
            candidate_mask = None
            multi_gt_points = 0
            base_candidate_mask = None
            oaa_overlap = None
            oaa_missing_gt = None

            if gt_segment.shape[0] > 0 and obs_info["cls_valid"].any():
                cls_segments = obs_info["cls_segments"][obs_info["cls_valid"]]
                cls_labels = gt_label[obs_info["cls_valid"]]
                cls_gt = cls_segments[None].expand(num_pts, cls_segments.shape[0], 2)
                center_t = point[:, 0, None]
                left = center_t - cls_gt[:, :, 0]
                right = cls_gt[:, :, 1] - center_t
                cls_reg_targets = torch.stack((left, right), dim=-1)
                base_candidate_mask = self._build_candidate_mask(point, cls_gt, cls_reg_targets)
                candidate_mask = base_candidate_mask
                if self.oaa_enabled:
                    oaa_overlap = self._build_oaa_overlap_matrix(point, obs_info)
                    oaa_mask = oaa_overlap > self.oaa_min_overlap_ratio
                    gated_candidate_mask = base_candidate_mask & oaa_mask
                    # Keep OAA diagnostic clean, but never allow gating to erase all
                    # candidates for a GT column.
                    oaa_missing_gt = ~gated_candidate_mask.any(dim=0)
                    if oaa_missing_gt.any():
                        gated_candidate_mask[:, oaa_missing_gt] = base_candidate_mask[:, oaa_missing_gt]
                    candidate_mask = gated_candidate_mask
                assign_weights, total_cost = self._build_assignment_weights(point, cls_segments, candidate_mask)

                one_hot = F.one_hot(cls_labels.long(), self.num_classes).to(assign_weights.dtype)
                weighted_labels = assign_weights[:, :, None] * one_hot[None, :, :]
                cls_target = weighted_labels.max(dim=1).values.clamp_(min=0.0, max=1.0)
                multi_gt_points = int(((assign_weights > 0).sum(dim=1) > 1).sum().item())

                reg_col_mask = obs_info["reg_valid"][obs_info["cls_valid"]]
                if reg_col_mask.any():
                    reg_segments = obs_info["reg_segments"][obs_info["reg_valid"]]
                    reg_assign = assign_weights[:, reg_col_mask]
                    reg_gt = reg_segments[None].expand(num_pts, reg_segments.shape[0], 2)
                    reg_left = center_t - reg_gt[:, :, 0]
                    reg_right = reg_gt[:, :, 1] - center_t
                    denom_left = point[:, 3, None].clamp_min(self.reg_denom_floor)
                    denom_right = point[:, 4, None].clamp_min(self.reg_denom_floor)
                    reg_targets_log = torch.stack(
                        [
                            torch.log1p((reg_left / denom_left).clamp_min(0.0)),
                            torch.log1p((reg_right / denom_right).clamp_min(0.0)),
                        ],
                        dim=-1,
                    )
                    reg_weight, best_gt_idx = reg_assign.max(dim=1)
                    reg_target = reg_targets_log[torch.arange(num_pts, device=point.device), best_gt_idx]
                    reg_target = torch.where(reg_weight[:, None] > 0, reg_target, reg_target.new_zeros(reg_target.shape))

            gt_cls.append(cls_target)
            gt_reg.append(reg_target)
            reg_weight_list.append(reg_weight)

            if self.debug_enabled:
                positive_mask = cls_target.max(dim=-1).values > 0
                obs_count = obs_info["obs_count"]
                debug_state.setdefault("oabs_gt_total_per_sample", []).append(int(gt_segment.shape[0]))
                debug_state.setdefault("oabs_gt_zero_obs_per_sample", []).append(int((obs_count == 0).sum().item()))
                debug_state.setdefault("oabs_gt_single_obs_per_sample", []).append(int((obs_count == 1).sum().item()))
                debug_state.setdefault("oabs_gt_multi_obs_per_sample", []).append(int((obs_count >= 2).sum().item()))
                debug_state.setdefault("oabs_cls_valid_gt_per_sample", []).append(int(obs_info["cls_valid"].sum().item()))
                debug_state.setdefault("oabs_reg_valid_gt_per_sample", []).append(int(obs_info["reg_valid"].sum().item()))
                debug_state.setdefault("oabs_positive_points_per_sample", []).append(int(positive_mask.sum().item()))
                debug_state.setdefault("oabs_reg_points_per_sample", []).append(int((reg_weight > 0).sum().item()))
                debug_state.setdefault("oabs_c_start_mean_per_sample", []).append(
                    float(obs_info["c_start"][obs_info["cls_valid"]].mean().item()) if obs_info["cls_valid"].any() else 0.0
                )
                debug_state.setdefault("oabs_c_end_mean_per_sample", []).append(
                    float(obs_info["c_end"][obs_info["cls_valid"]].mean().item()) if obs_info["cls_valid"].any() else 0.0
                )
                visible_reg_mask = obs_info["reg_valid"]
                debug_state.setdefault("oabs_visible_ratio_mean_per_sample", []).append(
                    float(obs_info["visible_ratio"][visible_reg_mask].mean().item()) if visible_reg_mask.any() else 0.0
                )
                debug_state.setdefault("oabs_visible_ratio_min_per_sample", []).append(
                    float(obs_info["visible_ratio"][visible_reg_mask].min().item()) if visible_reg_mask.any() else 0.0
                )
                debug_state.setdefault("oabs_visible_len_mean_per_sample", []).append(
                    float(obs_info["visible_lengths"][visible_reg_mask].mean().item()) if visible_reg_mask.any() else 0.0
                )
                debug_state.setdefault("oabs_gt_len_mean_per_sample", []).append(
                    float(obs_info["gt_lengths"][obs_info["cls_valid"]].mean().item()) if obs_info["cls_valid"].any() else 0.0
                )
                if candidate_mask is not None:
                    debug_state.setdefault("oabs_candidate_points_per_sample", []).append(
                        int(candidate_mask.any(dim=1).sum().item())
                    )
                else:
                    debug_state.setdefault("oabs_candidate_points_per_sample", []).append(0)
                if self.oaa_enabled and base_candidate_mask is not None:
                    debug_state.setdefault("oaa_candidate_points_pre_gate_per_sample", []).append(
                        int(base_candidate_mask.any(dim=1).sum().item())
                    )
                    debug_state.setdefault("oaa_candidate_points_post_gate_per_sample", []).append(
                        int(candidate_mask.any(dim=1).sum().item())
                    )
                    debug_state.setdefault("oaa_gated_gt_fallback_per_sample", []).append(
                        int(oaa_missing_gt.sum().item()) if oaa_missing_gt is not None else 0
                    )
                    if oaa_overlap is not None and base_candidate_mask.any():
                        valid_overlap = oaa_overlap[base_candidate_mask]
                        debug_state.setdefault("oaa_overlap_mean_per_sample", []).append(
                            float(valid_overlap.mean().item()) if valid_overlap.numel() > 0 else 0.0
                        )
                        debug_state.setdefault("oaa_overlap_max_per_sample", []).append(
                            float(valid_overlap.max().item()) if valid_overlap.numel() > 0 else 0.0
                        )
                    else:
                        debug_state.setdefault("oaa_overlap_mean_per_sample", []).append(0.0)
                        debug_state.setdefault("oaa_overlap_max_per_sample", []).append(0.0)
                    for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                        debug_state.setdefault(f"oaa_level{level_idx}_candidate_pre_gate_per_sample", []).append(
                            int(base_candidate_mask[start_idx:end_idx].any(dim=1).sum().item())
                        )
                        debug_state.setdefault(f"oaa_level{level_idx}_candidate_post_gate_per_sample", []).append(
                            int(candidate_mask[start_idx:end_idx].any(dim=1).sum().item())
                        )
                if total_cost is not None and torch.isfinite(total_cost).any():
                    finite_cost = total_cost[torch.isfinite(total_cost)]
                    debug_state.setdefault("oabs_cost_min_per_sample", []).append(float(finite_cost.min().item()))
                    debug_state.setdefault("oabs_cost_max_per_sample", []).append(float(finite_cost.max().item()))
                else:
                    debug_state.setdefault("oabs_cost_min_per_sample", []).append(float("inf"))
                    debug_state.setdefault("oabs_cost_max_per_sample", []).append(float("inf"))
                debug_state.setdefault("oabs_multi_gt_points_per_sample", []).append(int(multi_gt_points))
                for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                    debug_state.setdefault(f"oabs_level{level_idx}_pos_per_sample", []).append(
                        int(positive_mask[start_idx:end_idx].sum().item())
                    )

        return gt_cls, gt_reg, reg_weight_list, obs_infos, debug_state

    @torch.no_grad()
    def prepare_boundary_targets_oabs(self, points, mask_list, temporal_grid_list, gt_segments, obs_infos):
        concat_points = torch.cat(points, dim=1)
        concat_valid = torch.cat(mask_list, dim=1).bool()
        concat_fresh = torch.cat([grid["fresh_mask"] for grid in temporal_grid_list], dim=1).bool()
        boundary_targets = []
        boundary_weights = []
        debug_state = {}

        for point, valid_mask, fresh_mask, gt_segment, obs_info in zip(
            concat_points, concat_valid, concat_fresh, gt_segments, obs_infos
        ):
            num_pts = int(point.shape[0])
            target = point.new_zeros((num_pts, 2))
            weight = point.new_zeros((num_pts, 2))

            eligible = obs_info["cls_valid"]
            if gt_segment.shape[0] > 0 and eligible.any():
                raw_segments = gt_segment[eligible]
                center_t = point[:, 0, None]
                sigma_start = (self.oabs_boundary_gamma * obs_info["delta_start"][eligible])[None, :].clamp_min(
                    self.reg_denom_floor
                )
                sigma_end = (self.oabs_boundary_gamma * obs_info["delta_end"][eligible])[None, :].clamp_min(
                    self.reg_denom_floor
                )

                start_per_gt = torch.exp(-(center_t - raw_segments[None, :, 0]).abs() / sigma_start)
                end_per_gt = torch.exp(-(center_t - raw_segments[None, :, 1]).abs() / sigma_end)

                if self.oabs_observed_only:
                    observed_gate = fresh_mask[:, None].to(point.dtype)
                    start_per_gt = start_per_gt * observed_gate
                    end_per_gt = end_per_gt * observed_gate

                start_target, start_idx = start_per_gt.max(dim=1)
                end_target, end_idx = end_per_gt.max(dim=1)
                start_weight = obs_info["c_start"][eligible][start_idx]
                end_weight = obs_info["c_end"][eligible][end_idx]

                if self.oabs_observed_only:
                    fresh = fresh_mask.to(point.dtype)
                    start_weight = start_weight * fresh
                    end_weight = end_weight * fresh

                target = torch.stack([start_target, end_target], dim=-1)
                weight = torch.stack([start_weight, end_weight], dim=-1)
                target = target * valid_mask[:, None].to(point.dtype)
                weight = weight * valid_mask[:, None].to(point.dtype)

            boundary_targets.append(target)
            boundary_weights.append(weight)

            if self.debug_enabled:
                weighted_target = target * weight
                debug_state.setdefault("oabs_boundary_target_mass_per_sample", []).append(float(target.sum().item()))
                debug_state.setdefault("oabs_boundary_weighted_mass_per_sample", []).append(
                    float(weighted_target.sum().item())
                )
                debug_state.setdefault("oabs_boundary_weight_mean_per_sample", []).append(
                    float(weight[valid_mask].mean().item()) if valid_mask.any() else 0.0
                )
                debug_state.setdefault("oabs_boundary_fresh_points_per_sample", []).append(int(fresh_mask.sum().item()))

        return boundary_targets, boundary_weights, debug_state

    def losses(self, cls_pred, reg_pred, boundary_pred, mask_list, points, temporal_grid_list, gt_segments, gt_labels):
        raw_gt_segments = gt_segments
        gt_cls, gt_reg, reg_weight, obs_infos, target_debug = self.prepare_targets(
            points, temporal_grid_list, gt_segments, gt_labels
        )

        gt_cls = torch.stack(gt_cls)
        gt_reg = torch.stack(gt_reg)
        reg_weight = torch.stack(reg_weight)
        valid_mask = torch.cat(mask_list, dim=1)

        pos_weight = gt_cls.max(dim=-1).values * valid_mask.to(gt_cls.dtype)
        pos_mass = float(pos_weight.sum().item())
        pos_count = int(torch.logical_and(gt_cls.max(dim=-1).values > 0, valid_mask).sum().item())

        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(pos_mass, 1.0)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(pos_mass, 1.0)

        cls_pred_flat = [tensor.permute(0, 2, 1) for tensor in cls_pred]
        cls_pred_flat = torch.cat(cls_pred_flat, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        gt_target = gt_target * (1 - self.label_smoothing)
        gt_target = gt_target + self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred_flat, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg_split = gt_reg.permute(0, 2, 1).split(split_size, dim=-1)
        pred_segments = self.get_refined_proposals(points, reg_pred)
        gt_segments = self.get_refined_proposals(points, gt_reg_split)

        reg_mask = torch.logical_and(reg_weight > 0, valid_mask)
        if reg_mask.any():
            reg_loss_raw = self.reg_loss(pred_segments[reg_mask], gt_segments[reg_mask], reduction="none").reshape(-1)
            reg_loss = (reg_loss_raw * reg_weight[reg_mask]).sum()
            reg_loss /= loss_normalizer
        else:
            reg_loss = pred_segments.sum() * 0

        if self.loss_weight > 0:
            reg_loss_weight = self.loss_weight
        else:
            reg_loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        boundary_loss = cls_loss.new_zeros(())
        boundary_mass = 0.0
        boundary_pos_count = 0
        boundary_debug = {}
        if self.use_boundary_aux and boundary_pred and self.boundary_loss is not None:
            boundary_pred_flat = [tensor.permute(0, 2, 1) for tensor in boundary_pred]
            boundary_pred_flat = torch.cat(boundary_pred_flat, dim=1)

            if self.oabs_mode == "full":
                boundary_targets, boundary_weights, boundary_debug = self.prepare_boundary_targets_oabs(
                    points,
                    mask_list,
                    temporal_grid_list,
                    raw_gt_segments,
                    obs_infos,
                )
                boundary_targets = torch.stack(boundary_targets)
                boundary_weights = torch.stack(boundary_weights)
                boundary_loss_raw = self.boundary_loss(
                    boundary_pred_flat[valid_mask],
                    boundary_targets[valid_mask],
                    reduction="none",
                )
                boundary_loss = (boundary_loss_raw * boundary_weights[valid_mask]).sum()
                boundary_mass_tensor = (
                    boundary_targets * boundary_weights * valid_mask.unsqueeze(-1).to(boundary_targets.dtype)
                )
                boundary_mass = float(boundary_mass_tensor.sum().item())
                boundary_pos_count = int(
                    torch.logical_and((boundary_targets * boundary_weights).max(dim=-1).values > 0.5, valid_mask).sum().item()
                )
            else:
                boundary_targets = torch.stack(super().prepare_boundary_targets(points, raw_gt_segments))
                boundary_loss = self.boundary_loss(
                    boundary_pred_flat[valid_mask],
                    boundary_targets[valid_mask],
                    reduction="sum",
                )
                boundary_mass_tensor = boundary_targets.max(dim=-1).values * valid_mask.to(boundary_targets.dtype)
                boundary_mass = float(boundary_mass_tensor.sum().item())
                boundary_pos_count = int(
                    torch.logical_and(boundary_targets.max(dim=-1).values > 0.5, valid_mask).sum().item()
                )

            boundary_loss /= max(boundary_mass, 1.0)

        losses = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss * reg_loss_weight,
        }
        if self.use_boundary_aux:
            losses["boundary_loss"] = boundary_loss * self.boundary_loss_weight

        if self.debug_enabled:
            debug_state = dict(target_debug)
            debug_state.update(boundary_debug)
            debug_state["head_v2_pos_mass_total"] = pos_mass
            debug_state["head_v2_positive_count_total"] = pos_count
            debug_state["head_v2_pos_mass_to_count_ratio"] = float(pos_mass / max(pos_count, 1))
            debug_state["head_v2_valid_points_total"] = int(valid_mask.sum().item())
            debug_state["head_v2_reg_points_total"] = int(reg_mask.sum().item())
            debug_state["head_v2_loss_normalizer"] = float(
                loss_normalizer.item() if torch.is_tensor(loss_normalizer) else loss_normalizer
            )
            debug_state["head_v3_boundary_mass_total"] = boundary_mass
            debug_state["head_v3_boundary_positive_count_total"] = boundary_pos_count
            self._latest_debug_state = debug_state

        return losses
