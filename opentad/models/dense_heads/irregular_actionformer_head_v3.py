import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_loss
from ..bricks import ConvModule
from .irregular_actionformer_head_v2 import IrregularActionFormerHeadV2


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@HEADS.register_module()
class IrregularActionFormerHeadV3(IrregularActionFormerHeadV2):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        predictor_kernel_size=1,
        soft_assign_topk=9,
        soft_assign_temperature=1.0,
        soft_center_cost_weight=1.0,
        soft_scale_cost_weight=0.5,
        reg_denom_floor=0.5,
        geometry_hidden_channels=128,
        geometry_scale=0.25,
        boundary_loss_weight=0.2,
        boundary_sigma=1.0,
        boundary_num_convs=1,
        boundary_predictor_kernel_size=1,
        boundary_inference=None,
        debug_cfg=None,
    ):
        self.geometry_hidden_channels = geometry_hidden_channels
        self.geometry_scale = geometry_scale
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_sigma = boundary_sigma
        self.boundary_num_convs = boundary_num_convs
        self.boundary_predictor_kernel_size = boundary_predictor_kernel_size
        self.use_boundary_aux = boundary_loss_weight > 0
        self.boundary_inference = self._build_boundary_inference_cfg(boundary_inference)
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_convs=num_convs,
            prior_generator=prior_generator,
            loss=loss,
            loss_normalizer=loss_normalizer,
            loss_normalizer_momentum=loss_normalizer_momentum,
            center_sample=center_sample,
            center_sample_radius=center_sample_radius,
            label_smoothing=label_smoothing,
            cls_prior_prob=cls_prior_prob,
            loss_weight=loss_weight,
            predictor_kernel_size=predictor_kernel_size,
            soft_assign_topk=soft_assign_topk,
            soft_assign_temperature=soft_assign_temperature,
            soft_center_cost_weight=soft_center_cost_weight,
            soft_scale_cost_weight=soft_scale_cost_weight,
            reg_denom_floor=reg_denom_floor,
            debug_cfg=debug_cfg,
        )
        if self.use_boundary_aux:
            boundary_loss_cfg = _cfg_get(loss, "boundary_loss", None)
            if boundary_loss_cfg is None:
                raise ValueError(
                    "IrregularActionFormerHeadV3 requires loss.boundary_loss when boundary_loss_weight > 0."
                )
            self.boundary_loss = build_loss(boundary_loss_cfg)
        else:
            self.boundary_loss = None

    def _init_layers(self):
        super()._init_layers()

        self.geom_channels = 5
        self.geometry_encoder = nn.Sequential(
            nn.Conv1d(self.geom_channels, self.geometry_hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.geometry_hidden_channels, self.feat_channels * 2, kernel_size=1),
        )

        if self.use_boundary_aux:
            self.boundary_convs = nn.ModuleList()
            for idx in range(self.boundary_num_convs):
                in_channels = self.in_channels if idx == 0 else self.feat_channels
                self.boundary_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type="LN"),
                        act_cfg=dict(type="relu"),
                    )
                )

            padding = self.boundary_predictor_kernel_size // 2
            self.boundary_head = nn.Conv1d(
                self.feat_channels,
                2,
                kernel_size=self.boundary_predictor_kernel_size,
                padding=padding,
            )
            if self.cls_prior_prob > 0:
                bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
                nn.init.constant_(self.boundary_head.bias, bias_value)
        else:
            self.boundary_convs = None

    def _build_geometry_features(self, temporal_grid):
        cell_left = temporal_grid["cell_left"].clamp_min(self.reg_denom_floor)
        cell_right = temporal_grid["cell_right"].clamp_min(self.reg_denom_floor)
        point_scale = (cell_left + cell_right).clamp_min(self.reg_denom_floor)
        level_scale = temporal_grid.get("level_scale", point_scale.mean(dim=1))
        if level_scale.dim() == 1:
            level_scale = level_scale[:, None]
        level_scale = level_scale.clamp_min(self.reg_denom_floor)

        fresh = temporal_grid["fresh_mask"].to(point_scale.dtype)
        valid = temporal_grid["valid_mask"].to(point_scale.dtype)
        log_point_scale = torch.log(point_scale)
        log_scale_ratio = torch.log((point_scale / level_scale).clamp_min(1e-6))
        log_asymmetry = torch.log((cell_right / cell_left).clamp_min(1e-6))
        symmetry = (2.0 * torch.minimum(cell_left, cell_right) / point_scale).clamp(0.0, 1.0)

        geom = torch.stack([fresh, log_point_scale, log_scale_ratio, log_asymmetry, symmetry], dim=1)
        geom = geom * valid.unsqueeze(1)
        return geom

    def _apply_geometry_modulation(self, feat, mask, temporal_grid):
        geom = self._build_geometry_features(temporal_grid)
        gamma_beta = self.geometry_encoder(geom)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        feat = feat * (1.0 + self.geometry_scale * torch.tanh(gamma))
        feat = feat + self.geometry_scale * beta
        feat = feat * mask.unsqueeze(1).to(feat.dtype)
        return feat

    def _forward_single_level(self, feat, mask, level_idx, temporal_grid):
        feat = self._apply_geometry_modulation(feat, mask, temporal_grid)

        cls_feat = feat
        reg_feat = feat
        branch_mask = mask
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat, _ = cls_conv(cls_feat, branch_mask)
            reg_feat, _ = reg_conv(reg_feat, branch_mask)

        cls_pred = self.cls_head(cls_feat)
        reg_pred = F.relu(self.scale[level_idx](self.reg_head(reg_feat)))

        boundary_pred = None
        if self.use_boundary_aux:
            boundary_feat = feat
            for boundary_conv in self.boundary_convs:
                boundary_feat, _ = boundary_conv(boundary_feat, branch_mask)
            boundary_pred = self.boundary_head(boundary_feat)

        return cls_pred, reg_pred, boundary_pred

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
        return self.losses(cls_pred, reg_pred, boundary_pred, mask_list, points, gt_segments, gt_labels)

    def _build_boundary_inference_cfg(self, boundary_inference):
        cfg = dict(
            enabled=False,
            peak_kernel=3,
            peak_ratio=0.5,
            score_thresh=0.05,
            preselect_topk_per_level=96,
            bank_topk=192,
            candidate_topk=12,
            stale_factor=0.6,
            min_symmetry=0.25,
            symmetry_power=0.5,
            window_scale_factor=2.5,
            window_length_factor=0.5,
            center_guard_factor=0.25,
            min_duration_factor=0.25,
            boundary_score_weight=1.0,
            distance_weight=1.0,
            duration_weight=0.75,
            center_weight=0.5,
            containment_weight=0.5,
            scale_weight=0.25,
            score_alpha=0.35,
            refine_strength_power=1.0,
        )
        if boundary_inference is not None:
            cfg.update(dict(boundary_inference))
        return cfg

    def _resolve_boundary_inference_cfg(self, infer_cfg=None):
        cfg = dict(self.boundary_inference)
        if infer_cfg is None:
            return cfg
        if isinstance(infer_cfg, dict):
            extra = infer_cfg.get("boundary_inference", None)
        else:
            extra = getattr(infer_cfg, "boundary_inference", None)
        if extra is not None:
            cfg.update(dict(extra))
        return cfg

    def _select_level_boundary_mask(self, score, valid_mask, cfg):
        valid_mask = valid_mask.bool()
        if not valid_mask.any():
            return valid_mask.new_zeros(valid_mask.shape)

        peak_kernel = max(int(cfg["peak_kernel"]), 1)
        if peak_kernel % 2 == 0:
            peak_kernel += 1

        masked_score = score.masked_fill(~valid_mask, -1e6)
        peak = masked_score[None, None] == F.max_pool1d(
            masked_score[None, None],
            kernel_size=peak_kernel,
            stride=1,
            padding=peak_kernel // 2,
        )
        peak = peak[0, 0] & valid_mask

        level_max = masked_score[valid_mask].max()
        dynamic_thresh = torch.maximum(level_max * float(cfg["peak_ratio"]), score.new_tensor(float(cfg["score_thresh"])))
        high = valid_mask & (score >= dynamic_thresh)
        candidate_mask = peak | high

        preselect_topk = int(cfg["preselect_topk_per_level"])
        if preselect_topk > 0 and int(candidate_mask.sum().item()) > preselect_topk:
            masked_candidate_score = score.masked_fill(~candidate_mask, -1e6)
            topk_idx = torch.topk(masked_candidate_score, k=preselect_topk, dim=0).indices
            reduced_mask = candidate_mask.new_zeros(candidate_mask.shape)
            reduced_mask[topk_idx] = True
            candidate_mask = reduced_mask & valid_mask

        if not candidate_mask.any():
            fallback_idx = masked_score.argmax()
            candidate_mask = valid_mask.new_zeros(valid_mask.shape)
            candidate_mask[fallback_idx] = True
        return candidate_mask

    def _collect_boundary_bank(self, points, boundary_pred, mask_list, temporal_grid_list, sample_idx, cfg):
        start_bank = dict(time=[], score=[], point_scale=[], side_scale=[], fresh=[], symmetry=[], level=[])
        end_bank = dict(time=[], score=[], point_scale=[], side_scale=[], fresh=[], symmetry=[], level=[])

        stale_factor = float(cfg["stale_factor"])
        symmetry_power = float(cfg["symmetry_power"])
        min_symmetry = float(cfg["min_symmetry"])

        for level_idx, (point_level, boundary_level, mask_level, temporal_grid) in enumerate(
            zip(points, boundary_pred, mask_list, temporal_grid_list)
        ):
            point = point_level[sample_idx]
            valid_mask = mask_level[sample_idx].bool()
            if not valid_mask.any():
                continue

            boundary_prob = boundary_level[sample_idx].sigmoid()
            fresh_mask = temporal_grid.get("fresh_mask", mask_level)[sample_idx].to(point.dtype)
            point_scale = (point[:, 3] + point[:, 4]).clamp_min(self.reg_denom_floor)
            symmetry = (
                2.0 * torch.minimum(point[:, 3], point[:, 4]) / point_scale
            ).clamp_min(min_symmetry).clamp_max(1.0)
            support = torch.where(
                fresh_mask > 0,
                torch.ones_like(fresh_mask),
                fresh_mask.new_full(fresh_mask.shape, stale_factor),
            )
            support = support * symmetry.pow(symmetry_power)

            start_score = boundary_prob[0] * support
            end_score = boundary_prob[1] * support
            start_mask = self._select_level_boundary_mask(start_score, valid_mask, cfg)
            end_mask = self._select_level_boundary_mask(end_score, valid_mask, cfg)

            for bank, select_mask, score_tensor, side_scale in (
                (start_bank, start_mask, start_score, point[:, 3]),
                (end_bank, end_mask, end_score, point[:, 4]),
            ):
                bank["time"].append(point[:, 0][select_mask])
                bank["score"].append(score_tensor[select_mask])
                bank["point_scale"].append(point_scale[select_mask])
                bank["side_scale"].append(side_scale[select_mask].clamp_min(self.reg_denom_floor))
                bank["fresh"].append(fresh_mask[select_mask])
                bank["symmetry"].append(symmetry[select_mask])
                bank["level"].append(point[:, 0][select_mask].new_full((int(select_mask.sum().item()),), level_idx))

        def _finalize_bank(bank):
            if len(bank["time"]) == 0:
                return dict(
                    time=torch.empty(0),
                    score=torch.empty(0),
                    point_scale=torch.empty(0),
                    side_scale=torch.empty(0),
                    fresh=torch.empty(0),
                    symmetry=torch.empty(0),
                    level=torch.empty(0, dtype=torch.long),
                )

            merged = {}
            for key, value_list in bank.items():
                merged[key] = torch.cat(value_list, dim=0)

            bank_topk = int(cfg["bank_topk"])
            if bank_topk > 0 and merged["score"].numel() > bank_topk:
                topk_idx = torch.topk(merged["score"], k=bank_topk, dim=0).indices
                for key in merged.keys():
                    merged[key] = merged[key][topk_idx]
            return merged

        return _finalize_bank(start_bank), _finalize_bank(end_bank)

    def _select_boundary_subset(self, bank, raw_boundary, raw_center, raw_length, raw_side_scale, side, cfg):
        if bank["time"].numel() == 0:
            return None

        raw_length = raw_length.clamp_min(self.reg_denom_floor)
        raw_side_scale = raw_side_scale.clamp_min(self.reg_denom_floor)
        search_window = torch.maximum(
            raw_side_scale * float(cfg["window_scale_factor"]),
            raw_length * float(cfg["window_length_factor"]),
        )
        center_guard = raw_length * float(cfg["center_guard_factor"])

        if side == "start":
            side_mask = bank["time"] <= raw_center + center_guard
            crossing = F.relu(bank["time"] - raw_center) / raw_length
        else:
            side_mask = bank["time"] >= raw_center - center_guard
            crossing = F.relu(raw_center - bank["time"]) / raw_length

        if not side_mask.any():
            side_mask = torch.ones_like(side_mask)

        distance = (bank["time"] - raw_boundary).abs() / search_window.clamp_min(self.reg_denom_floor)
        unary_quality = bank["score"] * torch.exp(
            -float(cfg["distance_weight"]) * distance
            -float(cfg["containment_weight"]) * crossing
        )
        unary_quality = unary_quality.masked_fill(~side_mask, -1.0)

        candidate_topk = min(int(cfg["candidate_topk"]), unary_quality.numel())
        if candidate_topk <= 0:
            return None

        topk_quality, topk_idx = torch.topk(unary_quality, k=candidate_topk, dim=0)
        valid_topk = topk_quality > -0.5
        if not valid_topk.any():
            nearest_idx = distance.argmin()
            topk_idx = nearest_idx[None]
        else:
            topk_idx = topk_idx[valid_topk]

        subset = {}
        for key, value in bank.items():
            subset[key] = value[topk_idx]
        subset["distance"] = distance[topk_idx]
        subset["unary_quality"] = bank["score"][topk_idx] * torch.exp(
            -float(cfg["distance_weight"]) * distance[topk_idx]
            -float(cfg["containment_weight"]) * crossing[topk_idx]
        )
        return subset

    def _boundary_pair_refine(self, proposal, point, start_subset, end_subset, cfg):
        if start_subset is None or end_subset is None:
            return proposal, proposal.new_tensor(1.0), proposal.new_zeros(())

        raw_start = proposal[0]
        raw_end = proposal[1]
        raw_length = (raw_end - raw_start).clamp_min(self.reg_denom_floor)
        raw_center = 0.5 * (raw_start + raw_end)
        point_center = point[0]

        start_time = start_subset["time"][:, None]
        end_time = end_subset["time"][None, :]
        pair_length = end_time - start_time

        min_pair_length = float(cfg["min_duration_factor"]) * torch.sqrt(
            start_subset["side_scale"][:, None] * end_subset["side_scale"][None, :]
        )
        valid_pair = pair_length > min_pair_length.clamp_min(self.reg_denom_floor)
        if not valid_pair.any():
            return proposal, proposal.new_tensor(1.0), proposal.new_zeros(())

        start_shift = (start_time - raw_start).abs() / torch.maximum(
            point[3] * float(cfg["window_scale_factor"]),
            raw_length * float(cfg["window_length_factor"]),
        ).clamp_min(self.reg_denom_floor)
        end_shift = (end_time - raw_end).abs() / torch.maximum(
            point[4] * float(cfg["window_scale_factor"]),
            raw_length * float(cfg["window_length_factor"]),
        ).clamp_min(self.reg_denom_floor)
        pair_center = 0.5 * (start_time + end_time)
        center_cost = (pair_center - raw_center).abs() / (0.5 * raw_length + 0.5 * (point[3] + point[4])).clamp_min(
            self.reg_denom_floor
        )
        duration_cost = torch.abs(torch.log((pair_length / raw_length).clamp_min(1e-6)))
        containment_cost = (
            F.relu(start_time - point_center) + F.relu(point_center - end_time)
        ) / raw_length.clamp_min(self.reg_denom_floor)
        scale_pair = torch.sqrt(start_subset["point_scale"][:, None] * end_subset["point_scale"][None, :]).clamp_min(
            self.reg_denom_floor
        )
        scale_cost = torch.abs(torch.log((raw_length / scale_pair).clamp_min(1e-6)))

        boundary_conf = torch.sqrt(start_subset["score"][:, None] * end_subset["score"][None, :]).clamp(0.0, 1.0)
        geom_quality = torch.exp(
            -float(cfg["distance_weight"]) * (start_shift + end_shift)
            -float(cfg["duration_weight"]) * duration_cost
            -float(cfg["center_weight"]) * center_cost
            -float(cfg["containment_weight"]) * containment_cost
            -float(cfg["scale_weight"]) * scale_cost
        )
        pair_quality = boundary_conf.pow(float(cfg["boundary_score_weight"])) * geom_quality
        pair_quality = pair_quality.masked_fill(~valid_pair, -1.0)

        flat_idx = pair_quality.view(-1).argmax()
        best_quality = pair_quality.view(-1)[flat_idx].clamp_min(0.0)
        if float(best_quality.item()) <= 0:
            return proposal, proposal.new_tensor(1.0), proposal.new_zeros(())

        start_idx = torch.div(flat_idx, pair_quality.shape[1], rounding_mode="floor")
        end_idx = torch.fmod(flat_idx, pair_quality.shape[1])
        snapped = proposal.new_tensor([start_subset["time"][start_idx], end_subset["time"][end_idx]])

        refine_strength = best_quality.pow(float(cfg["refine_strength_power"])).clamp(0.0, 1.0)
        refined = proposal + refine_strength * (snapped - proposal)
        refined_start = torch.minimum(refined[0], refined[1] - self.reg_denom_floor)
        refined_end = torch.maximum(refined[1], refined_start + self.reg_denom_floor)
        refined = torch.stack([refined_start, refined_end])
        shift = (refined - proposal).abs().sum()
        return refined, best_quality, shift

    def _boundary_aware_inference(self, points, reg_pred, cls_pred, boundary_pred, mask_list, temporal_grid_list, cfg):
        raw_proposals = self.get_refined_proposals(points, reg_pred)
        raw_scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()
        point_tensor = torch.cat(points, dim=1)
        valid_mask = torch.cat(mask_list, dim=1)

        refined_proposals = []
        refined_scores = []
        debug_entries = []

        for sample_idx, sample_mask in enumerate(valid_mask):
            sample_points = point_tensor[sample_idx][sample_mask]
            sample_proposals = raw_proposals[sample_idx][sample_mask]
            sample_scores = raw_scores[sample_idx][sample_mask]
            start_bank, end_bank = self._collect_boundary_bank(
                points, boundary_pred, mask_list, temporal_grid_list, sample_idx, cfg
            )

            if sample_points.shape[0] == 0:
                refined_proposals.append(sample_proposals)
                refined_scores.append(sample_scores)
                debug_entries.append(dict(start_candidates=0, end_candidates=0, quality_mean=0.0, shift_mean=0.0))
                continue

            sample_refined = []
            sample_quality = []
            sample_shift = []
            for proposal, point in zip(sample_proposals, sample_points):
                start_subset = self._select_boundary_subset(
                    start_bank,
                    raw_boundary=proposal[0],
                    raw_center=0.5 * (proposal[0] + proposal[1]),
                    raw_length=(proposal[1] - proposal[0]).clamp_min(self.reg_denom_floor),
                    raw_side_scale=point[3],
                    side="start",
                    cfg=cfg,
                )
                end_subset = self._select_boundary_subset(
                    end_bank,
                    raw_boundary=proposal[1],
                    raw_center=0.5 * (proposal[0] + proposal[1]),
                    raw_length=(proposal[1] - proposal[0]).clamp_min(self.reg_denom_floor),
                    raw_side_scale=point[4],
                    side="end",
                    cfg=cfg,
                )
                refined, quality, shift = self._boundary_pair_refine(proposal, point, start_subset, end_subset, cfg)
                sample_refined.append(refined)
                sample_quality.append(quality)
                sample_shift.append(shift)

            sample_refined = torch.stack(sample_refined, dim=0)
            sample_quality = torch.stack(sample_quality, dim=0).clamp_min(1e-6).clamp_max(1.0)
            sample_shift = torch.stack(sample_shift, dim=0)
            alpha = float(cfg["score_alpha"])
            sample_scores = sample_scores.clamp_min(1e-6).pow(1.0 - alpha) * sample_quality[:, None].pow(alpha)

            refined_proposals.append(sample_refined)
            refined_scores.append(sample_scores)
            debug_entries.append(
                dict(
                    start_candidates=int(start_bank["time"].numel()),
                    end_candidates=int(end_bank["time"].numel()),
                    quality_mean=float(sample_quality.mean().item()),
                    shift_mean=float(sample_shift.mean().item()),
                    shift_max=float(sample_shift.max().item()),
                )
            )

        if self.debug_enabled:
            self._latest_debug_state = dict(
                self._latest_debug_state,
                head_v3_boundary_infer_enabled=True,
                head_v3_boundary_infer_start_candidates_per_sample=[entry["start_candidates"] for entry in debug_entries],
                head_v3_boundary_infer_end_candidates_per_sample=[entry["end_candidates"] for entry in debug_entries],
                head_v3_boundary_infer_quality_mean_per_sample=[entry["quality_mean"] for entry in debug_entries],
                head_v3_boundary_infer_shift_mean_per_sample=[entry["shift_mean"] for entry in debug_entries],
                head_v3_boundary_infer_shift_max_per_sample=[entry.get("shift_max", 0.0) for entry in debug_entries],
            )

        return refined_proposals, refined_scores

    def forward_test(self, feat_list, mask_list, temporal_grid_list, infer_cfg=None, **kwargs):
        cls_pred = []
        reg_pred = []
        boundary_pred = []
        for level_idx, (feat, mask, temporal_grid) in enumerate(zip(feat_list, mask_list, temporal_grid_list)):
            cls_out, reg_out, boundary_out = self._forward_single_level(feat, mask, level_idx, temporal_grid)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)
            boundary_pred.append(boundary_out)

        points = self.prior_generator(feat_list, temporal_grid_list)
        boundary_cfg = self._resolve_boundary_inference_cfg(infer_cfg)
        if not boundary_cfg.get("enabled", False) or not self.use_boundary_aux:
            return self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)

        return self._boundary_aware_inference(points, reg_pred, cls_pred, boundary_pred, mask_list, temporal_grid_list, boundary_cfg)

    @torch.no_grad()
    def prepare_boundary_targets(self, points, gt_segments):
        concat_points = torch.cat(points, dim=1)
        boundary_targets = []

        for point, gt_segment in zip(concat_points, gt_segments):
            num_pts = point.shape[0]
            num_gts = gt_segment.shape[0]
            if num_gts == 0:
                boundary_targets.append(point.new_zeros((num_pts, 2)))
                continue

            center_t = point[:, 0, None]
            sigma_start = self.boundary_sigma * point[:, 3].clamp_min(self.reg_denom_floor)[:, None]
            sigma_end = self.boundary_sigma * point[:, 4].clamp_min(self.reg_denom_floor)[:, None]

            start_dist = (center_t - gt_segment[None, :, 0]).abs()
            end_dist = (center_t - gt_segment[None, :, 1]).abs()

            start_target = torch.exp(-start_dist / sigma_start)
            end_target = torch.exp(-end_dist / sigma_end)

            boundary_targets.append(
                torch.stack([start_target.max(dim=1).values, end_target.max(dim=1).values], dim=-1)
            )
        return boundary_targets

    def losses(self, cls_pred, reg_pred, boundary_pred, mask_list, points, gt_segments, gt_labels):
        raw_gt_segments = gt_segments
        gt_cls, gt_reg, reg_weight, target_debug = self.prepare_targets(points, gt_segments, gt_labels)

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
        if self.use_boundary_aux and boundary_pred and self.boundary_loss is not None:
            boundary_targets = torch.stack(self.prepare_boundary_targets(points, raw_gt_segments))
            boundary_pred_flat = [tensor.permute(0, 2, 1) for tensor in boundary_pred]
            boundary_pred_flat = torch.cat(boundary_pred_flat, dim=1)
            boundary_mass_tensor = boundary_targets.max(dim=-1).values * valid_mask.to(boundary_targets.dtype)
            boundary_mass = float(boundary_mass_tensor.sum().item())
            boundary_pos_count = int(torch.logical_and(boundary_targets.max(dim=-1).values > 0.5, valid_mask).sum().item())

            boundary_loss = self.boundary_loss(
                boundary_pred_flat[valid_mask],
                boundary_targets[valid_mask],
                reduction="sum",
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
