import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale


@HEADS.register_module()
class IrregularActionFormerHeadV2(nn.Module):
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
        debug_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.predictor_kernel_size = predictor_kernel_size
        self.soft_assign_topk = soft_assign_topk
        self.soft_assign_temperature = soft_assign_temperature
        self.soft_center_cost_weight = soft_center_cost_weight
        self.soft_scale_cost_weight = soft_scale_cost_weight
        self.reg_denom_floor = reg_denom_floor
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(float(loss_normalizer)))

        debug_cfg = {} if debug_cfg is None else dict(debug_cfg)
        self.debug_enabled = bool(debug_cfg.get("enable", False))
        self._latest_debug_state = {}

        self.prior_generator = build_prior_generator(prior_generator)
        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for idx in range(self.num_convs):
            in_channels = self.in_channels if idx == 0 else self.feat_channels
            self.cls_convs.append(
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
            self.reg_convs.append(
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

        padding = self.predictor_kernel_size // 2
        self.cls_head = nn.Conv1d(
            self.feat_channels,
            self.num_classes,
            kernel_size=self.predictor_kernel_size,
            padding=padding,
        )
        self.reg_head = nn.Conv1d(
            self.feat_channels,
            2,
            kernel_size=self.predictor_kernel_size,
            padding=padding,
        )
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def _forward_single_level(self, feat, mask, level_idx):
        cls_feat = feat
        reg_feat = feat
        branch_mask = mask
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat, _ = cls_conv(cls_feat, branch_mask)
            reg_feat, _ = reg_conv(reg_feat, branch_mask)

        cls_pred = self.cls_head(cls_feat)
        reg_pred = F.relu(self.scale[level_idx](self.reg_head(reg_feat)))
        return cls_pred, reg_pred

    def forward_train(self, feat_list, mask_list, temporal_grid_list, gt_segments, gt_labels, **kwargs):
        cls_pred = []
        reg_pred = []
        for level_idx, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_out, reg_out = self._forward_single_level(feat, mask, level_idx)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)

        points = self.prior_generator(feat_list, temporal_grid_list)
        return self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)

    def forward_test(self, feat_list, mask_list, temporal_grid_list, **kwargs):
        cls_pred = []
        reg_pred = []
        for level_idx, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_out, reg_out = self._forward_single_level(feat, mask, level_idx)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)

        points = self.prior_generator(feat_list, temporal_grid_list)
        return self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)

    def get_refined_proposals(self, points, reg_pred):
        point_tensor = torch.cat(points, dim=1)
        reg_tensor = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)
        left_denom = point_tensor[:, :, 3].clamp_min(self.reg_denom_floor)
        right_denom = point_tensor[:, :, 4].clamp_min(self.reg_denom_floor)
        left = torch.expm1(reg_tensor[:, :, 0].clamp_min(0.0)) * left_denom
        right = torch.expm1(reg_tensor[:, :, 1].clamp_min(0.0)) * right_denom
        start = point_tensor[:, :, 0] - left
        end = point_tensor[:, :, 0] + right
        return torch.stack((start, end), dim=-1)

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list):
        proposals = self.get_refined_proposals(points, reg_pred)
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()
        masks = torch.cat(mask_list, dim=1)

        new_proposals = []
        new_scores = []
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])
            new_scores.append(score[mask])
        return new_proposals, new_scores

    def collect_debug_state(self):
        return dict(self._latest_debug_state)

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
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

        cls_pred = [tensor.permute(0, 2, 1) for tensor in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        gt_target = gt_target * (1 - self.label_smoothing)
        gt_target = gt_target + self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
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
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

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
            self._latest_debug_state = debug_state

        return {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}

    def _build_candidate_mask(self, point, gt_segs, reg_targets):
        center_t = point[:, 0, None]
        inside_gt_seg = reg_targets.min(dim=-1).values > 0
        if self.center_sample != "radius":
            return inside_gt_seg

        center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
        radius_base = (point[:, 3, None] * point[:, 4, None]).clamp_min(self.reg_denom_floor**2).sqrt()
        radius = self.center_sample_radius * radius_base
        t_mins = center_pts - radius
        t_maxs = center_pts + radius
        cb_left = center_t - torch.maximum(t_mins, gt_segs[:, :, 0])
        cb_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - center_t
        center_seg = torch.stack((cb_left, cb_right), dim=-1)
        candidate_mask = center_seg.min(dim=-1).values > 0

        missing_gt = ~candidate_mask.any(dim=0)
        if missing_gt.any():
            candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]
        return candidate_mask

    def _build_assignment_weights(self, point, gt_segment, candidate_mask):
        center_t = point[:, 0, None]
        point_scale = (point[:, 3, None] + point[:, 4, None]).clamp_min(self.reg_denom_floor)
        gt_center = 0.5 * (gt_segment[:, 0] + gt_segment[:, 1])[None, :]
        gt_len = (gt_segment[:, 1] - gt_segment[:, 0])[None, :].clamp_min(self.reg_denom_floor)

        center_cost = (center_t - gt_center).abs() / (0.5 * gt_len + 0.5 * point_scale).clamp_min(self.reg_denom_floor)
        scale_cost = torch.abs(torch.log(gt_len / point_scale))
        total_cost = self.soft_center_cost_weight * center_cost + self.soft_scale_cost_weight * scale_cost
        total_cost = total_cost.masked_fill(~candidate_mask, float("inf"))

        num_pts, num_gts = total_cost.shape
        topk = min(self.soft_assign_topk, num_pts)
        trans_cost = total_cost.transpose(0, 1)
        topk_cost, topk_idx = torch.topk(trans_cost, k=topk, dim=1, largest=False)
        valid_topk = torch.isfinite(topk_cost)

        weights = total_cost.new_zeros(num_pts, num_gts)
        if valid_topk.any():
            quality = torch.exp(-topk_cost / max(self.soft_assign_temperature, 1e-6))
            quality = torch.where(valid_topk, quality, quality.new_zeros(1))
            quality = quality / quality.max(dim=1, keepdim=True).values.clamp_min(1e-6)

            gt_index = torch.arange(num_gts, device=point.device)[:, None].expand_as(topk_idx)
            weights[topk_idx[valid_topk], gt_index[valid_topk]] = quality[valid_topk]
        return weights, total_cost

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=1)
        gt_cls = []
        gt_reg = []
        reg_weight_list = []
        debug_state = {}

        level_lengths = [int(level.shape[1]) for level in points]
        level_offsets = []
        offset = 0
        for length in level_lengths:
            level_offsets.append((offset, offset + length))
            offset += length

        for point, gt_segment, gt_label in zip(concat_points, gt_segments, gt_labels):
            num_pts = point.shape[0]
            num_gts = gt_segment.shape[0]

            if num_gts == 0:
                gt_cls.append(gt_segment.new_zeros((num_pts, self.num_classes)))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                reg_weight_list.append(gt_segment.new_zeros((num_pts,)))
                continue

            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            center_t = point[:, 0, None]
            left = center_t - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - center_t
            reg_targets = torch.stack((left, right), dim=-1)

            candidate_mask = self._build_candidate_mask(point, gt_segs, reg_targets)
            assign_weights, total_cost = self._build_assignment_weights(point, gt_segment, candidate_mask)

            one_hot = F.one_hot(gt_label.long(), self.num_classes).to(assign_weights.dtype)
            weighted_labels = assign_weights[:, :, None] * one_hot[None, :, :]
            cls_targets = weighted_labels.max(dim=1).values
            cls_targets.clamp_(min=0.0, max=1.0)

            denom_left = point[:, 3, None].clamp_min(self.reg_denom_floor)
            denom_right = point[:, 4, None].clamp_min(self.reg_denom_floor)
            reg_targets_log = torch.stack(
                [
                    torch.log1p((left / denom_left).clamp_min(0.0)),
                    torch.log1p((right / denom_right).clamp_min(0.0)),
                ],
                dim=-1,
            )

            reg_weight, best_gt_idx = assign_weights.max(dim=1)
            reg_target = reg_targets_log[torch.arange(num_pts, device=point.device), best_gt_idx]
            reg_target = torch.where(reg_weight[:, None] > 0, reg_target, reg_target.new_zeros(reg_target.shape))

            gt_cls.append(cls_targets)
            gt_reg.append(reg_target)
            reg_weight_list.append(reg_weight)

            if self.debug_enabled:
                positive_mask = cls_targets.max(dim=-1).values > 0
                multi_gt_mask = (assign_weights > 0).sum(dim=1) > 1
                debug_state.setdefault("head_v2_num_gt_per_sample", []).append(int(num_gts))
                debug_state.setdefault("head_v2_candidate_points_per_sample", []).append(int(candidate_mask.any(dim=1).sum().item()))
                debug_state.setdefault("head_v2_positive_points_per_sample", []).append(int(positive_mask.sum().item()))
                debug_state.setdefault("head_v2_multi_gt_points_per_sample", []).append(int(multi_gt_mask.sum().item()))
                debug_state.setdefault("head_v2_reg_weight_sum_per_sample", []).append(float(reg_weight.sum().item()))
                debug_state.setdefault("head_v2_reg_weight_absmax_per_sample", []).append(float(reg_weight.max().item()))
                if torch.isfinite(total_cost).any():
                    finite_cost = total_cost[torch.isfinite(total_cost)]
                    debug_state.setdefault("head_v2_cost_min_per_sample", []).append(float(finite_cost.min().item()))
                    debug_state.setdefault("head_v2_cost_max_per_sample", []).append(float(finite_cost.max().item()))
                else:
                    debug_state.setdefault("head_v2_cost_min_per_sample", []).append(float("inf"))
                    debug_state.setdefault("head_v2_cost_max_per_sample", []).append(float("inf"))

                if positive_mask.any():
                    pos_reg = reg_target[positive_mask]
                    debug_state.setdefault("head_v2_reg_absmax_per_sample", []).append(float(pos_reg.abs().max().item()))
                else:
                    debug_state.setdefault("head_v2_reg_absmax_per_sample", []).append(0.0)

                for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                    level_pos = positive_mask[start_idx:end_idx]
                    debug_state.setdefault(f"head_v2_level{level_idx}_pos_per_sample", []).append(
                        int(level_pos.sum().item())
                    )

        return gt_cls, gt_reg, reg_weight_list, debug_state
