import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_loss, build_prior_generator
from ..bricks import ConvModule, Scale


@HEADS.register_module()
class IrregularActionFormerBridgeHead(nn.Module):
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
        tower_kernel_size=3,
        predictor_kernel_size=3,
        assignment_mode="hard",
        regression_mode="symmetric_linear",
        soft_assign_topk=9,
        soft_assign_temperature=1.0,
        soft_center_cost_weight=1.0,
        soft_scale_cost_weight=0.5,
        soft_loss_normalizer_mode="pos_mass",
        soft_reg_weight_mode="soft",
        soft_cls_target_mode="soft",
        reg_denom_floor=0.5,
        filter_similar_gt=True,
        cls_loss_weight=1.0,
        reg_loss_weight=None,
        detach_cls_input_from_backbone=False,
        detach_reg_input_from_backbone=False,
        cls_loss_weight_schedule=None,
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
        self.tower_kernel_size = tower_kernel_size
        self.predictor_kernel_size = predictor_kernel_size
        self.assignment_mode = assignment_mode
        self.regression_mode = regression_mode
        self.soft_assign_topk = soft_assign_topk
        self.soft_assign_temperature = soft_assign_temperature
        self.soft_center_cost_weight = soft_center_cost_weight
        self.soft_scale_cost_weight = soft_scale_cost_weight
        self.soft_loss_normalizer_mode = soft_loss_normalizer_mode
        self.soft_reg_weight_mode = soft_reg_weight_mode
        self.soft_cls_target_mode = soft_cls_target_mode
        self.reg_denom_floor = reg_denom_floor
        self.filter_similar_gt = filter_similar_gt
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.detach_cls_input_from_backbone = detach_cls_input_from_backbone
        self.detach_reg_input_from_backbone = detach_reg_input_from_backbone
        self.cls_loss_weight_schedule = None if cls_loss_weight_schedule is None else dict(cls_loss_weight_schedule)
        self.current_train_epoch = 0
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(float(loss_normalizer)))

        if self.assignment_mode not in {"hard", "soft", "oracle_point"}:
            raise ValueError(f"Unsupported assignment_mode: {self.assignment_mode}")
        if self.regression_mode not in {"symmetric_linear", "asymmetric_log1p"}:
            raise ValueError(f"Unsupported regression_mode: {self.regression_mode}")
        if self.soft_loss_normalizer_mode not in {"pos_mass", "pos_count"}:
            raise ValueError(f"Unsupported soft_loss_normalizer_mode: {self.soft_loss_normalizer_mode}")
        if self.soft_reg_weight_mode not in {"soft", "binary"}:
            raise ValueError(f"Unsupported soft_reg_weight_mode: {self.soft_reg_weight_mode}")
        if self.soft_cls_target_mode not in {"soft", "binary"}:
            raise ValueError(f"Unsupported soft_cls_target_mode: {self.soft_cls_target_mode}")
        if self.tower_kernel_size <= 0 or self.tower_kernel_size % 2 == 0:
            raise ValueError(f"tower_kernel_size must be a positive odd integer, got {self.tower_kernel_size}")
        if self.predictor_kernel_size <= 0 or self.predictor_kernel_size % 2 == 0:
            raise ValueError(f"predictor_kernel_size must be a positive odd integer, got {self.predictor_kernel_size}")

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
        tower_padding = self.tower_kernel_size // 2
        for idx in range(self.num_convs):
            in_channels = self.in_channels if idx == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    kernel_size=self.tower_kernel_size,
                    stride=1,
                    padding=tower_padding,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
            self.reg_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    kernel_size=self.tower_kernel_size,
                    stride=1,
                    padding=tower_padding,
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
        cls_feat = feat.detach() if self.detach_cls_input_from_backbone else feat
        reg_feat = feat.detach() if self.detach_reg_input_from_backbone else feat
        branch_mask = mask
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat, _ = cls_conv(cls_feat, branch_mask)
            reg_feat, _ = reg_conv(reg_feat, branch_mask)

        cls_pred = self.cls_head(cls_feat)
        reg_pred = F.relu(self.scale[level_idx](self.reg_head(reg_feat)))
        return cls_pred, reg_pred

    def _generate_points(self, feat_list, temporal_grid_list=None):
        if temporal_grid_list is not None:
            try:
                return self.prior_generator(feat_list, temporal_grid_list)
            except TypeError:
                return self.prior_generator(feat_list)

        try:
            return self.prior_generator(feat_list)
        except TypeError as exc:
            raise ValueError(
                "prior_generator requires temporal_grid_list, but forward received temporal_grid_list=None."
            ) from exc

    def forward_train(self, feat_list, mask_list, temporal_grid_list=None, gt_segments=None, gt_labels=None, **kwargs):
        cls_pred = []
        reg_pred = []
        for level_idx, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_out, reg_out = self._forward_single_level(feat, mask, level_idx)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)

        points = self._generate_points(feat_list, temporal_grid_list)
        return self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)

    def forward_test(self, feat_list, mask_list, temporal_grid_list=None, **kwargs):
        cls_pred = []
        reg_pred = []
        for level_idx, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_out, reg_out = self._forward_single_level(feat, mask, level_idx)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)

        points = self._generate_points(feat_list, temporal_grid_list)
        return self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)

    def _concat_points(self, points):
        if len(points) == 0:
            raise ValueError("points must not be empty")
        dim = 1 if points[0].dim() == 3 else 0
        return torch.cat(points, dim=dim)

    def _points_per_sample(self, points, batch_size):
        point_tensor = self._concat_points(points)
        if point_tensor.dim() == 2:
            return [point_tensor] * batch_size
        if point_tensor.dim() == 3:
            if point_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch between points ({point_tensor.shape[0]}) and targets ({batch_size})."
                )
            return list(point_tensor)
        raise ValueError(f"Unsupported point tensor shape: {tuple(point_tensor.shape)}")

    def _point_fields(self, point_tensor):
        center = point_tensor[..., 0]
        reg_min = point_tensor[..., 1]
        reg_max = point_tensor[..., 2]
        if point_tensor.shape[-1] >= 5:
            left_scale = point_tensor[..., 3].clamp_min(self.reg_denom_floor)
            right_scale = point_tensor[..., 4].clamp_min(self.reg_denom_floor)
            point_scale = (left_scale + right_scale).clamp_min(self.reg_denom_floor)
        else:
            point_scale = point_tensor[..., 3].clamp_min(self.reg_denom_floor)
            left_scale = point_scale
            right_scale = point_scale
        return center, reg_min, reg_max, left_scale, right_scale, point_scale

    def _level_offsets(self, points):
        offsets = []
        start = 0
        for level in points:
            level_len = int(level.shape[1] if level.dim() == 3 else level.shape[0])
            offsets.append((start, start + level_len))
            start += level_len
        return offsets

    def _encode_regression_targets(self, left, right, left_scale, right_scale, point_scale):
        if self.regression_mode == "symmetric_linear":
            return torch.stack(
                [
                    (left / point_scale).clamp_min(0.0),
                    (right / point_scale).clamp_min(0.0),
                ],
                dim=-1,
            )

        return torch.stack(
            [
                torch.log1p((left / left_scale).clamp_min(0.0)),
                torch.log1p((right / right_scale).clamp_min(0.0)),
            ],
            dim=-1,
        )

    def get_refined_proposals(self, points, reg_pred):
        point_tensor = self._concat_points(points)
        reg_tensor = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)
        center, _, _, left_scale, right_scale, point_scale = self._point_fields(point_tensor)

        if self.regression_mode == "symmetric_linear":
            left = reg_tensor[:, :, 0] * point_scale
            right = reg_tensor[:, :, 1] * point_scale
        else:
            left = torch.expm1(reg_tensor[:, :, 0].clamp_min(0.0)) * left_scale
            right = torch.expm1(reg_tensor[:, :, 1].clamp_min(0.0)) * right_scale

        start = center - left
        end = center + right
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

    def set_train_epoch(self, curr_epoch):
        self.current_train_epoch = int(curr_epoch)

    def _resolve_cls_loss_weight(self):
        if self.cls_loss_weight_schedule is None:
            return float(self.cls_loss_weight)

        warmup_epochs = int(self.cls_loss_weight_schedule.get("warmup_epochs", 0))
        warmup_value = float(self.cls_loss_weight_schedule.get("warmup_value", 0.0))
        after_warmup_value = float(self.cls_loss_weight_schedule.get("after_warmup_value", self.cls_loss_weight))
        if self.current_train_epoch < warmup_epochs:
            return warmup_value
        return after_warmup_value

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg, reg_weight, target_debug = self.prepare_targets(points, gt_segments, gt_labels)

        gt_cls = torch.stack(gt_cls)
        gt_reg = torch.stack(gt_reg)
        reg_weight = torch.stack(reg_weight)
        valid_mask = torch.cat(mask_list, dim=1)

        if self.assignment_mode == "soft":
            pos_weight = gt_cls.max(dim=-1).values * valid_mask.to(gt_cls.dtype)
            pos_mass = float(pos_weight.sum().item())
            pos_count = int(torch.logical_and(gt_cls.max(dim=-1).values > 0, valid_mask).sum().item())
            if self.soft_loss_normalizer_mode == "pos_count":
                normalizer_base = float(max(pos_count, 1))
            else:
                normalizer_base = max(pos_mass, 1.0)
        else:
            pos_binary = torch.logical_and(gt_cls.sum(dim=-1) > 0, valid_mask)
            pos_mass = float(pos_binary.sum().item())
            pos_count = int(pos_binary.sum().item())
            normalizer_base = float(max(pos_mass, 1.0))

        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * normalizer_base
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = normalizer_base

        cls_pred_flat = [tensor.permute(0, 2, 1) for tensor in cls_pred]
        cls_pred_flat = torch.cat(cls_pred_flat, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        if self.assignment_mode == "soft" and self.soft_cls_target_mode == "binary":
            gt_target = (gt_target > 0).to(gt_target.dtype)
        gt_target = gt_target * (1 - self.label_smoothing)
        gt_target = gt_target + self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred_flat, gt_target, reduction="sum")
        cls_loss /= loss_normalizer
        effective_cls_loss_weight = self._resolve_cls_loss_weight()
        cls_loss = cls_loss * effective_cls_loss_weight

        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg_split = gt_reg.permute(0, 2, 1).split(split_size, dim=-1)
        pred_segments = self.get_refined_proposals(points, reg_pred)
        gt_segments = self.get_refined_proposals(points, gt_reg_split)

        reg_mask = torch.logical_and(reg_weight > 0, valid_mask)
        effective_reg_weight_sum = 0.0
        if reg_mask.any():
            if self.assignment_mode == "soft":
                reg_loss_raw = self.reg_loss(pred_segments[reg_mask], gt_segments[reg_mask], reduction="none").reshape(-1)
                reg_weight_flat = reg_weight[reg_mask]
                if self.soft_reg_weight_mode == "binary":
                    reg_weight_flat = torch.ones_like(reg_weight_flat)
                effective_reg_weight_sum = float(reg_weight_flat.sum().item())
                reg_loss = (reg_loss_raw * reg_weight_flat).sum()
            else:
                effective_reg_weight_sum = float(reg_mask.sum().item())
                reg_loss = self.reg_loss(pred_segments[reg_mask], gt_segments[reg_mask], reduction="sum")
            reg_loss /= loss_normalizer
        else:
            reg_loss = pred_segments.sum() * 0

        if self.reg_loss_weight is not None:
            reg_loss_weight = self.reg_loss_weight
        elif self.loss_weight > 0:
            reg_loss_weight = self.loss_weight
        else:
            reg_loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        if self.debug_enabled:
            debug_state = dict(target_debug)
            debug_state["bridge_assignment_mode"] = self.assignment_mode
            debug_state["bridge_regression_mode"] = self.regression_mode
            debug_state["bridge_soft_loss_normalizer_mode"] = self.soft_loss_normalizer_mode
            debug_state["bridge_soft_reg_weight_mode"] = self.soft_reg_weight_mode
            debug_state["bridge_soft_cls_target_mode"] = self.soft_cls_target_mode
            debug_state["bridge_detach_cls_input_from_backbone"] = bool(self.detach_cls_input_from_backbone)
            debug_state["bridge_detach_reg_input_from_backbone"] = bool(self.detach_reg_input_from_backbone)
            debug_state["bridge_train_epoch"] = int(self.current_train_epoch)
            debug_state["bridge_pos_mass_total"] = pos_mass
            debug_state["bridge_positive_count_total"] = pos_count
            debug_state["bridge_valid_points_total"] = int(valid_mask.sum().item())
            debug_state["bridge_reg_points_total"] = int(reg_mask.sum().item())
            debug_state["bridge_cls_loss_weight"] = float(self.cls_loss_weight)
            debug_state["bridge_cls_loss_weight_effective"] = float(effective_cls_loss_weight)
            debug_state["bridge_reg_loss_weight"] = (
                float(reg_loss_weight) if not torch.is_tensor(reg_loss_weight) else float(reg_loss_weight.item())
            )
            debug_state["bridge_cls_target_mass_total"] = float(gt_cls.max(dim=-1).values[valid_mask].sum().item())
            debug_state["bridge_reg_weight_sum_total"] = float(reg_weight[valid_mask].sum().item())
            debug_state["bridge_reg_effective_weight_sum_total"] = effective_reg_weight_sum
            debug_state["bridge_loss_normalizer_base"] = normalizer_base
            debug_state["bridge_loss_normalizer"] = float(
                loss_normalizer.item() if torch.is_tensor(loss_normalizer) else loss_normalizer
            )
            self._latest_debug_state = debug_state

        return {"cls_loss": cls_loss, "reg_loss": reg_loss * reg_loss_weight}

    def _build_candidate_mask(self, point, gt_segs, reg_targets):
        center_t, _, _, left_scale, right_scale, point_scale = self._point_fields(point)
        center_t = center_t[:, None]
        inside_gt_seg = reg_targets.min(dim=-1).values > 0
        if self.center_sample != "radius":
            return inside_gt_seg

        center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
        radius_base = torch.sqrt((left_scale[:, None] * right_scale[:, None]).clamp_min(self.reg_denom_floor**2))
        if point.shape[-1] < 5:
            radius_base = point_scale[:, None]
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
        center_t, _, _, _, _, point_scale = self._point_fields(point)
        center_t = center_t[:, None]
        point_scale = point_scale[:, None]
        gt_center = 0.5 * (gt_segment[:, 0] + gt_segment[:, 1])[None, :]
        gt_len = (gt_segment[:, 1] - gt_segment[:, 0])[None, :].clamp_min(self.reg_denom_floor)

        center_cost = (center_t - gt_center).abs() / (0.5 * gt_len + 0.5 * point_scale).clamp_min(self.reg_denom_floor)
        scale_cost = torch.abs(torch.log((gt_len / point_scale).clamp_min(1e-6)))
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
    def _prepare_targets_hard(self, points, gt_segments, gt_labels):
        point_list = self._points_per_sample(points, len(gt_segments))
        level_offsets = self._level_offsets(points)
        gt_cls = []
        gt_reg = []
        reg_weight_list = []
        debug_state = {}

        for point, gt_segment, gt_label in zip(point_list, gt_segments, gt_labels):
            # point: [N_pts, 4or5]
            num_pts = point.shape[0]
            center_t, reg_min, reg_max, left_scale, right_scale, point_scale = self._point_fields(point)
            num_gts = gt_segment.shape[0]
            if num_gts == 0:
                gt_cls.append(gt_segment.new_zeros((num_pts, self.num_classes)))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                reg_weight_list.append(gt_segment.new_zeros((num_pts,)))
                continue

            lens = (gt_segment[:, 1] - gt_segment[:, 0])[None, :].repeat(num_pts, 1)
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = center_t[:, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - center_t[:, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                radius = point_scale[:, None] * self.center_sample_radius
                t_mins = center_pts - radius
                t_maxs = center_pts + radius
                cb_left = center_t[:, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - center_t[:, None]
                center_seg = torch.stack((cb_left, cb_right), dim=-1)
                inside_gt_seg_mask = center_seg.min(dim=-1).values > 0
            else:
                inside_gt_seg_mask = reg_targets.min(dim=-1).values > 0

            max_regress_distance = reg_targets.max(dim=-1).values
            inside_regress_range = torch.logical_and(
                max_regress_distance >= reg_min[:, None],
                max_regress_distance <= reg_max[:, None],
            )

            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            min_len, min_len_inds = lens.min(dim=1)

            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot
            cls_targets.clamp_(min=0.0, max=1.0)

            reg_encoded = self._encode_regression_targets(
                left,
                right,
                left_scale[:, None],
                right_scale[:, None],
                point_scale[:, None],
            )
            reg_target = reg_encoded[torch.arange(num_pts, device=point.device), min_len_inds]
            reg_weight = (min_len < float("inf")).to(reg_targets.dtype)
            reg_target = torch.where(reg_weight[:, None] > 0, reg_target, reg_target.new_zeros(reg_target.shape))

            gt_cls.append(cls_targets)
            gt_reg.append(reg_target)
            reg_weight_list.append(reg_weight)

            if self.debug_enabled:
                positive_mask = cls_targets.sum(dim=-1) > 0
                debug_state.setdefault("bridge_num_gt_per_sample", []).append(int(num_gts))
                debug_state.setdefault("bridge_positive_points_per_sample", []).append(int(positive_mask.sum().item()))
                debug_state.setdefault("bridge_reg_weight_sum_per_sample", []).append(float(reg_weight.sum().item()))
                for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                    level_pos = positive_mask[start_idx:end_idx]
                    debug_state.setdefault(f"bridge_level{level_idx}_positive_points_per_sample", []).append(
                        int(level_pos.sum().item())
                    )
                    debug_state.setdefault(f"bridge_level{level_idx}_reg_weight_sum_per_sample", []).append(
                        float(reg_weight[start_idx:end_idx].sum().item())
                    )

        return gt_cls, gt_reg, reg_weight_list, debug_state

    @torch.no_grad()
    def _prepare_targets_soft(self, points, gt_segments, gt_labels):
        point_list = self._points_per_sample(points, len(gt_segments))
        level_offsets = self._level_offsets(points)
        gt_cls = []
        gt_reg = []
        reg_weight_list = []
        debug_state = {}

        for point, gt_segment, gt_label in zip(point_list, gt_segments, gt_labels):
            num_pts = point.shape[0]
            num_gts = gt_segment.shape[0]
            if num_gts == 0:
                gt_cls.append(gt_segment.new_zeros((num_pts, self.num_classes)))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                reg_weight_list.append(gt_segment.new_zeros((num_pts,)))
                continue

            center_t, _, _, left_scale, right_scale, point_scale = self._point_fields(point)
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = center_t[:, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - center_t[:, None]
            reg_targets = torch.stack((left, right), dim=-1)

            candidate_mask = self._build_candidate_mask(point, gt_segs, reg_targets)
            assign_weights, total_cost = self._build_assignment_weights(point, gt_segment, candidate_mask)

            one_hot = F.one_hot(gt_label.long(), self.num_classes).to(assign_weights.dtype)
            weighted_labels = assign_weights[:, :, None] * one_hot[None, :, :]
            cls_targets = weighted_labels.max(dim=1).values
            cls_targets.clamp_(min=0.0, max=1.0)

            reg_encoded = self._encode_regression_targets(
                left,
                right,
                left_scale[:, None],
                right_scale[:, None],
                point_scale[:, None],
            )
            reg_weight, best_gt_idx = assign_weights.max(dim=1)
            reg_target = reg_encoded[torch.arange(num_pts, device=point.device), best_gt_idx]
            reg_target = torch.where(reg_weight[:, None] > 0, reg_target, reg_target.new_zeros(reg_target.shape))

            gt_cls.append(cls_targets)
            gt_reg.append(reg_target)
            reg_weight_list.append(reg_weight)

            if self.debug_enabled:
                positive_mask = cls_targets.max(dim=-1).values > 0
                multi_gt_mask = (assign_weights > 0).sum(dim=1) > 1
                debug_state.setdefault("bridge_num_gt_per_sample", []).append(int(num_gts))
                debug_state.setdefault("bridge_candidate_points_per_sample", []).append(int(candidate_mask.any(dim=1).sum().item()))
                debug_state.setdefault("bridge_positive_points_per_sample", []).append(int(positive_mask.sum().item()))
                debug_state.setdefault("bridge_multi_gt_points_per_sample", []).append(int(multi_gt_mask.sum().item()))
                debug_state.setdefault("bridge_reg_weight_sum_per_sample", []).append(float(reg_weight.sum().item()))
                debug_state.setdefault("bridge_candidate_gt_covered_per_sample", []).append(
                    int(candidate_mask.any(dim=0).sum().item())
                )
                debug_state.setdefault("bridge_zero_candidate_gt_per_sample", []).append(
                    int((~candidate_mask.any(dim=0)).sum().item())
                )
                if positive_mask.any():
                    debug_state.setdefault("bridge_reg_target_absmax_per_sample", []).append(
                        float(reg_target[positive_mask].abs().max().item())
                    )
                else:
                    debug_state.setdefault("bridge_reg_target_absmax_per_sample", []).append(0.0)
                if torch.isfinite(total_cost).any():
                    finite_cost = total_cost[torch.isfinite(total_cost)]
                    debug_state.setdefault("bridge_cost_min_per_sample", []).append(float(finite_cost.min().item()))
                    debug_state.setdefault("bridge_cost_max_per_sample", []).append(float(finite_cost.max().item()))
                for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                    level_candidate = candidate_mask[start_idx:end_idx].any(dim=1)
                    level_positive = positive_mask[start_idx:end_idx]
                    debug_state.setdefault(f"bridge_level{level_idx}_candidate_points_per_sample", []).append(
                        int(level_candidate.sum().item())
                    )
                    debug_state.setdefault(f"bridge_level{level_idx}_positive_points_per_sample", []).append(
                        int(level_positive.sum().item())
                    )
                    debug_state.setdefault(f"bridge_level{level_idx}_reg_weight_sum_per_sample", []).append(
                        float(reg_weight[start_idx:end_idx].sum().item())
                    )

        return gt_cls, gt_reg, reg_weight_list, debug_state

    @torch.no_grad()
    def _prepare_targets_oracle_point(self, points, gt_segments, gt_labels):
        point_list = self._points_per_sample(points, len(gt_segments))
        level_offsets = self._level_offsets(points)
        gt_cls = []
        gt_reg = []
        reg_weight_list = []
        debug_state = {}

        for point, gt_segment, gt_label in zip(point_list, gt_segments, gt_labels):
            num_pts = point.shape[0]
            num_gts = gt_segment.shape[0]
            if num_gts == 0:
                gt_cls.append(gt_segment.new_zeros((num_pts, self.num_classes)))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                reg_weight_list.append(gt_segment.new_zeros((num_pts,)))
                continue

            center_t, _, _, left_scale, right_scale, point_scale = self._point_fields(point)
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            gt_center = 0.5 * (gt_segment[:, 0] + gt_segment[:, 1])

            left = center_t[:, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - center_t[:, None]
            inside_gt = torch.logical_and(left >= 0, right >= 0)
            center_cost = (center_t[:, None] - gt_center[None, :]).abs()

            oracle_cost = center_cost.masked_fill(~inside_gt, float("inf"))
            missing_inside = ~torch.isfinite(oracle_cost).any(dim=0)
            if missing_inside.any():
                oracle_cost[:, missing_inside] = center_cost[:, missing_inside]

            oracle_idx = oracle_cost.argmin(dim=0)
            oracle_mask = torch.zeros((num_pts, num_gts), device=point.device, dtype=torch.bool)
            oracle_mask[oracle_idx, torch.arange(num_gts, device=point.device)] = True

            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(center_t.dtype)
            cls_targets = oracle_mask.to(center_t.dtype) @ gt_label_one_hot
            cls_targets.clamp_(min=0.0, max=1.0)

            reg_encoded = self._encode_regression_targets(
                left,
                right,
                left_scale[:, None],
                right_scale[:, None],
                point_scale[:, None],
            )
            masked_oracle_cost = oracle_cost.masked_fill(~oracle_mask, float("inf"))
            min_cost, best_gt_idx = masked_oracle_cost.min(dim=1)
            reg_weight = torch.isfinite(min_cost).to(center_t.dtype)
            reg_target = reg_encoded[torch.arange(num_pts, device=point.device), best_gt_idx]
            reg_target = torch.where(reg_weight[:, None] > 0, reg_target, reg_target.new_zeros(reg_target.shape))

            gt_cls.append(cls_targets)
            gt_reg.append(reg_target)
            reg_weight_list.append(reg_weight)

            if self.debug_enabled:
                positive_mask = cls_targets.sum(dim=-1) > 0
                debug_state.setdefault("bridge_num_gt_per_sample", []).append(int(num_gts))
                debug_state.setdefault("bridge_candidate_gt_covered_per_sample", []).append(int(num_gts))
                debug_state.setdefault("bridge_candidate_points_per_sample", []).append(int(positive_mask.sum().item()))
                debug_state.setdefault("bridge_positive_points_per_sample", []).append(int(positive_mask.sum().item()))
                debug_state.setdefault("bridge_reg_weight_sum_per_sample", []).append(float(reg_weight.sum().item()))
                debug_state.setdefault("bridge_oracle_inside_gt_covered_per_sample", []).append(
                    int(inside_gt.any(dim=0).sum().item())
                )
                debug_state.setdefault("bridge_oracle_center_cost_min_per_sample", []).append(
                    float(center_cost.min().item())
                )
                debug_state.setdefault("bridge_oracle_center_cost_max_per_sample", []).append(
                    float(center_cost.max().item())
                )
                if positive_mask.any():
                    debug_state.setdefault("bridge_reg_target_absmax_per_sample", []).append(
                        float(reg_target[positive_mask].abs().max().item())
                    )
                else:
                    debug_state.setdefault("bridge_reg_target_absmax_per_sample", []).append(0.0)
                for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                    level_positive = positive_mask[start_idx:end_idx]
                    debug_state.setdefault(f"bridge_level{level_idx}_candidate_points_per_sample", []).append(
                        int(level_positive.sum().item())
                    )
                    debug_state.setdefault(f"bridge_level{level_idx}_positive_points_per_sample", []).append(
                        int(level_positive.sum().item())
                    )
                    debug_state.setdefault(f"bridge_level{level_idx}_reg_weight_sum_per_sample", []).append(
                        float(reg_weight[start_idx:end_idx].sum().item())
                    )

        return gt_cls, gt_reg, reg_weight_list, debug_state

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        if self.assignment_mode == "hard":
            return self._prepare_targets_hard(points, gt_segments, gt_labels)
        if self.assignment_mode == "oracle_point":
            return self._prepare_targets_oracle_point(points, gt_segments, gt_labels)
        return self._prepare_targets_soft(points, gt_segments, gt_labels)
