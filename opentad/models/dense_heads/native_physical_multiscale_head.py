import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale


@HEADS.register_module()
class NativePhysicalMultiScaleHead(nn.Module):
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
        min_center_radius=0.0,
        label_smoothing=0.0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        filter_similar_gt=True,
        use_regress_range=True,
        debug_cfg=None,
    ):
        super().__init__()
        if prior_generator is None:
            raise ValueError("NativePhysicalMultiScaleHead requires a prior_generator config.")
        if loss is None or not self._has_cfg_key(loss, "cls_loss") or not self._has_cfg_key(loss, "reg_loss"):
            raise ValueError("NativePhysicalMultiScaleHead requires loss.cls_loss and loss.reg_loss configs.")
        if center_sample not in {"radius", "none"}:
            raise ValueError(f"Unsupported center_sample: {center_sample}")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt
        self.loss_weight = loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.min_center_radius = min_center_radius
        self.use_regress_range = use_regress_range
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(float(loss_normalizer)))

        debug_cfg = {} if debug_cfg is None else dict(debug_cfg)
        self.debug_enabled = bool(debug_cfg.get("enable", False))
        self._latest_debug_state = {}

        self.prior_generator = build_prior_generator(prior_generator)
        self._init_layers()

        self.cls_loss = build_loss(self._cfg_get(loss, "cls_loss"))
        self.reg_loss = build_loss(self._cfg_get(loss, "reg_loss"))

    @staticmethod
    def _has_cfg_key(cfg, key):
        if isinstance(cfg, dict):
            return key in cfg
        return hasattr(cfg, key)

    @staticmethod
    def _cfg_get(cfg, key):
        if isinstance(cfg, dict):
            return cfg[key]
        return getattr(cfg, key)

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

        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def _forward_single_level(self, feat, mask, level_idx):
        cls_feat = feat
        reg_feat = feat
        branch_mask = mask.bool()
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat, _ = cls_conv(cls_feat, branch_mask)
            reg_feat, _ = reg_conv(reg_feat, branch_mask)

        cls_pred = self.cls_head(cls_feat)
        reg_pred = F.softplus(self.scale[level_idx](self.reg_head(reg_feat)))
        return cls_pred, reg_pred

    def forward_train(self, feat_list, mask_list, temporal_grid_list=None, gt_segments=None, gt_labels=None, **kwargs):
        if gt_segments is None or gt_labels is None:
            raise ValueError("forward_train requires gt_segments and gt_labels.")

        cls_pred = []
        reg_pred = []
        for level_idx, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_out, reg_out = self._forward_single_level(feat, mask, level_idx)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)

        points = self.prior_generator(feat_list, temporal_grid_list)
        return self.losses(cls_pred, reg_pred, mask_list, temporal_grid_list, points, gt_segments, gt_labels)

    def forward_test(self, feat_list, mask_list, temporal_grid_list=None, **kwargs):
        cls_pred = []
        reg_pred = []
        for level_idx, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_out, reg_out = self._forward_single_level(feat, mask, level_idx)
            cls_pred.append(cls_out)
            reg_pred.append(reg_out)

        points = self.prior_generator(feat_list, temporal_grid_list)
        return self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list, temporal_grid_list)

    def _valid_mask_single_level(self, mask, grid):
        valid = mask.bool() & grid["valid_mask"].bool()
        if "fresh_mask" in grid:
            valid = valid & grid["fresh_mask"].bool()
        return valid

    def _valid_mask(self, mask_list, temporal_grid_list):
        return torch.cat(
            [self._valid_mask_single_level(mask, grid) for mask, grid in zip(mask_list, temporal_grid_list)],
            dim=1,
        )

    def _concat_points(self, points):
        return torch.cat(points, dim=1)

    def _level_offsets(self, points):
        offsets = []
        start = 0
        for level in points:
            length = int(level.shape[1])
            offsets.append((start, start + length))
            start += length
        return offsets

    def get_refined_proposals(self, points, reg_pred):
        point_tensor = self._concat_points(points)
        reg_tensor = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)

        center = point_tensor[:, :, 0].to(reg_tensor.dtype)
        left_scale = point_tensor[:, :, 3].to(reg_tensor.dtype).clamp_min(1e-6)
        right_scale = point_tensor[:, :, 4].to(reg_tensor.dtype).clamp_min(1e-6)
        left = reg_tensor[:, :, 0] * left_scale
        right = reg_tensor[:, :, 1] * right_scale
        start = center - left
        end = center + right
        return torch.stack((start, end), dim=-1)

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list, temporal_grid_list):
        proposals = self.get_refined_proposals(points, reg_pred)
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()
        valid_mask = self._valid_mask(mask_list, temporal_grid_list)

        new_proposals = []
        new_scores = []
        for proposal, score, keep in zip(proposals, scores, valid_mask):
            new_proposals.append(proposal[keep])
            new_scores.append(score[keep])
        return new_proposals, new_scores

    def collect_debug_state(self):
        return dict(self._latest_debug_state)

    def losses(self, cls_pred, reg_pred, mask_list, temporal_grid_list, points, gt_segments, gt_labels):
        gt_cls, gt_segments_assigned, target_debug = self.prepare_targets(
            points, mask_list, temporal_grid_list, gt_segments, gt_labels,
        )

        valid_mask = self._valid_mask(mask_list, temporal_grid_list)
        pos_mask = (gt_cls.sum(-1) > 0) & valid_mask
        num_pos = int(pos_mask.sum().item())

        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        cls_flat = torch.cat([pred.permute(0, 2, 1) for pred in cls_pred], dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        gt_target = gt_target * (1 - self.label_smoothing) + self.label_smoothing / (self.num_classes + 1)
        cls_loss = self.cls_loss(cls_flat, gt_target, reduction="sum") / loss_normalizer
        cls_loss = cls_loss * self.cls_loss_weight

        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        target_segments = gt_segments_assigned[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            reg_loss = self.reg_loss(pred_segments, target_segments, reduction="sum") / loss_normalizer
        reg_loss = reg_loss * self.reg_loss_weight * self.loss_weight

        if self.debug_enabled:
            valid_count = int(valid_mask.sum().item())
            level_offsets = self._level_offsets(points)
            debug_state = dict(target_debug)
            debug_state["native_ms_head_num_pos_total"] = num_pos
            debug_state["native_ms_head_valid_points_total"] = valid_count
            debug_state["native_ms_head_positive_ratio_total"] = float(num_pos / max(valid_count, 1))
            debug_state["native_ms_head_loss_normalizer"] = float(
                loss_normalizer.item() if torch.is_tensor(loss_normalizer) else loss_normalizer
            )
            for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                level_valid = valid_mask[:, start_idx:end_idx]
                level_pos = pos_mask[:, start_idx:end_idx]
                debug_state[f"native_ms_head_level{level_idx}_valid_total"] = int(level_valid.sum().item())
                debug_state[f"native_ms_head_level{level_idx}_pos_total"] = int(level_pos.sum().item())
            self._latest_debug_state = debug_state

        return {"cls_loss": cls_loss, "reg_loss": reg_loss}

    @torch.no_grad()
    def prepare_targets(self, points, mask_list, temporal_grid_list, gt_segments, gt_labels):
        concat_points = self._concat_points(points)
        valid_mask = self._valid_mask(mask_list, temporal_grid_list)
        level_offsets = self._level_offsets(points)

        gt_cls = []
        gt_assigned_segments = []
        debug_state = {
            "native_ms_head_num_gt_per_sample": [],
            "native_ms_head_center_hits_per_sample": [],
            "native_ms_head_range_hits_per_sample": [],
            "native_ms_head_matched_hits_per_sample": [],
            "native_ms_head_pos_per_sample": [],
            "native_ms_head_assigned_gt_span_mean_per_sample": [],
        }

        for point, valid, gt_segment, gt_label in zip(concat_points, valid_mask, gt_segments, gt_labels):
            num_points = point.shape[0]
            cls_targets = point.new_zeros((num_points, self.num_classes))
            assigned_segments = point.new_zeros((num_points, 2))
            centers = point[:, 0]
            reg_min = point[:, 1]
            reg_max = point[:, 2]
            point_scale = 0.5 * (point[:, 3] + point[:, 4]).clamp_min(1e-6)

            num_gts = int(gt_segment.shape[0])
            debug_state["native_ms_head_num_gt_per_sample"].append(num_gts)

            if num_gts == 0:
                gt_cls.append(cls_targets)
                gt_assigned_segments.append(assigned_segments)
                debug_state["native_ms_head_center_hits_per_sample"].append(0)
                debug_state["native_ms_head_range_hits_per_sample"].append(0)
                debug_state["native_ms_head_matched_hits_per_sample"].append(0)
                debug_state["native_ms_head_pos_per_sample"].append(0)
                debug_state["native_ms_head_assigned_gt_span_mean_per_sample"].append(0.0)
                continue

            gt_segment = gt_segment.to(device=point.device, dtype=point.dtype)
            gt_label = gt_label.to(device=point.device)
            gt_start = gt_segment[:, 0][None, :]
            gt_end = gt_segment[:, 1][None, :]
            gt_center = 0.5 * (gt_start + gt_end)
            gt_span = (gt_end - gt_start).clamp_min(1e-6)
            center_t = centers[:, None]

            left = center_t - gt_start
            right = gt_end - center_t
            reg_targets = torch.stack((left, right), dim=-1)
            inside_segment = reg_targets.min(dim=-1).values > 0

            if self.center_sample == "radius":
                radius = torch.maximum(
                    point_scale[:, None] * self.center_sample_radius,
                    point_scale.new_tensor(float(self.min_center_radius)),
                )
                t_mins = gt_center - radius
                t_maxs = gt_center + radius
                cb_left = center_t - torch.maximum(t_mins, gt_start)
                cb_right = torch.minimum(t_maxs, gt_end) - center_t
                center_segment = torch.stack((cb_left, cb_right), dim=-1)
                inside_center = center_segment.min(dim=-1).values > 0
            else:
                inside_center = inside_segment

            max_regress_distance = reg_targets.max(dim=-1).values
            if self.use_regress_range:
                inside_regress_range = (max_regress_distance >= reg_min[:, None]) & (
                    max_regress_distance <= reg_max[:, None]
                )
            else:
                inside_regress_range = torch.ones_like(inside_segment)

            center_hits = (inside_segment & inside_center & valid[:, None]).any(dim=1)
            range_hits = (inside_segment & inside_regress_range & valid[:, None]).any(dim=1)
            matched_mask = inside_segment & inside_center & inside_regress_range & valid[:, None]
            matched_hits = matched_mask.any(dim=1)

            debug_state["native_ms_head_center_hits_per_sample"].append(int(center_hits.sum().item()))
            debug_state["native_ms_head_range_hits_per_sample"].append(int(range_hits.sum().item()))
            debug_state["native_ms_head_matched_hits_per_sample"].append(int(matched_hits.sum().item()))

            # Assign each point to the GT with shortest span among matched
            lens = gt_span.repeat(num_points, 1)
            lens = lens.masked_fill(~matched_mask, float("inf"))
            min_len, assign_idx = lens.min(dim=1)
            finite_assignment = torch.isfinite(min_len)

            if self.filter_similar_gt:
                cls_match = (lens <= (min_len[:, None] + 1e-3)) & torch.isfinite(lens)
            else:
                cls_match = matched_mask

            one_hot = F.one_hot(gt_label.long(), self.num_classes).to(cls_targets.dtype)
            cls_targets = cls_match.to(cls_targets.dtype) @ one_hot
            cls_targets.clamp_(min=0.0, max=1.0)
            assigned_segments[finite_assignment] = gt_segment[assign_idx[finite_assignment]]

            positive_mask = cls_targets.sum(dim=-1) > 0
            debug_state["native_ms_head_pos_per_sample"].append(int(positive_mask.sum().item()))
            if finite_assignment.any():
                assigned_spans = (assigned_segments[finite_assignment, 1] - assigned_segments[finite_assignment, 0]).clamp_min(0.0)
                debug_state["native_ms_head_assigned_gt_span_mean_per_sample"].append(float(assigned_spans.mean().item()))
            else:
                debug_state["native_ms_head_assigned_gt_span_mean_per_sample"].append(0.0)

            gt_cls.append(cls_targets)
            gt_assigned_segments.append(assigned_segments)

        return torch.stack(gt_cls), torch.stack(gt_assigned_segments), debug_state
