import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule
from ..builder import HEADS, build_loss


@HEADS.register_module()
class NativePhysicalPointHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=2,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        cls_prior_prob=0.01,
        label_smoothing=0.0,
        center_sample_radius=1.5,
        min_center_radius=1.0,
        loss_weight=1.0,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        debug_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.center_sample_radius = center_sample_radius
        self.min_center_radius = min_center_radius
        self.loss_weight = loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(float(loss_normalizer)))

        debug_cfg = {} if debug_cfg is None else dict(debug_cfg)
        self.debug_enabled = bool(debug_cfg.get("enable", False))
        self._latest_debug_state = {}

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

        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def _forward_single_level(self, feat, mask):
        cls_feat = feat
        reg_feat = feat
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat, mask = cls_conv(cls_feat, mask)
            reg_feat, mask = reg_conv(reg_feat, mask)

        cls_pred = self.cls_head(cls_feat)
        reg_pred = F.softplus(self.reg_head(reg_feat))
        return cls_pred, reg_pred

    def _single_level_inputs(self, feat_list, mask_list, temporal_grid_list):
        if len(feat_list) != 1 or len(mask_list) != 1 or len(temporal_grid_list) != 1:
            raise ValueError("NativePhysicalPointHead is a single-level overfit head; set projection arch branch depth to 0 and neck=None.")
        grid = temporal_grid_list[0]
        return feat_list[0], mask_list[0].bool(), grid

    def forward_train(self, feat_list, mask_list, temporal_grid_list=None, gt_segments=None, gt_labels=None, **kwargs):
        feat, mask, grid = self._single_level_inputs(feat_list, mask_list, temporal_grid_list)
        cls_pred, reg_pred = self._forward_single_level(feat, mask)
        return self.losses(cls_pred, reg_pred, mask, grid, gt_segments, gt_labels)

    def forward_test(self, feat_list, mask_list, temporal_grid_list=None, **kwargs):
        feat, mask, grid = self._single_level_inputs(feat_list, mask_list, temporal_grid_list)
        cls_pred, reg_pred = self._forward_single_level(feat, mask)
        return self.get_valid_proposals_scores(grid, reg_pred, cls_pred, mask)

    def _valid_mask(self, mask, grid):
        valid = mask.bool() & grid["valid_mask"].bool()
        if "fresh_mask" in grid:
            valid = valid & grid["fresh_mask"].bool()
        return valid

    def get_refined_proposals(self, grid, reg_pred):
        center = grid["center"].to(reg_pred.dtype)
        reg = reg_pred.permute(0, 2, 1)
        start = center - reg[:, :, 0]
        end = center + reg[:, :, 1]
        return torch.stack((start, end), dim=-1)

    def get_valid_proposals_scores(self, grid, reg_pred, cls_pred, mask):
        proposals = self.get_refined_proposals(grid, reg_pred)
        scores = cls_pred.permute(0, 2, 1).sigmoid()
        valid = self._valid_mask(mask, grid)

        new_proposals = []
        new_scores = []
        for proposal, score, keep in zip(proposals, scores, valid):
            new_proposals.append(proposal[keep])
            new_scores.append(score[keep])
        return new_proposals, new_scores

    def collect_debug_state(self):
        return dict(self._latest_debug_state)

    def losses(self, cls_pred, reg_pred, mask, grid, gt_segments, gt_labels):
        gt_cls, gt_segments_assigned, target_debug = self.prepare_targets(grid, mask, gt_segments, gt_labels)
        valid = self._valid_mask(mask, grid)
        pos_mask = (gt_cls.sum(-1) > 0) & valid
        num_pos = int(pos_mask.sum().item())

        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        cls_flat = cls_pred.permute(0, 2, 1)[valid]
        gt_target = gt_cls[valid]
        gt_target = gt_target * (1 - self.label_smoothing) + self.label_smoothing / (self.num_classes + 1)
        cls_loss = self.cls_loss(cls_flat, gt_target, reduction="sum") / loss_normalizer
        cls_loss = cls_loss * self.cls_loss_weight

        pred_segments = self.get_refined_proposals(grid, reg_pred)[pos_mask]
        target_segments = gt_segments_assigned[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            reg_loss = self.reg_loss(pred_segments, target_segments, reduction="sum") / loss_normalizer
        reg_loss = reg_loss * self.reg_loss_weight * self.loss_weight

        if self.debug_enabled:
            valid_count = int(valid.sum().item())
            debug_state = dict(target_debug)
            debug_state["native_head_valid_points_total"] = valid_count
            debug_state["native_head_num_pos_total"] = num_pos
            debug_state["native_head_positive_ratio_total"] = float(num_pos / max(valid_count, 1))
            debug_state["native_head_loss_normalizer"] = float(
                loss_normalizer.item() if torch.is_tensor(loss_normalizer) else loss_normalizer
            )
            self._latest_debug_state = debug_state

        return {"cls_loss": cls_loss, "reg_loss": reg_loss}

    @torch.no_grad()
    def prepare_targets(self, grid, mask, gt_segments, gt_labels):
        center = grid["center"]
        valid = self._valid_mask(mask, grid)
        local_scale = (0.5 * (grid["cell_left"] + grid["cell_right"])).clamp_min(1e-6)
        gt_cls = []
        gt_assigned_segments = []
        debug_state = {
            "native_head_center_min": [],
            "native_head_center_max": [],
            "native_head_num_gt_per_sample": [],
            "native_head_center_hits_per_sample": [],
            "native_head_pos_per_sample": [],
            "native_head_assigned_gt_span_mean_per_sample": [],
        }

        for centers, valid_mask, scales, gt_segment, gt_label in zip(center, valid, local_scale, gt_segments, gt_labels):
            num_points = centers.shape[0]
            cls_targets = centers.new_zeros((num_points, self.num_classes))
            assigned_segments = centers.new_zeros((num_points, 2))
            debug_state["native_head_num_gt_per_sample"].append(int(gt_segment.shape[0]))
            if valid_mask.any():
                debug_state["native_head_center_min"].append(float(centers[valid_mask].min().item()))
                debug_state["native_head_center_max"].append(float(centers[valid_mask].max().item()))
            else:
                debug_state["native_head_center_min"].append(None)
                debug_state["native_head_center_max"].append(None)

            if gt_segment.numel() == 0:
                gt_cls.append(cls_targets)
                gt_assigned_segments.append(assigned_segments)
                debug_state["native_head_center_hits_per_sample"].append(0)
                debug_state["native_head_pos_per_sample"].append(0)
                debug_state["native_head_assigned_gt_span_mean_per_sample"].append(0.0)
                continue

            gt_segment = gt_segment.to(device=centers.device, dtype=centers.dtype)
            gt_label = gt_label.to(device=centers.device)
            gt_start = gt_segment[:, 0][None, :]
            gt_end = gt_segment[:, 1][None, :]
            gt_center = 0.5 * (gt_start + gt_end)
            gt_span = (gt_end - gt_start).clamp_min(1e-6)

            center_t = centers[:, None]
            inside_segment = (center_t > gt_start) & (center_t < gt_end)
            radius = torch.maximum(scales[:, None] * self.center_sample_radius, scales.new_tensor(self.min_center_radius))
            inside_center = (center_t >= gt_center - radius) & (center_t <= gt_center + radius)
            matched = inside_segment & inside_center & valid_mask[:, None]

            debug_state["native_head_center_hits_per_sample"].append(int((inside_center & inside_segment & valid_mask[:, None]).any(dim=1).sum().item()))

            if matched.any():
                center_cost = (center_t - gt_center).abs() / gt_span
                center_cost = center_cost.masked_fill(~matched, float("inf"))
                min_cost, assign_idx = center_cost.min(dim=1)
                pos = torch.isfinite(min_cost)
                one_hot = F.one_hot(gt_label.long(), self.num_classes).to(cls_targets.dtype)
                cls_targets[pos] = one_hot[assign_idx[pos]]
                assigned_segments[pos] = gt_segment[assign_idx[pos]]
                assigned_spans = gt_span.reshape(-1)[assign_idx[pos]]
                debug_state["native_head_pos_per_sample"].append(int(pos.sum().item()))
                debug_state["native_head_assigned_gt_span_mean_per_sample"].append(float(assigned_spans.mean().item()))
            else:
                debug_state["native_head_pos_per_sample"].append(0)
                debug_state["native_head_assigned_gt_span_mean_per_sample"].append(0.0)

            gt_cls.append(cls_targets)
            gt_assigned_segments.append(assigned_segments)

        return torch.stack(gt_cls), torch.stack(gt_assigned_segments), debug_state
