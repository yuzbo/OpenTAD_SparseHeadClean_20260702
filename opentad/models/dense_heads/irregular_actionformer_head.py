import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale


@HEADS.register_module()
class IrregularActionFormerHead(nn.Module):
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
        filter_similar_gt=True,
        use_regress_range=True,
        debug_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt
        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.use_regress_range = use_regress_range
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(loss_normalizer))
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
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.feat_channels
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
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat, mask = cls_conv(cls_feat, mask)
            reg_feat, mask = reg_conv(reg_feat, mask)

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
        start = point_tensor[:, :, 0] - reg_tensor[:, :, 0] * point_tensor[:, :, 3]
        end = point_tensor[:, :, 0] + reg_tensor[:, :, 1] * point_tensor[:, :, 3]
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

    def _tensor_stats(self, tensor, name):
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        finite_count = int(finite.sum().item())
        numel = detached.numel()
        stats_tensor = detached if torch.is_floating_point(detached) or torch.is_complex(detached) else detached.to(torch.float32)
        if finite_count > 0:
            finite_tensor = stats_tensor[finite]
            return {
                f"{name}_shape": tuple(detached.shape),
                f"{name}_dtype": str(detached.dtype),
                f"{name}_numel": int(numel),
                f"{name}_finite_count": finite_count,
                f"{name}_nonfinite_count": int(numel - finite_count),
                f"{name}_min": float(finite_tensor.min().item()),
                f"{name}_max": float(finite_tensor.max().item()),
                f"{name}_mean": float(finite_tensor.mean().item()),
                f"{name}_std": float(finite_tensor.std(unbiased=False).item()),
                f"{name}_absmax": float(finite_tensor.abs().max().item()),
            }
        return {
            f"{name}_shape": tuple(detached.shape),
            f"{name}_dtype": str(detached.dtype),
            f"{name}_numel": int(numel),
            f"{name}_finite_count": 0,
            f"{name}_nonfinite_count": int(numel),
        }

    def collect_debug_state(self):
        return dict(self._latest_debug_state)

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg, target_debug = self.prepare_targets(points, gt_segments, gt_labels)

        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and(gt_cls.sum(-1) > 0, valid_mask)
        num_pos = pos_mask.sum().item()

        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)
        if self.debug_enabled:
            debug_state = dict(target_debug)
            debug_state["head_num_pos_total"] = int(num_pos)
            debug_state["head_valid_points_total"] = int(valid_mask.sum().item())
            debug_state["head_positive_ratio_total"] = float(num_pos / max(int(valid_mask.sum().item()), 1))
            debug_state["head_loss_normalizer"] = float(loss_normalizer.item() if torch.is_tensor(loss_normalizer) else loss_normalizer)
            self._latest_debug_state = debug_state
        return {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=1)
        gt_cls = []
        gt_reg = []
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
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            center_t = point[:, 0, None]
            left = center_t - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - center_t
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # Keep center sampling compatible with the admissible regression
                # range on irregular grids. Otherwise coarse levels often have
                # center hits and range hits on disjoint points.
                base_radius = self.center_sample_radius * point[:, 3, None]
                aligned_radius = 0.5 * point[:, 1, None]
                radius = torch.maximum(base_radius, aligned_radius)
                t_mins = center_pts - radius
                t_maxs = center_pts + radius
                cb_left = center_t - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - center_t
                center_seg = torch.stack((cb_left, cb_right), dim=-1)
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            max_regress_distance = reg_targets.max(-1)[0]
            if self.use_regress_range:
                inside_regress_range = torch.logical_and(
                    max_regress_distance >= point[:, 1, None],
                    max_regress_distance <= point[:, 2, None],
                )
            else:
                inside_regress_range = torch.ones_like(inside_gt_seg_mask)

            matched_mask = torch.logical_and(inside_gt_seg_mask, inside_regress_range)

            if self.debug_enabled:
                center_hits = inside_gt_seg_mask.any(dim=1)
                range_hits = inside_regress_range.any(dim=1)
                matched = matched_mask.any(dim=1)
                debug_state.setdefault("head_num_gt_per_sample", []).append(int(num_gts))
                debug_state.setdefault("head_gt_span_absmax_per_sample", []).append(
                    float((gt_segment[:, 1] - gt_segment[:, 0]).abs().max().item())
                )
                debug_state.setdefault("head_center_hits_total_per_sample", []).append(int(center_hits.sum().item()))
                debug_state.setdefault("head_range_hits_total_per_sample", []).append(int(range_hits.sum().item()))
                debug_state.setdefault("head_matched_hits_total_per_sample", []).append(int(matched.sum().item()))
                for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                    level_center = center_hits[start_idx:end_idx]
                    level_range = range_hits[start_idx:end_idx]
                    level_matched = matched[start_idx:end_idx]
                    debug_state.setdefault(f"head_level{level_idx}_center_hits_per_sample", []).append(
                        int(level_center.sum().item())
                    )
                    debug_state.setdefault(f"head_level{level_idx}_range_hits_per_sample", []).append(
                        int(level_range.sum().item())
                    )
                    debug_state.setdefault(f"head_level{level_idx}_matched_hits_per_sample", []).append(
                        int(level_matched.sum().item())
                    )

            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = matched_mask.to(reg_targets.dtype) @ gt_label_one_hot
            cls_targets.clamp_(min=0.0, max=1.0)

            # Choose one GT for regression by matching the point to the GT whose scale
            # is closest to the center of this point's admissible regression range.
            range_mid = 0.5 * (point[:, 1, None] + point[:, 2, None])
            scale_cost = (max_regress_distance - range_mid).abs()
            scale_cost.masked_fill_(matched_mask == 0, float("inf"))
            min_cost, assign_inds = scale_cost.min(dim=1)

            reg_targets = reg_targets[torch.arange(num_pts, device=point.device), assign_inds]
            valid_reg = torch.isfinite(min_cost)
            reg_targets = torch.where(valid_reg[:, None], reg_targets, reg_targets.new_zeros(reg_targets.shape))
            reg_targets[:, 0] = reg_targets[:, 0] / point[:, 3].clamp_min(1e-6)
            reg_targets[:, 1] = reg_targets[:, 1] / point[:, 3].clamp_min(1e-6)

            if self.debug_enabled:
                positive_mask = cls_targets.sum(-1) > 0
                debug_state.setdefault("head_pos_total_per_sample", []).append(int(positive_mask.sum().item()))
                if positive_mask.any():
                    pos_reg = reg_targets[positive_mask]
                    debug_state.setdefault("head_reg_target_left_absmax_per_sample", []).append(
                        float(pos_reg[:, 0].abs().max().item())
                    )
                    debug_state.setdefault("head_reg_target_right_absmax_per_sample", []).append(
                        float(pos_reg[:, 1].abs().max().item())
                    )
                    debug_state.setdefault("head_reg_target_absmax_per_sample", []).append(
                        float(pos_reg.abs().max().item())
                    )
                    debug_state.setdefault("head_reg_target_mean_per_sample", []).append(
                        float(pos_reg.mean().item())
                    )
                else:
                    debug_state.setdefault("head_reg_target_left_absmax_per_sample", []).append(0.0)
                    debug_state.setdefault("head_reg_target_right_absmax_per_sample", []).append(0.0)
                    debug_state.setdefault("head_reg_target_absmax_per_sample", []).append(0.0)
                    debug_state.setdefault("head_reg_target_mean_per_sample", []).append(0.0)

                for level_idx, (start_idx, end_idx) in enumerate(level_offsets):
                    level_pos = positive_mask[start_idx:end_idx]
                    debug_state.setdefault(f"head_level{level_idx}_pos_per_sample", []).append(int(level_pos.sum().item()))
                    if level_pos.any():
                        level_reg = reg_targets[start_idx:end_idx][level_pos]
                        debug_state.setdefault(f"head_level{level_idx}_reg_absmax_per_sample", []).append(
                            float(level_reg.abs().max().item())
                        )
                    else:
                        debug_state.setdefault(f"head_level{level_idx}_reg_absmax_per_sample", []).append(0.0)

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg, debug_state
