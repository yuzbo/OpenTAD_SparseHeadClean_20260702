import torch

from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck
from .base import BaseDetector
from ..utils import build_temporal_grid, normalize_temporal_grid_input
from ..utils.post_processing import batched_nms, convert_to_seconds
from ..bricks import Scale, AffineDropPath
import torch.nn as nn


@DETECTORS.register_module()
class IrregularActionFormer(BaseDetector):
    def __init__(self, projection, rpn_head, neck=None, backbone=None, max_seq_len=384):
        super().__init__()
        if backbone is not None:
            self.backbone = build_backbone(backbone)
        if projection is not None:
            self.projection = build_projection(projection)
            max_seq_len = getattr(self.projection, "max_seq_len", max_seq_len)
        if neck is not None:
            self.neck = build_neck(neck)
        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

        self.max_seq_len = max_seq_len

    @property
    def with_backbone(self):
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def _pad_temporal_grid(self, temporal_grid, feat_len, max_len, masks):
        if temporal_grid is None:
            return None

        if torch.is_tensor(temporal_grid):
            center = temporal_grid
            fresh_mask = masks[:, :feat_len]
            cell_left = None
            cell_right = None
        else:
            center = temporal_grid["center"]
            fresh_mask = temporal_grid.get("fresh_mask", masks[:, :feat_len])
            cell_left = temporal_grid.get("cell_left", None)
            cell_right = temporal_grid.get("cell_right", None)

        if center.shape[1] == max_len:
            grid = {"center": center, "fresh_mask": fresh_mask}
            if cell_left is not None and cell_right is not None:
                grid["cell_left"] = cell_left
                grid["cell_right"] = cell_right
            return normalize_temporal_grid_input(grid, masks)

        if center.shape[1] == 1:
            gap = center.new_ones(center.shape[0], 1)
        else:
            gap = (center[:, -1:] - center[:, -2:-1]).clamp_min(1e-4)

        extra = max_len - center.shape[1]
        if extra > 0:
            steps = torch.arange(1, extra + 1, device=center.device, dtype=center.dtype)[None]
            pad_center = center[:, -1:] + gap * steps
            center = torch.cat([center, pad_center], dim=1)
            pad_fresh = fresh_mask.new_zeros(fresh_mask.shape[0], extra)
            fresh_mask = torch.cat([fresh_mask, pad_fresh], dim=1)
            if cell_left is not None and cell_right is not None:
                pad_left = cell_left[:, -1:].expand(-1, extra)
                pad_right = cell_right[:, -1:].expand(-1, extra)
                cell_left = torch.cat([cell_left, pad_left], dim=1)
                cell_right = torch.cat([cell_right, pad_right], dim=1)

        grid = {"center": center[:, :max_len], "fresh_mask": fresh_mask[:, :max_len]}
        if cell_left is not None and cell_right is not None:
            grid["cell_left"] = cell_left[:, :max_len]
            grid["cell_right"] = cell_right[:, :max_len]
        return normalize_temporal_grid_input(grid, masks)

    def _build_center_grid_from_positions(self, pos, native_end, mask):
        target_len = mask.shape[0]
        valid_points = int(pos.numel())
        if valid_points == 0:
            center = torch.zeros(target_len, device=mask.device, dtype=torch.float32)
            fresh_mask = torch.zeros(target_len, device=mask.device, dtype=torch.bool)
            return build_temporal_grid(center[None], valid_mask=mask[None], fresh_mask=fresh_mask[None])

        valid_points = min(valid_points, target_len)
        pos = pos[:valid_points].to(device=mask.device, dtype=torch.float32)
        native_end = max(float(native_end), float(pos[-1].item()) + 1.0)

        center = pos
        if valid_points < target_len:
            pad = center[-1:].repeat(target_len - valid_points)
            center = torch.cat([center, pad], dim=0)

        fresh_mask = torch.zeros(target_len, device=mask.device, dtype=torch.bool)
        fresh_mask[:valid_points] = True

        if valid_points == 1:
            left = center.new_ones(target_len)
            right = center.new_ones(target_len)
            right[0] = max(native_end - float(pos[0].item()), 1e-4)
        else:
            delta = (pos[1:] - pos[:-1]).clamp_min(1e-4)
            left = center.new_ones(target_len)
            right = center.new_ones(target_len)
            left[:valid_points] = torch.cat([delta[:1], delta], dim=0)
            right[:valid_points] = torch.cat([delta, pos.new_tensor([native_end - float(pos[-1].item())])], dim=0)

        left = left.clamp_min(1e-4)
        right = right.clamp_min(1e-4)
        if valid_points < target_len:
            left[valid_points:] = left[valid_points - 1]
            right[valid_points:] = right[valid_points - 1]

        return build_temporal_grid(
            center[None],
            valid_mask=mask[None],
            fresh_mask=fresh_mask[None],
            cell_left=left[None],
            cell_right=right[None],
        )

    def _temporal_grid_from_metas(self, metas, masks):
        if metas is None:
            return None
        if not all(("irregular_selected_positions" in meta) for meta in metas):
            return None

        grids = []
        target_len = masks.shape[1]
        for meta, mask in zip(metas, masks):
            pos = meta.get("irregular_selected_positions", None)
            valid_len = meta.get("irregular_selected_valid_len", None)
            if pos is None or valid_len is None:
                return None

            pos = torch.as_tensor(pos, device=mask.device, dtype=torch.float32)
            valid_len = max(int(round(float(valid_len))), 1)
            if pos.numel() == 0:
                pos = torch.zeros(1, device=mask.device, dtype=torch.float32)
                valid_len = 1

            if meta.get("irregular_native_axis", False):
                grids.append(self._build_center_grid_from_positions(pos, valid_len, mask))
                continue

            selected_len = max(min(valid_len, target_len), 1)
            pos = pos[:selected_len]
            if pos.numel() < target_len:
                pad = pos[-1:].repeat(target_len - pos.numel())
                pos = torch.cat([pos, pad], dim=0)

            fresh = torch.zeros(target_len, device=mask.device, dtype=torch.bool)
            fresh[:selected_len] = True
            grids.append(normalize_temporal_grid_input({"center": pos[:target_len][None], "fresh_mask": fresh[None]}, mask[None]))

        return {
            "center": torch.cat([grid["center"] for grid in grids], dim=0),
            "cell_left": torch.cat([grid["cell_left"] for grid in grids], dim=0),
            "cell_right": torch.cat([grid["cell_right"] for grid in grids], dim=0),
            "valid_mask": torch.cat([grid["valid_mask"] for grid in grids], dim=0),
            "fresh_mask": torch.cat([grid["fresh_mask"] for grid in grids], dim=0),
            "level_scale": torch.cat([grid["level_scale"] for grid in grids], dim=0),
        }

    def pad_data(self, inputs, masks, temporal_grid=None):
        feat_len = inputs.shape[-1]
        if feat_len == self.max_seq_len:
            return inputs, masks, normalize_temporal_grid_input(temporal_grid, masks)
        if feat_len < self.max_seq_len:
            max_len = self.max_seq_len
        else:
            max_len = feat_len

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        temporal_grid = self._pad_temporal_grid(temporal_grid, feat_len, max_len, pad_masks)
        return inputs, pad_masks, temporal_grid

    def _project_features(self, x, masks, temporal_grid):
        if self.with_projection:
            feat_list, mask_list, temporal_grid_list = self.projection(x, masks, temporal_grid)
        else:
            temporal_grid = normalize_temporal_grid_input(temporal_grid, masks)
            feat_list = (x,)
            mask_list = (masks.bool(),)
            temporal_grid_list = (temporal_grid,)

        if self.with_neck:
            feat_list, mask_list, temporal_grid_list = self.neck(feat_list, mask_list, temporal_grid_list)
        return feat_list, mask_list, temporal_grid_list

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, temporal_grids=None, **kwargs):
        losses = {}
        x = self.backbone(inputs, metas=metas) if self.with_backbone else inputs
        if temporal_grids is None:
            temporal_grids = self._temporal_grid_from_metas(metas, masks)
        x, masks, temporal_grid = self.pad_data(x, masks, temporal_grids)

        feat_list, mask_list, temporal_grid_list = self._project_features(x, masks, temporal_grid)

        losses.update(
            self.rpn_head.forward_train(
                feat_list,
                mask_list,
                temporal_grid_list=temporal_grid_list,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
        )
        losses["cost"] = sum(value for value in losses.values())
        return losses

    def set_train_epoch(self, curr_epoch):
        if hasattr(self, "rpn_head") and hasattr(self.rpn_head, "set_train_epoch"):
            self.rpn_head.set_train_epoch(curr_epoch)
        if hasattr(self, "backbone") and hasattr(self.backbone, "set_train_epoch"):
            self.backbone.set_train_epoch(curr_epoch)

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, temporal_grids=None, **kwargs):
        x = self.backbone(inputs, metas=metas) if self.with_backbone else inputs
        if temporal_grids is None:
            temporal_grids = self._temporal_grid_from_metas(metas, masks)
        x, masks, temporal_grid = self.pad_data(x, masks, temporal_grids)

        feat_list, mask_list, temporal_grid_list = self._project_features(x, masks, temporal_grid)

        return self.rpn_head.forward_test(
            feat_list,
            mask_list,
            temporal_grid_list=temporal_grid_list,
            **kwargs,
        )

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        rpn_proposals, rpn_scores = predictions
        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):
            segments = rpn_proposals[i].detach().cpu()
            scores = rpn_scores[i].detach().cpu()

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
            else:
                pred_prob = scores.flatten()
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)
                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs

            if post_cfg.sliding_window is False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]
            segments = convert_to_seconds(segments, metas[i])

            if isinstance(ext_cls, list):
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results:
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video
        return results

    def get_optim_groups(self, cfg):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("in_proj_weight") and isinstance(m, nn.MultiheadAttention):
                    decay.add(fpn)
                elif "output_norms" in fpn and pn in {"weight", "bias"}:
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay sets!"
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],
                "lr": cfg["lr"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "lr": cfg["lr"],
            },
        ]
        return optim_groups

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

    def collect_runtime_debug(self, data_dict, bad_param_name):
        report = {"bad_param_name": bad_param_name}
        if hasattr(self, "projection") and hasattr(self.projection, "collect_debug_state"):
            report.update(self.projection.collect_debug_state())
        if hasattr(self, "neck") and hasattr(self.neck, "collect_debug_state"):
            report.update(self.neck.collect_debug_state())
        if hasattr(self, "rpn_head") and hasattr(self.rpn_head, "collect_debug_state"):
            report.update(self.rpn_head.collect_debug_state())

        metas = data_dict.get("metas", [])
        if metas is not None:
            report["batch_video_names"] = [meta.get("video_name", "unknown") for meta in metas]
            report["batch_irregular_valid_len"] = [meta.get("irregular_selected_valid_len", None) for meta in metas]

        gt_segments = data_dict.get("gt_segments", None)
        if gt_segments is not None:
            report["batch_num_gt"] = [int(seg.shape[0]) for seg in gt_segments]
            report["batch_gt_absmax"] = [float(seg.abs().max().item()) if seg.numel() > 0 else 0.0 for seg in gt_segments]

        masks = data_dict.get("masks", None)
        if masks is not None:
            report["mask_valid_per_sample"] = [int(mask.sum().item()) for mask in masks]
            report.update(self._tensor_stats(masks.to(torch.float32), "batch_mask_tensor"))

        inputs = data_dict.get("inputs", None)
        if inputs is not None:
            report.update(self._tensor_stats(inputs, "batch_inputs"))
        return report
