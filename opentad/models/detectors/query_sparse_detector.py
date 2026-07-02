import torch

from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck
from ..utils.post_processing import batched_nms, convert_to_seconds
from .base import BaseDetector


@DETECTORS.register_module()
class QuerySparseDetector(BaseDetector):
    def __init__(self, rpn_head, backbone=None, projection=None, neck=None, max_seq_len=384):
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

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        max_len = max(self.max_seq_len, feat_len)
        if feat_len < max_len:
            inputs = torch.nn.functional.pad(inputs, [0, max_len - feat_len], value=0)

        if masks.shape[1] < max_len:
            pad_masks = masks.new_zeros((masks.shape[0], max_len))
            pad_masks[:, : masks.shape[1]] = masks
            masks = pad_masks
        elif masks.shape[1] > max_len:
            masks = masks[:, :max_len]
        return inputs, masks.bool()

    def _project_features(self, x, masks):
        if self.with_projection:
            projected = self.projection(x, masks)
            if len(projected) == 3:
                feat_list, mask_list, _ = projected
            else:
                feat_list, mask_list = projected
        else:
            feat_list = (x,)
            mask_list = (masks,)

        if self.with_neck:
            necked = self.neck(feat_list, mask_list)
            if len(necked) == 3:
                feat_list, mask_list, _ = necked
            else:
                feat_list, mask_list = necked
        return feat_list, mask_list

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, temporal_grids=None, **kwargs):
        losses = {}
        x = self.backbone(inputs, metas=metas) if self.with_backbone else inputs
        x, masks = self.pad_data(x, masks)
        feat_list, mask_list = self._project_features(x, masks)

        losses.update(
            self.rpn_head.forward_train(
                feat_list,
                mask_list,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
        )
        losses["cost"] = sum(value for value in losses.values())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, temporal_grids=None, **kwargs):
        x = self.backbone(inputs, metas=metas) if self.with_backbone else inputs
        x, masks = self.pad_data(x, masks)
        feat_list, mask_list = self._project_features(x, masks)
        return self.rpn_head.forward_test(feat_list, mask_list, **kwargs)

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

            if getattr(post_cfg, "sliding_window", False) is False and getattr(post_cfg, "nms", None) is not None:
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
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if name.startswith("backbone") or not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "query_embed" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            dict(params=decay, weight_decay=cfg["weight_decay"], lr=cfg["lr"]),
            dict(params=no_decay, weight_decay=0.0, lr=cfg["lr"]),
        ]