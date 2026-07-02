import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..builder import HEADS, build_loss


@HEADS.register_module()
class QueryDecoderHead(nn.Module):
    def __init__(
        self,
        num_classes=20,
        in_channels=384,
        d_model=256,
        n_queries=30,
        n_layers=3,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        cls_loss=None,
        reg_loss=None,
        loss_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_layers = n_layers
        self.loss_weight = loss_weight

        self.input_proj = nn.Identity() if in_channels == d_model else nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.query_embed = nn.Embedding(n_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.cls_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        self.cls_loss = build_loss(cls_loss) if cls_loss is not None else None
        self.reg_loss = build_loss(reg_loss) if reg_loss is not None else None

    def _forward_decoder(self, feat, mask):
        if feat.shape[1] != self.in_channels:
            raise ValueError(
                f"QueryDecoderHead expected {self.in_channels} input channels, got {feat.shape[1]}."
            )

        feat = self.input_proj(feat)
        feat = feat.permute(0, 2, 1).contiguous()
        mask = mask.bool()
        empty_mask = mask.sum(dim=1) == 0
        if empty_mask.any().item():
            mask = mask.clone()
            mask[empty_mask, 0] = True
        query = self.query_embed.weight.unsqueeze(0).expand(feat.shape[0], -1, -1)

        hs = self.decoder(tgt=query, memory=feat, memory_key_padding_mask=~mask)
        cls_pred = self.cls_head(hs)
        box_pred = self.box_head(hs).sigmoid()
        pred_segments = self._decode_segments(box_pred, mask)
        return cls_pred, box_pred, pred_segments

    def _decode_segments(self, box_pred, mask):
        valid_lens = mask.sum(dim=1).clamp(min=1).to(box_pred.dtype)
        center = box_pred[..., 0] * valid_lens[:, None]
        duration = box_pred[..., 1] * valid_lens[:, None]
        start = (center - 0.5 * duration).clamp(min=0.0)
        end = center + 0.5 * duration
        end = torch.minimum(end, valid_lens[:, None])
        return torch.stack([start, end], dim=-1)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred, box_pred, pred_segments = self._forward_decoder(feat_list[0], mask_list[0])
        return self._compute_loss(cls_pred, box_pred, pred_segments, gt_segments, gt_labels)

    def forward_test(self, feat_list, mask_list, **kwargs):
        cls_pred, _, pred_segments = self._forward_decoder(feat_list[0], mask_list[0])
        proposals = [pred_segments[i] for i in range(pred_segments.shape[0])]
        scores = [cls_pred[i].sigmoid() for i in range(cls_pred.shape[0])]
        return proposals, scores

    def _classification_loss(self, pred, target):
        if self.cls_loss is not None:
            return self.cls_loss(pred, target, reduction="sum")
        return F.binary_cross_entropy_with_logits(pred, target, reduction="sum")

    def _regression_loss(self, pred_segments, target_segments):
        if pred_segments.numel() == 0:
            return pred_segments.sum() * 0.0
        pred_segments = pred_segments.float()
        target_segments = target_segments.float()
        if self.reg_loss is not None:
            return self.reg_loss(pred_segments, target_segments, reduction="sum")
        return F.l1_loss(pred_segments, target_segments, reduction="sum")

    @torch.no_grad()
    def _match_single(self, cls_pred, pred_segments, gt_segments, gt_labels):
        if gt_segments.numel() == 0:
            return None

        device = cls_pred.device
        gt_segments = gt_segments.to(device=device, dtype=torch.float32)
        gt_labels = gt_labels.to(device=device).long().flatten()
        valid = (gt_labels >= 0) & (gt_labels < self.num_classes)
        if valid.sum().item() == 0:
            return None

        gt_segments = gt_segments[valid]
        gt_labels = gt_labels[valid]

        cls_prob = cls_pred.detach().sigmoid()
        cls_cost = -cls_prob[:, gt_labels]
        reg_cost = torch.cdist(pred_segments.detach(), gt_segments, p=1)
        cost = (reg_cost + cls_cost).float().cpu().numpy()

        pred_ind, gt_ind = linear_sum_assignment(cost)
        pred_ind = torch.as_tensor(pred_ind, dtype=torch.long, device=cls_pred.device)
        gt_ind = torch.as_tensor(gt_ind, dtype=torch.long, device=cls_pred.device)
        return pred_ind, gt_ind, gt_segments, gt_labels

    def _compute_loss(self, cls_pred, box_pred, pred_segments, gt_segments, gt_labels):
        cls_losses = []
        reg_losses = []

        for b in range(cls_pred.shape[0]):
            cls_target = cls_pred.new_zeros(self.n_queries, self.num_classes)
            match = self._match_single(cls_pred[b], pred_segments[b], gt_segments[b], gt_labels[b])

            if match is None:
                cls_loss = self._classification_loss(cls_pred[b], cls_target) / max(self.n_queries, 1)
                reg_loss = box_pred[b].sum() * 0.0
            else:
                pred_ind, gt_ind, valid_gt_segments, valid_gt_labels = match
                cls_target[pred_ind, valid_gt_labels[gt_ind]] = 1.0
                cls_loss = self._classification_loss(cls_pred[b], cls_target) / max(self.n_queries, 1)
                reg_loss = self._regression_loss(
                    pred_segments[b, pred_ind],
                    valid_gt_segments[gt_ind],
                ) / max(pred_ind.numel(), 1)

            cls_losses.append(cls_loss)
            reg_losses.append(reg_loss)

        total_cls_loss = torch.stack(cls_losses).mean()
        total_reg_loss = torch.stack(reg_losses).mean()

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = total_cls_loss.detach() / total_reg_loss.detach().clamp(min=0.01)

        return {"cls_loss": total_cls_loss, "reg_loss": total_reg_loss * loss_weight}