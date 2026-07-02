import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS


@HEADS.register_module()
class PhysicalSegmentHead(nn.Module):
    def __init__(
        self,
        num_classes=20,
        d_model=256,
        n_layers=2,
        n_head=4,
        max_freq=100.0,
        loss_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_freq = max_freq
        self.loss_weight = loss_weight

        self.input_proj = nn.Conv1d(384, d_model, 1)
        self.point_mlp = nn.Sequential(
            nn.Linear(d_model + 128, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_head = nn.Linear(d_model, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

    def _get_pos_encoding(self, positions, d_pos=128):
        positions = positions.float()
        div = torch.exp(
            torch.arange(0, d_pos, 2, device=positions.device, dtype=positions.dtype)
            * (-math.log(self.max_freq) / d_pos)
        )
        pe = positions.new_zeros((*positions.shape, d_pos))
        pe[:, :, 0::2] = torch.sin(positions.unsqueeze(-1) * div)
        pe[:, :, 1::2] = torch.cos(positions.unsqueeze(-1) * div)
        return pe

    def _single_level_inputs(self, feat_list, mask_list, temporal_grid_list):
        if len(feat_list) != 1 or len(mask_list) != 1:
            raise ValueError("PhysicalSegmentHead expects a single feature level; set projection=None and neck=None.")

        feat = feat_list[0]
        mask = mask_list[0].bool()
        if temporal_grid_list is not None and len(temporal_grid_list) > 0:
            grid = temporal_grid_list[0]
            positions = grid["center"].to(device=feat.device)
            if "valid_mask" in grid:
                mask = mask & grid["valid_mask"].to(device=feat.device).bool()
            if "fresh_mask" in grid:
                mask = mask & grid["fresh_mask"].to(device=feat.device).bool()
        else:
            positions = torch.arange(feat.shape[-1], device=feat.device, dtype=torch.float32)
            positions = positions.unsqueeze(0).repeat(feat.shape[0], 1)
        return feat, mask, positions

    def _forward(self, feat, mask, positions):
        x = self.input_proj(feat).permute(0, 2, 1)
        pos_enc = self._get_pos_encoding(positions).to(dtype=x.dtype)
        x = self.point_mlp(torch.cat([x, pos_enc], dim=-1))
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        key_padding_mask = ~mask
        if key_padding_mask.all(dim=1).any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[key_padding_mask.all(dim=1), 0] = False

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        cls_pred = self.cls_head(x)
        reg_pred = F.softplus(self.reg_head(x))
        return cls_pred, reg_pred

    def forward_train(
        self,
        feat_list,
        mask_list,
        gt_segments=None,
        gt_labels=None,
        temporal_grid_list=None,
        **kwargs,
    ):
        if gt_segments is None or gt_labels is None:
            raise ValueError("PhysicalSegmentHead.forward_train requires gt_segments and gt_labels.")

        feat, mask, positions = self._single_level_inputs(feat_list, mask_list, temporal_grid_list)
        cls_pred, reg_pred = self._forward(feat, mask, positions)
        return self._compute_losses(cls_pred, reg_pred, positions, mask, gt_segments, gt_labels)

    def _compute_losses(self, cls_pred, reg_pred, positions, mask, gt_segments, gt_labels):
        total_cls_loss = cls_pred.new_tensor(0.0)
        total_reg_loss = reg_pred.new_tensor(0.0)
        batch_size = cls_pred.shape[0]

        for b in range(batch_size):
            valid_mask = mask[b]
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0:
                total_cls_loss = total_cls_loss + cls_pred[b].sum() * 0.0
                total_reg_loss = total_reg_loss + reg_pred[b].sum() * 0.0
                continue

            pos_b = positions[b][valid_mask].to(dtype=reg_pred.dtype)
            cls_b = cls_pred[b][valid_mask]
            reg_b = reg_pred[b][valid_mask]
            gt_seg = gt_segments[b].to(device=pos_b.device, dtype=pos_b.dtype)
            gt_lbl = gt_labels[b].to(device=pos_b.device, dtype=torch.long)

            if gt_seg.numel() == 0:
                cls_target = cls_b.new_zeros(cls_b.shape)
                total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                    cls_b, cls_target, reduction="mean"
                )
                total_reg_loss = total_reg_loss + reg_b.sum() * 0.0
                continue

            gt_start = gt_seg[:, 0]
            gt_end = gt_seg[:, 1]
            gt_center = 0.5 * (gt_start + gt_end)

            dist = (pos_b[:, None] - gt_center[None, :]).abs()
            assign_idx = dist.argmin(dim=1)
            inside = (pos_b >= gt_start[assign_idx]) & (pos_b <= gt_end[assign_idx])

            cls_target = cls_b.new_zeros(cls_b.shape)
            if inside.any():
                cls_target[inside, gt_lbl[assign_idx[inside]]] = 1.0
            total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                cls_b, cls_target, reduction="mean"
            )

            if inside.any():
                left_target = (pos_b[inside] - gt_start[assign_idx[inside]]).abs()
                right_target = (gt_end[assign_idx[inside]] - pos_b[inside]).abs()
                reg_target = torch.stack([left_target, right_target], dim=-1)
                total_reg_loss = total_reg_loss + F.l1_loss(reg_b[inside], reg_target, reduction="mean")
            else:
                total_reg_loss = total_reg_loss + reg_b.sum() * 0.0

        loss_scale = self.loss_weight / batch_size
        return {
            "cls_loss": total_cls_loss * loss_scale,
            "reg_loss": total_reg_loss * loss_scale,
        }

    def forward_test(self, feat_list, mask_list, temporal_grid_list=None, **kwargs):
        feat, mask, positions = self._single_level_inputs(feat_list, mask_list, temporal_grid_list)
        cls_pred, reg_pred = self._forward(feat, mask, positions)
        cls_pred = cls_pred.sigmoid()

        proposals_list = []
        scores_list = []
        for pos, reg, score, keep in zip(positions, reg_pred, cls_pred, mask):
            pos = pos[keep].to(dtype=reg.dtype)
            reg = reg[keep]
            start = pos - reg[:, 0]
            end = pos + reg[:, 1]
            proposals_list.append(torch.stack([start, end], dim=-1))
            scores_list.append(score[keep])

        return proposals_list, scores_list
