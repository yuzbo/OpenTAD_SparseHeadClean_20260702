import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import Scale
from ..builder import HEADS
from .actionformer_head import ActionFormerHead


@HEADS.register_module()
class GeometryResidualCalibrator(ActionFormerHead):
    """ActionFormer head with a zero-initialized temporal geometry residual."""

    def __init__(
        self,
        *args,
        geo_hidden_dim=64,
        init_gate=0.0,
        max_delta=0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.geo_mlp = nn.Sequential(
            nn.Linear(2, geo_hidden_dim),
            nn.ReLU(),
            nn.Linear(geo_hidden_dim, 2),
        )
        self.gate = Scale(init_gate)
        self.max_delta = max_delta

    def _forward_raw(self, feat_list, mask_list):
        cls_pred = []
        reg_pred = []

        for level_idx, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for conv_idx in range(self.num_convs):
                cls_feat, mask = self.cls_convs[conv_idx](cls_feat, mask)
                reg_feat, mask = self.reg_convs[conv_idx](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[level_idx](self.reg_head(reg_feat))))

        return cls_pred, reg_pred

    def _apply_geometry_residual(self, reg_pred, mask_list, temporal_grid_list=None):
        if temporal_grid_list is None:
            return reg_pred
        if len(temporal_grid_list) != len(reg_pred):
            raise ValueError("temporal_grid_list must have one entry per FPN level.")

        calibrated = []
        for reg, mask, temporal_grid in zip(reg_pred, mask_list, temporal_grid_list):
            if temporal_grid is None:
                calibrated.append(reg)
                continue

            cell_left = temporal_grid["cell_left"].to(device=reg.device, dtype=reg.dtype)
            cell_right = temporal_grid["cell_right"].to(device=reg.device, dtype=reg.dtype)
            gap_stats = torch.stack((cell_left, cell_right), dim=-1)

            delta = self.geo_mlp(gap_stats).tanh().permute(0, 2, 1)
            if self.max_delta is not None:
                delta = delta * self.max_delta
            delta = self.gate(delta)
            delta = delta * mask.unsqueeze(1).to(delta.dtype)

            calibrated.append((reg + delta).clamp_min(0.0))

        return calibrated

    def forward_train(
        self,
        feat_list,
        mask_list,
        gt_segments,
        gt_labels,
        temporal_grid_list=None,
        **kwargs,
    ):
        cls_pred, reg_pred = self._forward_raw(feat_list, mask_list)
        reg_pred = self._apply_geometry_residual(reg_pred, mask_list, temporal_grid_list)
        points = self.prior_generator(feat_list)
        return self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)

    def forward_test(self, feat_list, mask_list, temporal_grid_list=None, **kwargs):
        cls_pred, reg_pred = self._forward_raw(feat_list, mask_list)
        reg_pred = self._apply_geometry_residual(reg_pred, mask_list, temporal_grid_list)
        points = self.prior_generator(feat_list)
        return self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)
