import torch
import torch.nn as nn

from ..builder import NECKS
from ..utils import build_temporal_grid, linear_interpolate_features
from ..projections.irregular_actionformer_proj import IrregularTemporalBlock


class _MaskedChannelNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True, clamp_absmax=None):
        super().__init__()
        self.eps = float(eps)
        self.clamp_absmax = None if clamp_absmax is None else float(clamp_absmax)
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, feat, valid_mask):
        mask = valid_mask.unsqueeze(1).to(feat.dtype)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (feat * mask).sum(dim=-1, keepdim=True) / denom
        centered = (feat - mean) * mask
        var = (centered * centered).sum(dim=-1, keepdim=True) / denom
        normed = centered / torch.sqrt(var + self.eps)
        if self.clamp_absmax is not None:
            normed = normed.clamp(min=-self.clamp_absmax, max=self.clamp_absmax)
        if self.weight is not None:
            normed = normed * self.weight + self.bias
        return normed * mask


@NECKS.register_module()
class IrregularFPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_levels,
        norm_cfg=None,
        attn_cfg=None,
        path_pdrop=0.0,
        debug_cfg=None,
    ):
        super().__init__()
        attn_cfg = {} if attn_cfg is None else dict(attn_cfg)
        debug_cfg = {} if debug_cfg is None else dict(debug_cfg)
        local_k = attn_cfg.get("local_k", 4)
        n_head = attn_cfg.get("n_head", 4)
        safe_geometry = attn_cfg.get("safe_geometry", True)
        geometry_fp32 = attn_cfg.get("geometry_fp32", True)
        rel_dt_clip = attn_cfg.get("rel_dt_clip", 64.0)
        rel_span_clip = attn_cfg.get("rel_span_clip", 8.0)
        self.debug_enabled = bool(debug_cfg.get("enable", False))
        self._latest_debug_state = {}
        self._latest_backward_state = {}

        if isinstance(in_channels, int):
            in_channels = [in_channels] * num_levels

        self.lateral_convs = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        for idx in range(num_levels):
            self.lateral_convs.append(nn.Conv1d(in_channels[idx], out_channels, kernel_size=1))
            self.output_blocks.append(
                IrregularTemporalBlock(
                    channels=out_channels,
                    num_heads=n_head,
                    neighborhood_size=local_k,
                    proj_pdrop=0.0,
                    path_pdrop=path_pdrop,
                    safe_geometry=safe_geometry,
                    geometry_fp32=geometry_fp32,
                    rel_dt_clip=rel_dt_clip,
                    rel_span_clip=rel_span_clip,
                )
            )
            self.output_blocks[-1].debug_enabled = self.debug_enabled
            self.output_blocks[-1].__dict__["debug_record_tensor"] = self._record_debug_tensor
            self.output_blocks[-1].debug_prefix = f"neck.output_block.{idx}"
            self.output_blocks[-1].agg.debug_enabled = self.debug_enabled
            self.output_blocks[-1].agg.__dict__["debug_record_tensor"] = self._record_debug_tensor
            self.output_blocks[-1].agg.__dict__["debug_record_vector"] = self._record_debug_tensor
            self.output_blocks[-1].agg.debug_prefix = f"neck.output_block.{idx}.agg"

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

    def _register_grad_hook(self, tensor, name):
        if not self.debug_enabled or not tensor.requires_grad:
            return

        def _hook(grad):
            self._latest_backward_state.update(self._tensor_stats(grad, name))

        tensor.register_hook(_hook)

    def _record_debug_tensor(self, name, tensor, register_grad=False):
        if not self.debug_enabled:
            return
        self._latest_debug_state.update(self._tensor_stats(tensor, name))
        if register_grad:
            self._register_grad_hook(tensor, f"{name}_grad")

    def collect_debug_state(self):
        state = {}
        state.update(self._latest_debug_state)
        state.update(self._latest_backward_state)
        target = None
        if len(self.lateral_convs) > 0:
            target = self.lateral_convs[0]
        if target is not None:
            state.update(self._tensor_stats(target.weight, "neck.lateral0_weight"))
            if target.weight.grad is not None:
                state.update(self._tensor_stats(target.weight.grad, "neck.lateral0_weight_grad"))
            if target.bias is not None:
                state.update(self._tensor_stats(target.bias, "neck.lateral0_bias"))
                if target.bias.grad is not None:
                    state.update(self._tensor_stats(target.bias.grad, "neck.lateral0_bias_grad"))
        return state

    def forward(self, input_list, mask_list, temporal_grid_list):
        if self.debug_enabled:
            self._latest_debug_state = {}
            self._latest_backward_state = {}
        laterals = []
        for idx, (feat, grid, lateral) in enumerate(zip(input_list, temporal_grid_list, self.lateral_convs)):
            lateral_out = lateral(feat) * grid["valid_mask"].unsqueeze(1).to(feat.dtype)
            laterals.append(lateral_out)
            if idx == 0:
                self._record_debug_tensor("neck.level0.input", feat, register_grad=True)
                self._record_debug_tensor("neck.level0.lateral", lateral_out, register_grad=True)

        for idx in range(len(laterals) - 1, 0, -1):
            upsampled = linear_interpolate_features(
                source_feat=laterals[idx],
                source_grid=temporal_grid_list[idx],
                target_grid=temporal_grid_list[idx - 1],
            )
            if idx - 1 == 0:
                self._record_debug_tensor("neck.level0.topdown", upsampled, register_grad=True)
            laterals[idx - 1] = laterals[idx - 1] + upsampled
            if idx - 1 == 0:
                self._record_debug_tensor("neck.level0.fused", laterals[idx - 1], register_grad=True)

        outputs = []
        for idx, (lateral, grid, block) in enumerate(zip(laterals, temporal_grid_list, self.output_blocks)):
            out = block(lateral, grid)
            outputs.append(out)
            if idx == 0:
                self._record_debug_tensor("neck.level0.output", out, register_grad=True)

        return tuple(outputs), tuple(mask_list), tuple(temporal_grid_list)


@NECKS.register_module()
class IrregularFPNDenseAdapter(IrregularFPN):
    """Run the irregular FPN, then resample each level onto a uniform dense grid.

    This keeps the irregular trunk computation intact while making the output
    consumable by dense heads that assume regular temporal coordinates.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_levels,
        norm_cfg=None,
        attn_cfg=None,
        path_pdrop=0.0,
        debug_cfg=None,
        strides=None,
        no_interp=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_levels=num_levels,
            norm_cfg=norm_cfg,
            attn_cfg=attn_cfg,
            path_pdrop=path_pdrop,
            debug_cfg=debug_cfg,
        )
        self.strides = strides if strides is not None else [1, 2, 4, 8, 16, 32]
        self.no_interp = no_interp

    def _build_dense_reference_grid(self, source_grid, level_idx):
        center = source_grid["center"]
        valid_mask = source_grid["valid_mask"]
        batch, length = center.shape
        stride = self.strides[level_idx]
        dense_center = (torch.arange(length, device=center.device, dtype=center.dtype) * stride)[None].repeat(batch, 1)
        return build_temporal_grid(dense_center, valid_mask=valid_mask, fresh_mask=valid_mask)

    def forward(self, input_list, mask_list, temporal_grid_list):
        outputs, mask_list, temporal_grid_list = super().forward(input_list, mask_list, temporal_grid_list)

        dense_outputs = []
        dense_masks = []
        dense_grids = []
        for level_idx, (feat, grid) in enumerate(zip(outputs, temporal_grid_list)):
            dense_grid = self._build_dense_reference_grid(grid, level_idx)
            if self.no_interp:
                # A0: keep selected/rank-time features and only replace grid coords.
                dense_feat = feat
            else:
                dense_feat = linear_interpolate_features(feat, grid, dense_grid)
            dense_outputs.append(dense_feat)
            dense_masks.append(dense_grid["valid_mask"])
            dense_grids.append(dense_grid)

        return tuple(dense_outputs), tuple(dense_masks), tuple(dense_grids)


@NECKS.register_module()
class IrregularFPNDenseAdapterNorm(IrregularFPNDenseAdapter):
    """Dense adapter with masked channel normalization after irregular->dense resampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_levels,
        norm_cfg=None,
        attn_cfg=None,
        path_pdrop=0.0,
        debug_cfg=None,
        output_norm_eps=1e-5,
        output_norm_affine=True,
        output_norm_clamp_absmax=None,
        strides=None,
        no_interp=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_levels=num_levels,
            norm_cfg=norm_cfg,
            attn_cfg=attn_cfg,
            path_pdrop=path_pdrop,
            debug_cfg=debug_cfg,
            strides=strides,
            no_interp=no_interp,
        )
        self.output_norms = nn.ModuleList(
            [
                _MaskedChannelNorm1d(
                    out_channels,
                    eps=output_norm_eps,
                    affine=output_norm_affine,
                    clamp_absmax=output_norm_clamp_absmax,
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, input_list, mask_list, temporal_grid_list):
        outputs, mask_list, temporal_grid_list = super(IrregularFPNDenseAdapter, self).forward(
            input_list, mask_list, temporal_grid_list
        )

        dense_outputs = []
        dense_masks = []
        dense_grids = []
        for level_idx, (feat, grid) in enumerate(zip(outputs, temporal_grid_list)):
            dense_grid = self._build_dense_reference_grid(grid, level_idx)
            if self.no_interp:
                # A0: keep selected/rank-time features and only replace grid coords.
                dense_feat = feat
            else:
                dense_feat = linear_interpolate_features(feat, grid, dense_grid)
            dense_feat = self.output_norms[level_idx](dense_feat, dense_grid["valid_mask"])
            dense_outputs.append(dense_feat)
            dense_masks.append(dense_grid["valid_mask"])
            dense_grids.append(dense_grid)
            if self.debug_enabled and level_idx == 0:
                self._record_debug_tensor("neck.dense_adapter_norm.level0.output", dense_feat, register_grad=True)

        return tuple(dense_outputs), tuple(dense_masks), tuple(dense_grids)
