import math
import torch
import torch.nn as nn

from ..builder import PROJECTIONS
from ..bricks import AffineDropPath
from ..utils import normalize_temporal_grid_input, downsample_temporal_grid


def get_continuous_sinusoid_encoding(position, channels):
    half_dim = channels // 2
    if half_dim == 0:
        return position.new_zeros(position.shape[0], channels, position.shape[1])
    freq = torch.arange(half_dim, device=position.device, dtype=position.dtype)
    freq = torch.exp(-math.log(10000.0) * freq / max(half_dim - 1, 1))
    phase = position.unsqueeze(-1) * freq.view(1, 1, -1)
    encoding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if encoding.shape[-1] < channels:
        encoding = torch.cat([encoding, encoding.new_zeros(*encoding.shape[:-1], channels - encoding.shape[-1])], dim=-1)
    return encoding.transpose(1, 2)


def downsample_features(feat, grid):
    even_feat = feat[:, :, 0::2]
    odd_feat = feat[:, :, 1::2]
    even_valid = grid["valid_mask"][:, 0::2]
    odd_valid = grid["valid_mask"][:, 1::2]
    even_width = 0.5 * (grid["cell_left"][:, 0::2] + grid["cell_right"][:, 0::2])
    odd_width = 0.5 * (grid["cell_left"][:, 1::2] + grid["cell_right"][:, 1::2])

    if odd_feat.shape[-1] < even_feat.shape[-1]:
        odd_feat = torch.nn.functional.pad(odd_feat, (0, 1))
        odd_valid = torch.nn.functional.pad(odd_valid.to(feat.dtype), (0, 1), value=0).bool()
        odd_width = torch.nn.functional.pad(odd_width, (0, 1), value=0.0)

    even_weight = even_width * even_valid.to(feat.dtype)
    odd_weight = odd_width * odd_valid.to(feat.dtype)
    weight = (even_weight + odd_weight).clamp_min(1e-6)

    out = (even_feat * even_weight.unsqueeze(1) + odd_feat * odd_weight.unsqueeze(1)) / weight.unsqueeze(1)
    return out


class IrregularLocalAggregation(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        neighborhood_size=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        safe_geometry=True,
        geometry_fp32=True,
        rel_dt_clip=64.0,
        rel_span_clip=8.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.neighborhood_size = neighborhood_size
        self.window_size = 2 * neighborhood_size + 1
        self.scale = self.head_dim ** -0.5
        self.safe_geometry = safe_geometry
        self.geometry_fp32 = geometry_fp32
        self.rel_dt_clip = rel_dt_clip
        self.rel_span_clip = rel_span_clip

        self.query = nn.Conv1d(channels, channels, kernel_size=1)
        self.key = nn.Conv1d(channels, channels, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.geometry_proj = nn.Conv2d(3, num_heads, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)
        self.debug_enabled = False
        self.debug_record_tensor = None
        self.debug_record_vector = None
        self.debug_prefix = ""

    def _window_tensor(self, tensor, value=0.0):
        pad = (self.neighborhood_size, self.neighborhood_size)
        tensor = torch.nn.functional.pad(tensor, pad, value=value)
        return tensor.unfold(-1, self.window_size, 1)

    def forward(self, x, temporal_grid):
        batch, _, length = x.shape
        valid_mask = temporal_grid["valid_mask"]
        fresh_mask = temporal_grid["fresh_mask"]

        q = self.query(x).view(batch, self.num_heads, self.head_dim, length).permute(0, 1, 3, 2)
        k = self.key(x).view(batch, self.num_heads, self.head_dim, length)
        v = self.value(x).view(batch, self.num_heads, self.head_dim, length)

        k_window = self._window_tensor(k.flatten(0, 1), value=0.0)
        v_window = self._window_tensor(v.flatten(0, 1), value=0.0)
        k_window = k_window.view(batch, self.num_heads, self.head_dim, length, self.window_size).permute(0, 1, 3, 4, 2)
        v_window = v_window.view(batch, self.num_heads, self.head_dim, length, self.window_size).permute(0, 1, 3, 4, 2)
        v_window = torch.nan_to_num(v_window, nan=0.0, posinf=0.0, neginf=0.0)

        center = temporal_grid["center"]
        span = temporal_grid["cell_left"] + temporal_grid["cell_right"]
        norm = (temporal_grid["cell_left"] * temporal_grid["cell_right"]).clamp_min(1e-6).sqrt()

        center_window = self._window_tensor(center, value=0.0)
        span_window = self._window_tensor(span, value=1.0)
        valid_window = self._window_tensor(valid_mask.to(x.dtype), value=0.0).bool()
        fresh_window = self._window_tensor(fresh_mask.to(x.dtype), value=0.0)

        if self.safe_geometry:
            center_window = torch.where(valid_window, center_window, center.unsqueeze(-1))
            span_window = torch.where(valid_window, span_window, span.unsqueeze(-1))
            fresh_window = torch.where(valid_window, fresh_window, fresh_window.new_zeros(1))

        rel_dt = (center_window - center.unsqueeze(-1)) / norm.unsqueeze(-1)
        rel_span = torch.log(span_window.clamp_min(1e-6) / span.unsqueeze(-1).clamp_min(1e-6))
        if self.rel_dt_clip is not None:
            rel_dt = rel_dt.clamp(min=-self.rel_dt_clip, max=self.rel_dt_clip)
        if self.rel_span_clip is not None:
            rel_span = rel_span.clamp(min=-self.rel_span_clip, max=self.rel_span_clip)
        geom = torch.stack([rel_dt, rel_span, fresh_window], dim=1)
        if self.debug_enabled and self.debug_record_tensor is not None:
            prefix = self.debug_prefix
            self.debug_record_tensor(f"{prefix}.norm", norm)
            self.debug_record_tensor(f"{prefix}.rel_dt", rel_dt)
            self.debug_record_tensor(f"{prefix}.rel_span", rel_span)
            self.debug_record_tensor(f"{prefix}.geom", geom)
            self.debug_record_tensor(f"{prefix}.geometry_proj_weight", self.geometry_proj.weight)
            if self.geometry_proj.bias is not None:
                self.debug_record_tensor(f"{prefix}.geometry_proj_bias", self.geometry_proj.bias)
        if self.geometry_fp32:
            geom_bias = torch.nn.functional.conv2d(
                geom.float(),
                self.geometry_proj.weight.float(),
                self.geometry_proj.bias.float() if self.geometry_proj.bias is not None else None,
            ).to(dtype=x.dtype)
        else:
            geom_bias = self.geometry_proj(geom)
        geom_bias = geom_bias.masked_fill(~valid_window.unsqueeze(1), 0.0).permute(0, 1, 2, 3)

        attn_logits = (q.unsqueeze(-2) * self.scale * k_window).sum(dim=-1) + geom_bias
        if self.debug_enabled and self.debug_record_tensor is not None:
            prefix = self.debug_prefix
            self.debug_record_tensor(f"{prefix}.q", q)
            self.debug_record_tensor(f"{prefix}.k_window", k_window)
            self.debug_record_tensor(f"{prefix}.v_window", v_window)
            self.debug_record_tensor(f"{prefix}.geom_bias", geom_bias)
            self.debug_record_tensor(f"{prefix}.attn_logits", attn_logits)
            self.debug_record_tensor(f"{prefix}.valid_window", valid_window.to(x.dtype))
            valid_logits = attn_logits.masked_select(valid_window.unsqueeze(1))
            if self.debug_record_vector is not None:
                self.debug_record_vector(f"{prefix}.valid_logits_only", valid_logits)

        attn = attn_logits.masked_fill(~valid_window.unsqueeze(1), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        if self.debug_enabled and self.debug_record_tensor is not None:
            prefix = self.debug_prefix
            self.debug_record_tensor(f"{prefix}.attn_probs", attn, register_grad=True)
            attn_mass = attn.sum(dim=-1)
            self.debug_record_tensor(f"{prefix}.attn_row_sum", attn_mass)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * v_window).sum(dim=-2)
        out = out.permute(0, 1, 3, 2).reshape(batch, self.channels, length)
        if self.debug_enabled and self.debug_record_tensor is not None:
            prefix = self.debug_prefix
            self.debug_record_tensor(f"{prefix}.out_proj_input", out, register_grad=True)
            self.debug_record_tensor(f"{prefix}.out_proj_weight", self.out_proj.weight)
        out = self.proj_drop(self.out_proj(out))
        if self.debug_enabled and self.debug_record_tensor is not None:
            prefix = self.debug_prefix
            self.debug_record_tensor(f"{prefix}.out_proj_output", out, register_grad=True)
            if self.out_proj.weight.grad is not None:
                self.debug_record_tensor(f"{prefix}.out_proj_weight_grad", self.out_proj.weight.grad)
        out = out * valid_mask.unsqueeze(1).to(out.dtype)
        return out


class IrregularTemporalBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        neighborhood_size=4,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mlp_ratio=4.0,
        safe_geometry=True,
        geometry_fp32=True,
        rel_dt_clip=64.0,
        rel_span_clip=8.0,
    ):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.LayerNorm(channels)
        self.agg = IrregularLocalAggregation(
            channels=channels,
            num_heads=num_heads,
            neighborhood_size=neighborhood_size,
            proj_pdrop=proj_pdrop,
            safe_geometry=safe_geometry,
            geometry_fp32=geometry_fp32,
            rel_dt_clip=rel_dt_clip,
            rel_span_clip=rel_span_clip,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Dropout(proj_pdrop),
        )
        self.drop_path1 = AffineDropPath(channels, drop_prob=path_pdrop) if path_pdrop > 0 else nn.Identity()
        self.drop_path2 = AffineDropPath(channels, drop_prob=path_pdrop) if path_pdrop > 0 else nn.Identity()
        self.debug_enabled = False
        self.debug_record_tensor = None
        self.debug_prefix = ""

    def forward(self, x, temporal_grid):
        valid = temporal_grid["valid_mask"].unsqueeze(1).to(x.dtype)
        if self.debug_enabled and self.debug_record_tensor is not None:
            self.debug_record_tensor(f"{self.debug_prefix}.input", x, register_grad=True)
        attn_in = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.debug_enabled and self.debug_record_tensor is not None:
            self.debug_record_tensor(f"{self.debug_prefix}.norm1_out", attn_in, register_grad=True)
        agg_out = self.agg(attn_in, temporal_grid)
        if self.debug_enabled and self.debug_record_tensor is not None:
            self.debug_record_tensor(f"{self.debug_prefix}.agg_out", agg_out, register_grad=True)
        x = x + self.drop_path1(agg_out)
        if self.debug_enabled and self.debug_record_tensor is not None:
            self.debug_record_tensor(f"{self.debug_prefix}.post_attn_residual", x, register_grad=True)
        mlp_in = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.debug_enabled and self.debug_record_tensor is not None:
            self.debug_record_tensor(f"{self.debug_prefix}.norm2_out", mlp_in, register_grad=True)
        mlp_out = self.mlp(mlp_in) * valid
        if self.debug_enabled and self.debug_record_tensor is not None:
            self.debug_record_tensor(f"{self.debug_prefix}.mlp_out", mlp_out, register_grad=True)
        x = x + self.drop_path2(mlp_out)
        if self.debug_enabled and self.debug_record_tensor is not None:
            self.debug_record_tensor(f"{self.debug_prefix}.output_pre_mask", x, register_grad=True)
        return x * valid


@PROJECTIONS.register_module()
class IrregularConvTransformerProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),
        conv_cfg=None,
        norm_cfg=None,
        attn_cfg=None,
        path_pdrop=0.0,
        use_abs_pe=False,
        max_seq_len=2304,
        input_pdrop=0.0,
        debug_cfg=None,
    ):
        super().__init__()
        assert len(arch) == 3
        attn_cfg = {} if attn_cfg is None else dict(attn_cfg)
        debug_cfg = {} if debug_cfg is None else dict(debug_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len
        self.n_head = attn_cfg.get("n_head", 4)
        self.local_k = attn_cfg.get("local_k", 4)
        self.safe_geometry = attn_cfg.get("safe_geometry", True)
        self.geometry_fp32 = attn_cfg.get("geometry_fp32", True)
        self.rel_dt_clip = attn_cfg.get("rel_dt_clip", 64.0)
        self.rel_span_clip = attn_cfg.get("rel_span_clip", 8.0)
        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None
        self.debug_cfg = debug_cfg
        self.debug_enabled = bool(debug_cfg.get("enable", False))
        self.debug_target_embed_idx = int(debug_cfg.get("target_embed_idx", 0))
        self.debug_target_layer_idx = int(debug_cfg.get("target_layer_idx", 0))
        self._latest_debug_state = {}
        self._latest_backward_state = {}
        self._debug_bad_reports = 0
        self._debug_max_reports = int(debug_cfg.get("max_reports", 8))

        if isinstance(self.in_channels, (list, tuple)):
            self.proj = nn.ModuleList([nn.Conv1d(n_in, n_out, kernel_size=1) for n_in, n_out in zip(in_channels, out_channels)])
            proj_in_channels = sum(out_channels)
        else:
            self.proj = None
            proj_in_channels = in_channels

        self.embed = nn.ModuleList()
        for idx in range(arch[0]):
            self.embed.append(
                nn.Sequential(
                    nn.Conv1d(proj_in_channels if idx == 0 else out_channels, out_channels, kernel_size=1),
                    nn.GELU(),
                )
            )

        self.stem = nn.ModuleList(
            [
                IrregularTemporalBlock(
                    channels=out_channels,
                    num_heads=self.n_head,
                    neighborhood_size=self.local_k,
                    proj_pdrop=conv_cfg.get("proj_pdrop", 0.0) if conv_cfg is not None else 0.0,
                    path_pdrop=path_pdrop,
                    safe_geometry=self.safe_geometry,
                    geometry_fp32=self.geometry_fp32,
                    rel_dt_clip=self.rel_dt_clip,
                    rel_span_clip=self.rel_span_clip,
                )
                for _ in range(arch[1])
            ]
        )
        self.branch = nn.ModuleList(
            [
                IrregularTemporalBlock(
                    channels=out_channels,
                    num_heads=self.n_head,
                    neighborhood_size=self.local_k,
                    proj_pdrop=conv_cfg.get("proj_pdrop", 0.0) if conv_cfg is not None else 0.0,
                    path_pdrop=path_pdrop,
                    safe_geometry=self.safe_geometry,
                    geometry_fp32=self.geometry_fp32,
                    rel_dt_clip=self.rel_dt_clip,
                    rel_span_clip=self.rel_span_clip,
                )
                for _ in range(arch[2])
            ]
        )

        for idx, block in enumerate(self.stem):
            block.debug_enabled = self.debug_enabled
            block.__dict__["debug_record_tensor"] = self._record_debug_tensor
            block.debug_prefix = f"projection.stem.{idx}"
            block.agg.debug_enabled = self.debug_enabled
            block.agg.__dict__["debug_record_tensor"] = self._record_debug_tensor
            block.agg.__dict__["debug_record_vector"] = self._record_debug_vector
            block.agg.debug_prefix = f"projection.stem.{idx}.agg"
        for idx, block in enumerate(self.branch):
            block.debug_enabled = self.debug_enabled
            block.__dict__["debug_record_tensor"] = self._record_debug_tensor
            block.debug_prefix = f"projection.branch.{idx}"
            block.agg.debug_enabled = self.debug_enabled
            block.agg.__dict__["debug_record_tensor"] = self._record_debug_tensor
            block.agg.__dict__["debug_record_vector"] = self._record_debug_vector
            block.agg.debug_prefix = f"projection.branch.{idx}.agg"

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)

    def _tensor_stats(self, tensor, name):
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        finite_count = int(finite.sum().item())
        numel = detached.numel()
        stats_tensor = detached if torch.is_floating_point(detached) or torch.is_complex(detached) else detached.to(torch.float32)
        if finite_count > 0:
            finite_tensor = stats_tensor[finite]
            stats = {
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
        else:
            stats = {
                f"{name}_shape": tuple(detached.shape),
                f"{name}_dtype": str(detached.dtype),
                f"{name}_numel": int(numel),
                f"{name}_finite_count": 0,
                f"{name}_nonfinite_count": int(numel),
            }
        return stats

    def _record_debug_vector(self, name, tensor):
        if not self.debug_enabled:
            return
        self._latest_debug_state.update(self._tensor_stats(tensor, name))

    def _record_debug_tensor(self, name, tensor, register_grad=False):
        if not self.debug_enabled:
            return
        self._latest_debug_state.update(self._tensor_stats(tensor, name))
        if register_grad and tensor.requires_grad:
            self._maybe_register_embed_debug_hooks(tensor, f"{name}_grad")

    def _grid_stats(self, temporal_grid):
        center = temporal_grid["center"].detach()
        valid = temporal_grid["valid_mask"].detach()
        fresh = temporal_grid["fresh_mask"].detach()
        cell_left = temporal_grid["cell_left"].detach()
        cell_right = temporal_grid["cell_right"].detach()
        span = cell_left + cell_right
        valid_count = int(valid.sum().item())
        fresh_count = int(fresh.sum().item())
        stats = {
            "grid_valid_count": valid_count,
            "grid_fresh_count": fresh_count,
            "grid_center_min": float(center[valid].min().item()) if valid_count > 0 else None,
            "grid_center_max": float(center[valid].max().item()) if valid_count > 0 else None,
            "grid_cell_left_absmax": float(cell_left[valid].abs().max().item()) if valid_count > 0 else None,
            "grid_cell_right_absmax": float(cell_right[valid].abs().max().item()) if valid_count > 0 else None,
            "grid_span_mean": float(span[valid].mean().item()) if valid_count > 0 else None,
            "grid_span_absmax": float(span[valid].abs().max().item()) if valid_count > 0 else None,
        }
        return stats

    def _maybe_register_embed_debug_hooks(self, tensor, name):
        if not self.debug_enabled or not tensor.requires_grad:
            return

        def _hook(grad):
            self._latest_backward_state.update(self._tensor_stats(grad, name))

        tensor.register_hook(_hook)

    def _update_embed_debug_state(self, x_in, x_out, temporal_grid):
        if not self.debug_enabled:
            return
        state = {}
        state.update(self._tensor_stats(x_in, "embed0_input"))
        state.update(self._tensor_stats(x_out, "embed0_output"))
        state.update(self._grid_stats(temporal_grid))
        self._latest_debug_state = state
        self._latest_backward_state = {}
        self._maybe_register_embed_debug_hooks(x_out, "embed0_output_grad")
        self._maybe_register_embed_debug_hooks(x_in, "embed0_input_grad")

    def collect_debug_state(self):
        state = {}
        state.update(self._latest_debug_state)
        state.update(self._latest_backward_state)
        target = None
        if len(self.embed) > self.debug_target_embed_idx:
            embed_block = self.embed[self.debug_target_embed_idx]
            if len(embed_block) > self.debug_target_layer_idx and hasattr(embed_block[self.debug_target_layer_idx], "weight"):
                target = embed_block[self.debug_target_layer_idx]
        if target is not None:
            state.update(self._tensor_stats(target.weight, "embed0_weight"))
            if target.weight.grad is not None:
                state.update(self._tensor_stats(target.weight.grad, "embed0_weight_grad"))
            if target.bias is not None:
                state.update(self._tensor_stats(target.bias, "embed0_bias"))
                if target.bias.grad is not None:
                    state.update(self._tensor_stats(target.bias.grad, "embed0_bias_grad"))
        return state

    def forward(self, x, mask, temporal_grid=None):
        temporal_grid = normalize_temporal_grid_input(temporal_grid, mask)

        if self.proj is not None:
            splits = x.split(self.in_channels, dim=1)
            x = torch.cat([proj(feat) for proj, feat in zip(self.proj, splits)], dim=1)

        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        for embed_idx, embed in enumerate(self.embed):
            embed_in = x
            x = embed(x) * temporal_grid["valid_mask"].unsqueeze(1).to(x.dtype)
            if embed_idx == self.debug_target_embed_idx:
                self._update_embed_debug_state(embed_in, x, temporal_grid)

        if self.use_abs_pe:
            pe = get_continuous_sinusoid_encoding(temporal_grid["center"] / max(self.max_seq_len, 1), x.shape[1])
            x = x + pe * temporal_grid["valid_mask"].unsqueeze(1).to(x.dtype)

        for block in self.stem:
            x = block(x, temporal_grid)

        out_feats = [x]
        out_masks = [temporal_grid["valid_mask"]]
        out_grids = [temporal_grid]

        current_feat = x
        current_grid = temporal_grid
        for block in self.branch:
            current_feat = downsample_features(current_feat, current_grid)
            current_grid = downsample_temporal_grid(current_grid)
            current_feat = block(current_feat, current_grid)
            out_feats.append(current_feat)
            out_masks.append(current_grid["valid_mask"])
            out_grids.append(current_grid)

        return tuple(out_feats), tuple(out_masks), tuple(out_grids)
