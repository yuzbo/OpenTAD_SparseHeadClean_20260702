import torch
import torch.nn as nn


class TimeAlignedRasterizer(nn.Module):
    """Rasterize sparse temporal tokens onto a virtual uniform time axis.

    The module keeps the external token count unchanged: features are sampled
    source -> virtual uniform axis before adapter temporal convolution, then
    sampled back virtual -> source after convolution.
    """

    def __init__(self, channels, mode="linear", eps=1e-6, residual=False, residual_init=0.0):
        super().__init__()
        if mode not in {"nearest", "linear", "content_adaptive", "identity"}:
            raise ValueError(f"Unsupported TARA mode: {mode}")
        self.mode = mode
        self.eps = eps
        self.use_branch_residual = bool(residual)
        if self.use_branch_residual:
            self.branch_scale = nn.Parameter(torch.tensor(float(residual_init)))
        else:
            self.branch_scale = None

        if mode == "content_adaptive":
            self.content_gate = nn.Conv1d(channels, 1, kernel_size=1)
            nn.init.zeros_(self.content_gate.weight)
            nn.init.zeros_(self.content_gate.bias)
            self.residual_scale = nn.Parameter(torch.zeros(1))
        else:
            self.content_gate = None
            self.residual_scale = None

    def mix_with_source(self, source, tara):
        if self.branch_scale is None:
            return tara
        return source + self.branch_scale.to(dtype=source.dtype, device=source.device) * (tara - source)

    def _source_grid(self, time_embed, length, device):
        centers = time_embed[..., 0].to(device=device, dtype=torch.float32)
        centers = centers[:, :length]
        if time_embed.shape[-1] > 1:
            valid = time_embed[..., 1].to(device=device)[:, :length] > 0.5
        else:
            valid = torch.ones_like(centers, dtype=torch.bool)
        valid = valid & torch.isfinite(centers)
        return centers, valid

    def _uniform_grid(self, source_center, source_valid):
        batch, length = source_center.shape
        uniform = source_center.new_zeros(batch, length)
        uniform_valid = torch.zeros_like(source_valid)

        for idx in range(batch):
            valid = source_valid[idx]
            count = int(valid.sum().item())
            if count <= 0:
                continue

            src = source_center[idx, valid]
            start = src[0]
            end = src[-1]
            if count == 1 or torch.isclose(start, end):
                uniform[idx, 0] = start
            else:
                uniform[idx, :count] = torch.linspace(start, end, steps=count, device=source_center.device)
            uniform_valid[idx, :count] = True
        return uniform, uniform_valid

    def _sample(self, feat, source_center, target_center, source_valid, target_valid, mode):
        # feat: [B, S, C, T]
        batch, spatial, channels, target_len = feat.shape
        out = feat.new_zeros(batch, spatial, channels, target_center.shape[1])

        for idx in range(batch):
            src_keep = source_valid[idx]
            tgt_keep = target_valid[idx]
            if src_keep.sum().item() <= 0 or tgt_keep.sum().item() <= 0:
                continue

            src_x = source_center[idx, src_keep].to(torch.float32)
            src_y = feat[idx, :, :, src_keep]
            tgt_x = target_center[idx, tgt_keep].to(torch.float32)

            order = torch.argsort(src_x)
            src_x = src_x[order]
            src_y = src_y.index_select(-1, order)

            if src_x.numel() == 1:
                out[idx, :, :, tgt_keep] = src_y[..., :1].expand(-1, -1, tgt_x.numel())
                continue

            right_idx = torch.searchsorted(src_x, tgt_x, right=False).clamp(max=src_x.numel() - 1)
            left_idx = (right_idx - 1).clamp(min=0)

            if mode == "nearest":
                left_x = src_x[left_idx]
                right_x = src_x[right_idx]
                choose_right = (tgt_x - left_x).abs() >= (right_x - tgt_x).abs()
                gather_idx = torch.where(choose_right, right_idx, left_idx)
                out[idx, :, :, tgt_keep] = src_y.index_select(-1, gather_idx)
                continue

            x0 = src_x[left_idx]
            x1 = src_x[right_idx]
            denom = (x1 - x0).clamp_min(self.eps)
            alpha = ((tgt_x - x0) / denom).clamp(0.0, 1.0)
            same = (right_idx == left_idx) | ((x1 - x0).abs() < self.eps)
            alpha = torch.where(same, torch.zeros_like(alpha), alpha)
            alpha = alpha.to(dtype=src_y.dtype)

            y0 = src_y.index_select(-1, left_idx)
            y1 = src_y.index_select(-1, right_idx)
            out[idx, :, :, tgt_keep] = y0 * (1.0 - alpha.view(1, 1, -1)) + y1 * alpha.view(1, 1, -1)

        return out

    def source_to_uniform(self, feat, time_embed):
        if self.mode == "identity":
            length = feat.shape[-1]
            source_center, source_valid = self._source_grid(time_embed, length, feat.device)
            return feat, source_center, source_valid, source_center, source_valid

        length = feat.shape[-1]
        source_center, source_valid = self._source_grid(time_embed, length, feat.device)
        uniform_center, uniform_valid = self._uniform_grid(source_center, source_valid)

        if self.mode == "content_adaptive":
            linear = self._sample(feat, source_center, uniform_center, source_valid, uniform_valid, "linear")
            nearest = self._sample(feat, source_center, uniform_center, source_valid, uniform_valid, "nearest")
            summary = linear.mean(dim=1)
            gate = torch.sigmoid(self.content_gate(summary)).unsqueeze(1)
            feat_uniform = linear + self.residual_scale.to(linear.dtype) * gate * (nearest - linear)
        else:
            feat_uniform = self._sample(feat, source_center, uniform_center, source_valid, uniform_valid, self.mode)

        return feat_uniform, uniform_center, uniform_valid, source_center, source_valid

    def uniform_to_source(self, feat, uniform_center, uniform_valid, source_center, source_valid):
        return self._sample(feat, uniform_center, source_center, uniform_valid, source_valid, "linear")
