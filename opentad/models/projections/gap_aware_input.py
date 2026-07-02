import torch
import torch.nn as nn

from ..builder import PROJECTIONS, build_projection


@PROJECTIONS.register_module()
class GapAwareInput(nn.Module):
    """Inject gap statistics as extra channels before a standard projection."""

    def __init__(self, proj_cfg):
        super().__init__()
        self.inner_proj = build_projection(proj_cfg)
        self.gap_proj = nn.Conv1d(2, 2, 1, bias=False)
        nn.init.zeros_(self.gap_proj.weight)

    def forward(self, x, mask, temporal_grid=None):
        batch, _, length = x.shape

        if temporal_grid is not None:
            cell_left = temporal_grid.get("cell_left", torch.ones(batch, length, device=x.device, dtype=x.dtype))
            cell_right = temporal_grid.get("cell_right", torch.ones(batch, length, device=x.device, dtype=x.dtype))
            cell_left = cell_left.to(device=x.device, dtype=x.dtype)
            cell_right = cell_right.to(device=x.device, dtype=x.dtype)

            left = torch.log(cell_left.clamp(0.5, 128.0)).unsqueeze(1)
            right = torch.log(cell_right.clamp(0.5, 128.0)).unsqueeze(1)
            gap_stats = self.gap_proj(torch.cat([left, right], dim=1))
        else:
            gap_stats = x.new_zeros(batch, 2, length)

        x_aug = torch.cat([x, gap_stats.to(dtype=x.dtype)], dim=1)
        return self.inner_proj(x_aug, mask, temporal_grid)
