import torch
import torch.nn as nn

from ..builder import NECKS, build_neck


@NECKS.register_module()
class UniformTimestampWrapper(nn.Module):
    """Wrap an irregular neck but feed fake uniform timestamp spans."""

    def __init__(self, neck_cfg):
        super().__init__()
        self.inner_neck = build_neck(neck_cfg)

    def forward(self, input_list, mask_list, temporal_grid_list):
        fake_grids = []
        for grid in temporal_grid_list:
            if grid is None:
                fake_grids.append(None)
                continue

            fake = {key: value for key, value in grid.items()}
            center = fake["center"]
            fake["cell_left"] = torch.ones_like(center)
            fake["cell_right"] = torch.ones_like(center)
            fake_grids.append(fake)

        return self.inner_neck(input_list, mask_list, fake_grids)
