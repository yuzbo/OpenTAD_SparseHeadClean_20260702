import torch

from ...builder import PRIOR_GENERATORS


@PRIOR_GENERATORS.register_module()
class IrregularPointGenerator:
    def __init__(
        self,
        strides,
        regression_range,
        use_offset=False,
        range_mode="hard",
        overlap_factor=0.5,
    ):
        super().__init__()
        self.strides = strides
        self.regression_range = regression_range
        self.use_offset = use_offset
        self.range_mode = range_mode
        self.overlap_factor = overlap_factor

    def __call__(self, feat_list, temporal_grid_list):
        pts_list = []
        for feat, temporal_grid, reg_range in zip(feat_list, temporal_grid_list, self.regression_range):
            batch, _, length = feat.shape
            center = temporal_grid["center"]
            scale_left = temporal_grid["cell_left"]
            scale_right = temporal_grid["cell_right"]
            point_scale = (scale_left + scale_right).clamp_min(1e-6)

            reg_range = torch.as_tensor(reg_range, dtype=center.dtype, device=center.device)
            if self.range_mode == "hard":
                reg_min = reg_range[0] * point_scale
                reg_max = reg_range[1] * point_scale
            elif self.range_mode == "overlap_band":
                center_scale = 0.5 * (reg_range[0] + reg_range[1]) * point_scale
                half_width = 0.5 * (reg_range[1] - reg_range[0]) * point_scale
                overlap_pad = self.overlap_factor * point_scale
                reg_min = (center_scale - half_width - overlap_pad).clamp_min(0.0)
                reg_max = center_scale + half_width + overlap_pad
            else:
                raise ValueError(f"Unsupported range_mode: {self.range_mode}")

            points = torch.stack([center, reg_min, reg_max, point_scale, point_scale], dim=-1)
            pts_list.append(points)
        return pts_list


@PRIOR_GENERATORS.register_module()
class IrregularPointGeneratorV2(IrregularPointGenerator):
    def __call__(self, feat_list, temporal_grid_list):
        pts_list = []
        for feat, temporal_grid, reg_range in zip(feat_list, temporal_grid_list, self.regression_range):
            center = temporal_grid["center"]
            scale_left = temporal_grid["cell_left"].clamp_min(1e-6)
            scale_right = temporal_grid["cell_right"].clamp_min(1e-6)
            point_scale = (scale_left + scale_right).clamp_min(1e-6)

            reg_range = torch.as_tensor(reg_range, dtype=center.dtype, device=center.device)
            # HeadV2 keeps the [center, reg_min, reg_max, left_scale, right_scale]
            # point layout for interface compatibility, even though its current
            # cost-based assignment does not directly consume reg_min/reg_max.
            if self.range_mode == "hard":
                reg_min = reg_range[0] * point_scale
                reg_max = reg_range[1] * point_scale
            elif self.range_mode == "overlap_band":
                center_scale = 0.5 * (reg_range[0] + reg_range[1]) * point_scale
                half_width = 0.5 * (reg_range[1] - reg_range[0]) * point_scale
                overlap_pad = self.overlap_factor * point_scale
                reg_min = (center_scale - half_width - overlap_pad).clamp_min(0.0)
                reg_max = center_scale + half_width + overlap_pad
            else:
                raise ValueError(f"Unsupported range_mode: {self.range_mode}")

            points = torch.stack([center, reg_min, reg_max, scale_left, scale_right], dim=-1)
            pts_list.append(points)
        return pts_list
