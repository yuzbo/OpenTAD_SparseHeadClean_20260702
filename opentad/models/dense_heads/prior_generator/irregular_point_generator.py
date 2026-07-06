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
            elif self.range_mode == "absolute":
                reg_min = torch.full_like(center, reg_range[0])
                reg_max = torch.full_like(center, reg_range[1])
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
    def __init__(
        self,
        strides,
        regression_range,
        use_offset=False,
        range_mode="hard",
        overlap_factor=0.5,
        decode_scale_mode="cell",
        radius_scale_mode="geometric_mean",
        dense_compat_mode=None,
    ):
        if dense_compat_mode is not None:
            if dense_compat_mode != "official_actionformer":
                raise ValueError(f"Unsupported dense_compat_mode: {dense_compat_mode}")
            range_mode = "absolute"
            decode_scale_mode = "level_stride"
            radius_scale_mode = "level_stride"
        super().__init__(
            strides=strides,
            regression_range=regression_range,
            use_offset=use_offset,
            range_mode=range_mode,
            overlap_factor=overlap_factor,
        )
        self.decode_scale_mode = decode_scale_mode
        self.radius_scale_mode = radius_scale_mode
        self.dense_compat_mode = dense_compat_mode

    def _constant_scale(self, center, value):
        return torch.full_like(center, float(value)).clamp_min(1e-6)

    def _single_scale(self, center, scale_left, scale_right, stride, mode):
        if mode in {"cell_span", "full_cell_span", "local_cell_span"}:
            return (scale_left + scale_right).clamp_min(1e-6)
        if mode in {"half_cell_span", "left_right_mean"}:
            return (0.5 * (scale_left + scale_right)).clamp_min(1e-6)
        if mode == "min_side":
            return torch.minimum(scale_left, scale_right).clamp_min(1e-6)
        if mode == "geometric_mean":
            return torch.sqrt((scale_left * scale_right).clamp_min(1e-12)).clamp_min(1e-6)
        if mode == "level_stride":
            return self._constant_scale(center, stride)
        if mode == "unit":
            return torch.ones_like(center).clamp_min(1e-6)
        raise ValueError(f"Unsupported scale mode: {mode}")

    def _decode_scales(self, center, scale_left, scale_right, stride):
        mode = self.decode_scale_mode
        if mode == "cell":
            return scale_left, scale_right
        if mode == "level_stride":
            scale = self._constant_scale(center, stride)
            return scale, scale
        scale = self._single_scale(center, scale_left, scale_right, stride, mode)
        return scale, scale

    def _assert_official_dense_compat_grid(self, temporal_grid, stride):
        if self.dense_compat_mode != "official_actionformer":
            return
        center = temporal_grid["center"]
        expected = torch.arange(center.shape[-1], dtype=center.dtype, device=center.device) * float(stride)
        expected = expected.reshape(1, -1).expand_as(center)
        valid_mask = temporal_grid.get("valid_mask", None)
        if valid_mask is None:
            valid_mask = torch.ones_like(center, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=center.device).bool()
        if valid_mask.any() and not torch.allclose(center[valid_mask], expected[valid_mask], atol=1e-4, rtol=1e-4):
            raise ValueError(
                "dense_compat_mode='official_actionformer' requires dense-like temporal grid centers "
                f"arange(T) * stride (stride={stride}). Use a non-dense diagnostic mode for irregular grids."
            )

    def __call__(self, feat_list, temporal_grid_list):
        pts_list = []
        for feat, temporal_grid, reg_range, stride in zip(
            feat_list, temporal_grid_list, self.regression_range, self.strides
        ):
            self._assert_official_dense_compat_grid(temporal_grid, stride)
            center = temporal_grid["center"]
            scale_left = temporal_grid["cell_left"].clamp_min(1e-6)
            scale_right = temporal_grid["cell_right"].clamp_min(1e-6)
            legacy_cell_span = (scale_left + scale_right).clamp_min(1e-6)

            reg_range = torch.as_tensor(reg_range, dtype=center.dtype, device=center.device)
            # Layout:
            # [center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale].
            # The first five fields preserve the legacy HeadV2/V3 contract; the
            # last two make range/radius scale explicit for bridge-style heads.
            if self.range_mode in {"hard", "local_cell_span", "cell_span"}:
                range_scale = legacy_cell_span
                reg_min = reg_range[0] * range_scale
                reg_max = reg_range[1] * range_scale
            elif self.range_mode in {"absolute", "open"}:
                range_scale = torch.ones_like(center).clamp_min(1e-6)
                reg_min = torch.full_like(center, reg_range[0])
                reg_max = torch.full_like(center, reg_range[1])
            elif self.range_mode == "level_stride":
                range_scale = self._constant_scale(center, stride)
                reg_min = reg_range[0] * range_scale
                reg_max = reg_range[1] * range_scale
            elif self.range_mode == "overlap_band":
                range_scale = legacy_cell_span
                center_scale = 0.5 * (reg_range[0] + reg_range[1]) * range_scale
                half_width = 0.5 * (reg_range[1] - reg_range[0]) * range_scale
                overlap_pad = self.overlap_factor * range_scale
                reg_min = (center_scale - half_width - overlap_pad).clamp_min(0.0)
                reg_max = center_scale + half_width + overlap_pad
            else:
                raise ValueError(f"Unsupported range_mode: {self.range_mode}")

            decode_left, decode_right = self._decode_scales(center, scale_left, scale_right, stride)
            radius_scale = self._single_scale(center, scale_left, scale_right, stride, self.radius_scale_mode)
            points = torch.stack(
                [center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale],
                dim=-1,
            )
            pts_list.append(points)
        return pts_list
