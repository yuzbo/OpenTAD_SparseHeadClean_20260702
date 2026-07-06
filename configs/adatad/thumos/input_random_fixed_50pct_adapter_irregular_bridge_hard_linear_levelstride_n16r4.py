_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(
    rpn_head=dict(
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_legacy_full_cell_span=False,
        allow_center_fallback_inside_gt=False,
        prior_generator=dict(
            range_mode="level_stride",
            decode_scale_mode="level_stride",
            radius_scale_mode="level_stride",
        ),
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4"
