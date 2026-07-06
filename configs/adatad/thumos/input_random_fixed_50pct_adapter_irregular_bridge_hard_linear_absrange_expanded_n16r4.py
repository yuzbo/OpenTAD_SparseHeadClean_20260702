_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(
    rpn_head=dict(
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        prior_generator=dict(
            range_mode="absolute",
            regression_range=[(0, 8), (2, 16), (4, 32), (8, 64), (16, 128), (32, 10000)],
            decode_scale_mode="level_stride",
            radius_scale_mode="level_stride",
        ),
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4"
