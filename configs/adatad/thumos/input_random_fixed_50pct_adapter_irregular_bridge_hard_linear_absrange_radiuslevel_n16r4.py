_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py"]

model = dict(
    rpn_head=dict(
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_legacy_full_cell_span=False,
        prior_generator=dict(
            decode_scale_mode="level_stride",
            radius_scale_mode="level_stride",
        ),
    )
)

work_dir = (
    "exps/thumos/adatad/"
    "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4"
)
