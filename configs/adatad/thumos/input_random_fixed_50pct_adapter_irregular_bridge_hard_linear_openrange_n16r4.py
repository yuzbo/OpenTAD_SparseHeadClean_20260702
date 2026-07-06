_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(
    rpn_head=dict(
        center_radius_scale="full_cell_span",
        reg_denom_mode="full_cell_span",
        allow_legacy_full_cell_span=True,
        allow_center_fallback_inside_gt=True,
        prior_generator=dict(
            regression_range=[
                (0, 10000),
                (0, 10000),
                (0, 10000),
                (0, 10000),
                (0, 10000),
                (0, 10000),
            ],
            range_mode="open",
        )
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4"
