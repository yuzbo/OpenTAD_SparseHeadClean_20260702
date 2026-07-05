_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(
    rpn_head=dict(
        prior_generator=dict(
            regression_range=[
                (0, 10000),
                (0, 10000),
                (0, 10000),
                (0, 10000),
                (0, 10000),
                (0, 10000),
            ],
            range_mode="hard",
        )
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4"
