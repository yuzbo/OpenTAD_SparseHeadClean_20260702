_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(
    rpn_head=dict(
        regression_mode="asymmetric_log1p",
        reg_denom_floor=0.5,
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_log_n16r4"
