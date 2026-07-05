_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=True,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_reggate_n16r4"
