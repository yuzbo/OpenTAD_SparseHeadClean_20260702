_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py"]

model = dict(
    rpn_head=dict(
        geometry_scale=0.0,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_nogeometry_n16r4"
