_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv3_oabs_x_base.py"]

model = dict(
    rpn_head=dict(
        oabs_mode="visible",
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_headv3_oabs_visible_x"
