_base_ = ["./input_random_fixed_50pct_irregular_actionformer.py"]

model = dict(
    neck=None,
    rpn_head=dict(
        debug_cfg=dict(enable=True),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_no_neck"
