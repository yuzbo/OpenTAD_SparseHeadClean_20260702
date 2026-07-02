_base_ = ["./input_random_fixed_50pct_irregular_actionformer_remap.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=False,
        debug_cfg=dict(enable=True),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_remap_no_regrange"
