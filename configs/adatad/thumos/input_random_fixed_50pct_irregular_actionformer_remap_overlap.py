_base_ = ["./input_random_fixed_50pct_irregular_actionformer_remap.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=True,
        debug_cfg=dict(enable=True),
        prior_generator=dict(
            type="IrregularPointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            range_mode="overlap_band",
            overlap_factor=0.5,
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_remap_overlap"
