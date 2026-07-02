_base_ = ["./input_random_fixed_50pct_irregular_actionformer.py"]

model = dict(
    rpn_head=dict(
        type="IrregularActionFormerHeadV2",
        predictor_kernel_size=1,
        soft_assign_topk=9,
        soft_assign_temperature=1.0,
        soft_center_cost_weight=1.0,
        soft_scale_cost_weight=0.5,
        reg_denom_floor=0.5,
        prior_generator=dict(
            type="IrregularPointGeneratorV2",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            range_mode="hard",
            overlap_factor=0.5,
        ),
        debug_cfg=dict(enable=True),
    ),
)
