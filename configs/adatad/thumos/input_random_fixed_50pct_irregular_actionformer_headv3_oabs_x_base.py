_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv3_x.py"]

model = dict(
    rpn_head=dict(
        type="IrregularActionFormerHeadV3OABS",
        oabs_boundary_gamma=1.0,
        oabs_delta_max=768.0,
        oabs_singleton_expand=1.0,
        oabs_observed_only=True,
    ),
)
