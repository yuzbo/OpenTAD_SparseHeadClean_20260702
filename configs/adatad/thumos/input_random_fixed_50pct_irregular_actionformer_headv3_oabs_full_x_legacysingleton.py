_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x.py"]

model = dict(
    rpn_head=dict(
        oabs_singleton_mode="legacy_zero_span",
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_legacysingleton"
