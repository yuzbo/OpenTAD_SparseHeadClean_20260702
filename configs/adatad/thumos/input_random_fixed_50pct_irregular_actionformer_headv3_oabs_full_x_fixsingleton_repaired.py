_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_legacysingleton.py"]

model = dict(
    rpn_head=dict(
        oabs_singleton_mode="expanded_span",
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_fixsingleton_repaired"
