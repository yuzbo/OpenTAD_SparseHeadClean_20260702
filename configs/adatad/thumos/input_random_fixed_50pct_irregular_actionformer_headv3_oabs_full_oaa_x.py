_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x.py"]

model = dict(
    rpn_head=dict(
        oaa_enabled=True,
        oaa_min_overlap_ratio=0.05,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_oaa_x"
