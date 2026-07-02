_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv3_y.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            use_irregular_time_embed=True,
            time_embed_dim=5,
            time_embed_hidden=128,
            time_embed_scale=0.25,
        ),
        custom=dict(
            trainable_backbone_keywords=["time_embed"],
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_headv3_timeembed_y"
