_base_ = ["./input_random_fixed_50pct_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            use_irregular_time_embed=True,
            add_irregular_time_embed=False,
        ),
    ),
)

workflow = dict(
    checkpoint_interval=10,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_virtual_baseline"
