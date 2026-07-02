_base_ = ["./input_random_fixed_50pct_adapter_irregular_dense_control.py"]

model = dict(
    projection=dict(
        input_pdrop=0.0,
    ),
)

workflow = dict(
    checkpoint_interval=10,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_dense_control_pdrop0"
