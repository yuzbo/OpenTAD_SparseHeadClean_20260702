_base_ = ["./input_random_fixed_50pct_irregular_actionformer.py"]

model = dict(
    projection=None,
    neck=None,
    rpn_head=dict(
        _delete_=True,
        type="PhysicalSegmentHead",
        num_classes=20,
        d_model=256,
        n_layers=2,
        n_head=4,
        max_freq=100.0,
        loss_weight=1.0,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_physical_segment_pilot"
