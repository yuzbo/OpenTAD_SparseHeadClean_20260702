_base_ = ["./input_random_fixed_50pct_irregular_actionformer.py"]

model = dict(
    neck=None,
    projection=dict(
        arch=(2, 2, 0),
        path_pdrop=0.1,
    ),
    rpn_head=dict(
        _delete_=True,
        type="NativePhysicalPointHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample_radius=2.0,
        min_center_radius=1.0,
        cls_loss_weight=0.5,
        reg_loss_weight=1.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        debug_cfg=dict(enable=True),
    ),
)

workflow = dict(
    end_epoch=60,
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=30,
    runtime_debug_interval=200,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_native_physical_point_full_pilot"
