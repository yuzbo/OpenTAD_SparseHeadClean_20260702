_base_ = ["./input_random_fixed_50pct_irregular_actionformer.py"]

overfit_video = "video_validation_0000051"

window_size = 384
dense_window_size = 768
scale_factor = 1

dataset = dict(
    train=dict(
        allow_list=[overfit_video],
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="random_trunc",
                keep_ratio=0.5,
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                source_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=None,
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
)

model = dict(
    neck=None,
    projection=dict(
        arch=(2, 2, 0),
        path_pdrop=0.0,
        input_pdrop=0.0,
        debug_cfg=dict(enable=True),
    ),
    rpn_head=dict(
        _delete_=True,
        type="NativePhysicalPointHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        loss_normalizer=20,
        loss_normalizer_momentum=0.9,
        center_sample_radius=2.0,
        min_center_radius=1.0,
        cls_loss_weight=0.1,
        reg_loss_weight=1.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        debug_cfg=dict(enable=True),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=0),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    amp=False,
    fp16_compress=False,
    ema=False,
)

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.0, paramwise=True)
scheduler = dict(_delete_=True, type="MultiStepLR", milestones=[400], gamma=0.1, max_epoch=500)

workflow = dict(
    end_epoch=500,
    logging_interval=1,
    checkpoint_interval=100,
    disable_checkpoint=True,
    runtime_debug_interval=25,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_native_physical_point_single_video_fulllen_overfit"
