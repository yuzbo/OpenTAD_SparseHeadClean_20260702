_base_ = ["./input_random_fixed_50pct_adapter.py"]

annotation_path = "/root/autodl-tmp/annotations/thumos_14_anno.json"
class_map = "/root/autodl-tmp/annotations/category_idx.txt"
train_data_path = "/root/autodl-tmp/train"
test_data_path = "/root/autodl-tmp/test"

window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
        sample_stride=1,
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
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
        sample_stride=1,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                remap_gt_to_selected_axis=False,
                target_len=window_size,
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
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
        sample_stride=1,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
    type="IrregularActionFormer",
    backbone=dict(
        backbone=dict(
            total_frames=window_size * scale_factor,
            use_irregular_time_embed=True,
            add_irregular_time_embed=False,
        ),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
        ),
    ),
    projection=dict(
        _delete_=True,
        type="IrregularConvTransformerProj",
        in_channels=384,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=1, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(
            n_head=4,
            local_k=4,
            safe_geometry=True,
            geometry_fp32=True,
            rel_dt_clip=64.0,
            rel_span_clip=8.0,
        ),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=window_size,
        input_pdrop=0.2,
    ),
    neck=dict(
        _delete_=True,
        type="IrregularFPN",
        in_channels=512,
        out_channels=512,
        num_levels=6,
        attn_cfg=dict(
            n_head=4,
            local_k=4,
            safe_geometry=True,
            geometry_fp32=True,
            rel_dt_clip=64.0,
            rel_span_clip=8.0,
        ),
        path_pdrop=0.1,
    ),
    rpn_head=dict(
        _delete_=True,
        type="IrregularActionFormerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="IrregularPointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

solver = dict(
    train=dict(batch_size=8, num_workers=2),
    val=dict(batch_size=8, num_workers=2),
    test=dict(batch_size=8, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=10,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_actionformer_base"
