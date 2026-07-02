_base_ = ["./input_random_fixed_50pct_irregular_actionformer_step0_densehead.py"]

window_size = 384
scale_factor = 1
chunk_num = window_size * scale_factor // 16

model = dict(
    _delete_=True,
    type="QuerySparseDetector",
    max_seq_len=window_size,
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerCP",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=False,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=True,
            freeze_backbone=True,
        ),
    ),
    projection=None,
    neck=None,
    rpn_head=dict(
        type="QueryDecoderHead",
        num_classes=20,
        in_channels=384,
        d_model=256,
        n_queries=30,
        n_layers=3,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        cls_loss=dict(type="FocalLoss"),
        reg_loss=dict(type="DIOULoss"),
        loss_weight=1.0,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_query_sparse_pilot"