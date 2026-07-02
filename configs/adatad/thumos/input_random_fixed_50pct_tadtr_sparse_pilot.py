_base_ = ["./input_random_fixed_50pct_irregular_actionformer_step0_densehead.py"]

window_size = 384
scale_factor = 1
chunk_num = window_size * scale_factor // 16

model = dict(
    _delete_=True,
    type="TadTR",
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
    projection=dict(
        type="ConvSingleProj",
        in_channels=384,
        out_channels=256,
        num_convs=1,
        conv_cfg=dict(kernel_size=1, padding=0),
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=None,
    ),
    neck=None,
    transformer=dict(
        type="TadTRTransformer",
        num_proposals=40,
        num_classes=20,
        with_act_reg=True,
        roi_size=16,
        roi_extend_ratio=0.25,
        aux_loss=True,
        position_embedding=dict(
            type="PositionEmbeddingSine",
            num_pos_feats=256,
            temperature=10000,
            offset=-0.5,
            normalize=True,
        ),
        encoder=dict(
            type="DeformableDETREncoder",
            embed_dim=256,
            num_heads=8,
            num_points=4,
            attn_dropout=0.1,
            ffn_dim=1024,
            ffn_dropout=0.1,
            num_layers=4,
            num_feature_levels=1,
            post_norm=False,
        ),
        decoder=dict(
            type="DeformableDETRDecoder",
            embed_dim=256,
            num_heads=8,
            num_points=4,
            attn_dropout=0.1,
            ffn_dim=1024,
            ffn_dropout=0.1,
            num_layers=4,
            num_feature_levels=1,
            return_intermediate=True,
        ),
        loss=dict(
            type="TadTRSetCriterion",
            num_classes=20,
            matcher=dict(
                type="HungarianMatcher",
                cost_class=6.0,
                cost_bbox=5.0,
                cost_giou=2.0,
                cost_class_type="focal_loss_cost",
                iou_type="iou",
                use_multi_class=False,
            ),
            loss_class_type="focal_loss",
            weight_dict=dict(
                loss_class=2.0,
                loss_bbox=5.0,
                loss_iou=2.0,
                loss_actionness=4.0,
            ),
            use_multi_class=False,
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_tadtr_sparse_pilot"