_base_ = ["./input_random_fixed_50pct_irregular_actionformer.py"]

# R05D v2: Multi-scale native physical decoder with IrregularFPN
# FIX: regression_range calibrated for physical coordinate space [0, 768)
# In remap=False mode, point_scale ~ 2*stride (not 1*stride as in remap=True)
# So raw regression_range values must be ~halved to get equivalent physical coverage
#
# Physical coverage per level:
#   L0: [0, 8]    L1: [4, 16]   L2: [8, 32]
#   L3: [16, 64]  L4: [32, 128] L5: [64, 768]

model = dict(
    projection=dict(
        arch=(2, 2, 5),  # 6 FPN levels
        path_pdrop=0.1,
    ),
    neck=dict(
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
        type="NativePhysicalMultiScaleHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="IrregularPointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            # Calibrated for physical coords: point_scale ~ 2*stride
            # Physical coverage: [0,8] [4,16] [8,32] [16,64] [32,128] [64,768]
            regression_range=[(0, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 12)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        min_center_radius=0.0,
        label_smoothing=0.0,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        loss_weight=1.0,
        filter_similar_gt=True,
        use_regress_range=True,
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
    runtime_debug_interval=10,  # More frequent debug for v2
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_native_physical_multiscale_full_pilot_v2"
