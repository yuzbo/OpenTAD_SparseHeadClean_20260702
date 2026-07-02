_base_ = ["./input_random_fixed_50pct_adapter_irregular_actionformer_base.py"]

model = dict(
    projection=dict(
        _delete_=True,
        type="GridAwareConv1DTransformerProj",
        in_channels=384,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=384,
        input_pdrop=0.2,
    ),
    neck=dict(
        _delete_=True,
        type="GridAwareFPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="IrregularActionFormerHeadV3",
        predictor_kernel_size=1,
        soft_assign_topk=9,
        soft_assign_temperature=1.0,
        soft_center_cost_weight=1.0,
        soft_scale_cost_weight=0.5,
        reg_denom_floor=0.5,
        geometry_hidden_channels=128,
        geometry_scale=0.25,
        boundary_loss_weight=0.2,
        boundary_sigma=1.0,
        boundary_num_convs=1,
        boundary_predictor_kernel_size=1,
        prior_generator=dict(
            type="IrregularPointGeneratorV2",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            range_mode="hard",
            overlap_factor=0.5,
        ),
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
            boundary_loss=dict(type="SoftBCELoss"),
        ),
        debug_cfg=dict(enable=True),
    ),
)

workflow = dict(
    checkpoint_interval=10,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_headv3_x"
