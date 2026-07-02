_base_ = ["./input_random_fixed_50pct_irregular_actionformer_step0_densehead.py"]

model = dict(
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
        max_seq_len=384,
        input_pdrop=0.2,
    ),
    neck=dict(
        _delete_=True,
        type="IrregularFPNDenseAdapter",
        in_channels=512,
        out_channels=512,
        num_levels=6,
        strides=[1, 2, 4, 8, 16, 32],
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
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check"
