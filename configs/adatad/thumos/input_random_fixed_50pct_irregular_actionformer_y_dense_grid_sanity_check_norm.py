_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check.py"]

model = dict(
    neck=dict(
        _delete_=True,
        type="IrregularFPNDenseAdapterNorm",
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
        output_norm_eps=1e-5,
        output_norm_affine=True,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check_norm"
