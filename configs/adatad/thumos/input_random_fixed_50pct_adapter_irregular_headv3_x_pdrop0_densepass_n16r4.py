_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py"]

model = dict(
    projection=dict(
        _delete_=True,
        type="DensePassthroughConv1DTransformerProj",
        in_channels=384,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=384,
        input_pdrop=0.0,
    ),
    neck=dict(
        _delete_=True,
        type="DensePassthroughFPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_densepass_n16r4"
