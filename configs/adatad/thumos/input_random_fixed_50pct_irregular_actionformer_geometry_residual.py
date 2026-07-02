_base_ = ["./input_random_fixed_50pct_irregular_actionformer_step0_densehead.py"]

model = dict(
    rpn_head=dict(
        type="GeometryResidualCalibrator",
        geo_hidden_dim=64,
        init_gate=0.0,
        max_delta=0.25,
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_geometry_residual"
