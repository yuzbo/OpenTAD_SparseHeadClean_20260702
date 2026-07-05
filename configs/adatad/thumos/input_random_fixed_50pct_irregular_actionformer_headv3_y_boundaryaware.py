_base_ = ["./input_random_fixed_50pct_irregular_actionformer_headv3_y.py"]

model = dict(
    rpn_head=dict(
        boundary_loss_weight=0.2,
        boundary_inference=dict(
            enabled=True,
            peak_kernel=3,
            peak_ratio=0.5,
            score_thresh=0.05,
            preselect_topk_per_level=96,
            bank_topk=192,
            candidate_topk=12,
            stale_factor=0.6,
            min_symmetry=0.25,
            symmetry_power=0.5,
            window_scale_factor=2.5,
            window_length_factor=0.5,
            center_guard_factor=0.25,
            min_duration_factor=0.25,
            boundary_score_weight=1.0,
            distance_weight=1.0,
            duration_weight=0.75,
            center_weight=0.5,
            containment_weight=0.5,
            scale_weight=0.25,
            score_alpha=0.35,
            refine_strength_power=1.0,
        )
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_headv3_y_boundaryaware"
