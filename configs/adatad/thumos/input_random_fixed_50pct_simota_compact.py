_base_ = ["./input_random_fixed_50pct_irregular_actionformer_step0_densehead.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=False,
        center_sample_radius=1000.0,
        assigner=dict(
            type="AnchorFreeSimOTAAssigner",
            cls_weight=1.0,
            iou_weight=3.0,
            center_radius=1000.0,
            topk=9,
            dynamic_k=dict(
                type="dynamic_k_matching",
                keep_percent=0.3,
            ),
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_simota_compact"
