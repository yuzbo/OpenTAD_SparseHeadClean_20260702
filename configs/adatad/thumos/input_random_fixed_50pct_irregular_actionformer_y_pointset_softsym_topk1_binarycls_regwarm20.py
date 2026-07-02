_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls.py"]

model = dict(
    rpn_head=dict(
        cls_loss_weight_schedule=dict(
            warmup_epochs=20,
            warmup_value=0.0,
            after_warmup_value=1.0,
        ),
    ),
)

work_dir = (
    "exps/thumos/adatad/"
    "input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_regwarm20"
)
