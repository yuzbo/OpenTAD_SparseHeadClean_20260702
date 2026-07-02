_base_ = ["./input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity.py"]

solver = dict(
    amp=False,
    fp16_compress=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity_ampoff"
