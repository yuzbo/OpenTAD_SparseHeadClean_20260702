_base_ = ["./input_random_fixed_50pct_adapter.py"]

# Historical adapter baseline reproduction entry.
#
# The 2026-06-04 reference run only overrode N16R4 paths and re-enabled
# checkpointing. It inherited the trainable VisionTransformerAdapter,
# pure ActionFormer detector/head, random-fixed 384-of-768 sampling, and
# sigma=0.7 post-processing from input_random_fixed_50pct_adapter.py.
thumos_root = "/data/run01/sczc063/yuzibo/thumos14"
annotation_path = thumos_root + "/annotations/thumos_14_anno.json"
class_map = thumos_root + "/annotations/category_idx.txt"
train_data_path = thumos_root + "/train"
test_data_path = thumos_root + "/test"

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
)

evaluation = dict(
    ground_truth_filename=annotation_path,
)

workflow = dict(
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_n16r4_retrain_20260604"
