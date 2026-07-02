_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0.py"]

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

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4"
