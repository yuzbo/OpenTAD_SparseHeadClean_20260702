import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_gate_module():
    module_path = ROOT / "tools/summarize_stage2_dense_gate.py"
    spec = importlib.util.spec_from_file_location("stage2_dense_gate", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_target(root, exp_dir, average_map, training_over=True, quality=True):
    path = root / exp_dir
    path.mkdir(parents=True, exist_ok=True)
    lines = [
        "2026-07-06 Train INFO: Loaded annotations from validation subset.",
        f"2026-07-06 Train INFO: Average-mAP: {average_map:.2f} (%)",
        "2026-07-06 Train INFO: mAP at tIoU 0.30 is 70.00%",
        "2026-07-06 Train INFO: mAP at tIoU 0.70 is 30.00%",
    ]
    if training_over:
        lines.append("2026-07-06 Train INFO: Training Over...")
    (path / "log.json").write_text("\n".join(lines), encoding="utf-8")
    (path / "result_detection.json").write_text('{"results": {}}', encoding="utf-8")
    if quality:
        (path / "detection_quality_summary_test.json").write_text(
            json.dumps({"summary": {"recall@0.70": 0.5, "best_iou_mean": 0.8}}),
            encoding="utf-8",
        )


def test_stage2_dense_gate_summarizes_short_and_long_status(tmp_path):
    gate = load_gate_module()
    for target in gate.TARGETS:
        if target.split == "short":
            write_target(tmp_path, target.exp_dir, average_map=5.0)
        elif target.label == "near63_random_long":
            write_target(tmp_path, target.exp_dir, average_map=61.0)
        else:
            write_target(tmp_path, target.exp_dir, average_map=50.0)

    report = gate.summarize_stage2(tmp_path)
    targets = {item["label"]: item for item in report["targets"]}

    assert report["overall"]["can_submit_long_after_short"] is True
    assert report["overall"]["long_dense_sanity_pass"] is False
    assert targets["near63_random_long"]["threshold_pass"] is True
    assert targets["near65_uniform_long"]["threshold_pass"] is False
    assert "average_mAP_below_threshold" in targets["near65_uniform_long"]["blocked_reasons"]


def test_stage2_dense_gate_blocks_missing_quality_and_hard_errors(tmp_path):
    gate = load_gate_module()
    target = gate.TARGETS[0]
    write_target(tmp_path, target.exp_dir, average_map=5.0, quality=False)
    log_path = tmp_path / target.exp_dir / "log.json"
    log_path.write_text(log_path.read_text(encoding="utf-8") + "\nTraceback (most recent call last):\n", encoding="utf-8")

    report = gate.summarize_stage2(tmp_path)
    row = next(item for item in report["targets"] if item["label"] == target.label)

    assert row["ok"] is False
    assert "missing_artifacts" in row["blocked_reasons"]
    assert "hard_error" in row["blocked_reasons"]
    assert row["log"]["hard_errors"]["traceback"] is True
