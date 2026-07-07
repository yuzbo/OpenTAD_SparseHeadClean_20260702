import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path, name):
    module_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_experiment_results_reads_logs_quality_and_writes_tables(tmp_path):
    collector = load_module("tools/collect_experiment_results.py", "collect_experiment_results")
    exp = tmp_path / "exp_a"
    exp.mkdir()
    (exp / "log.json").write_text(
        "\n".join(
            [
                "Train INFO: Average-mAP: 42.44 (%)",
                "Train INFO: mAP at tIoU 0.30 is 65.56%",
                "Train INFO: mAP at tIoU 0.70 is 15.20%",
                "Train INFO: Training Over...",
            ]
        ),
        encoding="utf-8",
    )
    (exp / "result_detection.json").write_text(json.dumps({"results": {"v": []}}), encoding="utf-8")
    (exp / "detection_quality_summary_stage4.json").write_text(
        json.dumps({"summary": {"num_predictions": 123, "recall@0.30": 0.8, "best_iou_mean": 0.7}}),
        encoding="utf-8",
    )

    rows = collector.collect_experiments([("bridge_openrange", exp)])
    assert rows == [
        {
            "label": "bridge_openrange",
            "exp_dir": str(exp),
            "log_exists": True,
            "result_detection_exists": True,
            "training_over": True,
            "average_mAP": 42.44,
            "tiou_mAP@0.30": 65.56,
            "tiou_mAP@0.70": 15.2,
            "quality_num_predictions": 123,
            "quality_recall@0.30": 0.8,
            "quality_best_iou_mean": 0.7,
        }
    ]

    csv_path = tmp_path / "summary.csv"
    md_path = tmp_path / "summary.md"
    collector.write_csv(rows, csv_path)
    collector.write_markdown(rows, md_path)

    assert "bridge_openrange" in csv_path.read_text(encoding="utf-8")
    assert "| label |" in md_path.read_text(encoding="utf-8")


def test_detection_quality_prepost_records_nms_drop_reason(tmp_path):
    analyzer = load_module("tools/analyze_detection_quality.py", "detection_quality_prepost")
    gt = {
        "database": {
            "v1": {
                "subset": "validation",
                "annotations": [{"segment": [0.0, 10.0], "label": "A"}],
            }
        }
    }
    pre_rows = [
        {
            "video_id": "v1",
            "stage": "pre_nms",
            "label": "A",
            "score": 0.7,
            "segment_seconds": [0.0, 10.0],
        },
        {
            "video_id": "v1",
            "stage": "pre_nms",
            "label": "A",
            "score": 0.9,
            "segment_seconds": [0.0, 8.0],
        },
    ]
    post_rows = [
        {
            "video_id": "v1",
            "stage": "post_nms",
            "label": "A",
            "score": 0.9,
            "segment_seconds": [0.0, 8.0],
        }
    ]

    annotated_pre = analyzer.annotate_proposal_rows(gt, pre_rows, subset="validation")
    annotated_post = analyzer.annotate_proposal_rows(gt, post_rows, subset="validation")
    drops = analyzer.infer_nms_drop_reasons(annotated_pre, annotated_post, nms_iou_threshold=0.5)

    dropped = next(row for row in drops if row["score"] == 0.7)
    kept = next(row for row in drops if row["score"] == 0.9)
    assert dropped["best_iou"] == 1.0
    assert dropped["was_kept_by_nms"] is False
    assert dropped["nms_drop_reason"] == "suppressed_by_higher_score_overlap"
    assert dropped["suppressor_score"] == 0.9
    assert kept["was_kept_by_nms"] is True


def test_detection_quality_cli_writes_prepost_proposal_lifecycle_outputs(tmp_path, monkeypatch):
    analyzer = load_module("tools/analyze_detection_quality.py", "detection_quality_prepost_cli")
    gt_path = tmp_path / "gt.json"
    pre_path = tmp_path / "pre.jsonl"
    post_path = tmp_path / "post.jsonl"
    out_prefix = tmp_path / "proposal_lifecycle"
    gt_path.write_text(
        json.dumps(
            {
                "database": {
                    "v1": {
                        "subset": "validation",
                        "annotations": [{"segment": [0.0, 10.0], "label": "A"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pre_path.write_text(
        "\n".join(
            [
                json.dumps({"video_id": "v1", "stage": "pre_nms", "label": "A", "score": 0.7, "segment_seconds": [0, 10]}),
                json.dumps({"video_id": "v1", "stage": "pre_nms", "label": "A", "score": 0.9, "segment_seconds": [0, 8]}),
            ]
        ),
        encoding="utf-8",
    )
    post_path.write_text(
        json.dumps({"video_id": "v1", "stage": "post_nms", "label": "A", "score": 0.9, "segment_seconds": [0, 8]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_detection_quality.py",
            "--ground-truth",
            str(gt_path),
            "--pre-nms-jsonl",
            str(pre_path),
            "--post-nms-jsonl",
            str(post_path),
            "--proposal-output-prefix",
            str(out_prefix),
            "--output-json",
            str(tmp_path / "unused_detection_summary.json"),
        ],
    )
    analyzer.main()

    summary = json.loads((tmp_path / "proposal_lifecycle_summary.json").read_text(encoding="utf-8"))
    csv_text = (tmp_path / "proposal_lifecycle_nms.csv").read_text(encoding="utf-8")
    assert summary["pre_nms"]["num_proposals"] == 2
    assert summary["post_nms"]["num_proposals"] == 1
    assert summary["nms"]["nms_dropped_good_proposals"]["0.50"] == 1
    assert "suppressed_by_higher_score_overlap" in csv_text


def test_plot_detection_diagnostics_builds_specs_from_collector_rows(tmp_path):
    plotter = load_module("tools/plot_detection_diagnostics.py", "plot_detection_diagnostics")
    rows = [
        {"label": "dense_even", "average_mAP": "65.0", "quality_recall@0.70": "0.72"},
        {"label": "bridge_openrange", "average_mAP": "41.0", "quality_recall@0.70": "0.31"},
    ]

    specs = plotter.build_figure_specs(rows)

    assert specs["average_map"]["labels"] == ["dense_even", "bridge_openrange"]
    assert specs["average_map"]["values"] == [65.0, 41.0]
    assert specs["quality_recall@0.70"]["values"] == [0.72, 0.31]
