#!/usr/bin/env python3
"""Build lightweight paper-diagnostic plot specs from collected experiment rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METRICS = ("average_mAP", "quality_recall@0.50", "quality_recall@0.70", "quality_best_iou_mean")


def _to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _spec_key(metric):
    return metric.replace("mAP", "map")


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "rows" in data:
            return list(data["rows"])
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON input must be a list or contain a rows field: {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_figure_specs(rows, metrics=DEFAULT_METRICS):
    labels = [str(row.get("label", row.get("config_name", row.get("exp_dir", f"exp_{idx}")))) for idx, row in enumerate(rows)]
    specs = {}
    for metric in metrics:
        if not any(metric in row for row in rows):
            continue
        specs[_spec_key(metric)] = {
            "type": "bar",
            "metric": metric,
            "labels": labels,
            "values": [_to_float(row.get(metric)) for row in rows],
            "y_label": metric,
        }
    return specs


def write_specs(specs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(specs, handle, indent=2, sort_keys=True)


def plot_specs(specs, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, spec in specs.items():
        fig, ax = plt.subplots(figsize=(max(6.0, len(spec["labels"]) * 1.2), 4.0))
        ax.bar(spec["labels"], spec["values"], color="#4477aa")
        ax.set_ylabel(spec.get("y_label", metric))
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out_path = output_dir / f"{metric.replace('@', '_at_').replace('/', '_')}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        written.append(str(out_path))
    return written


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/JSON rows from tools/collect_experiment_results.py")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--plot-dir", default=None, help="Optional directory for PNG plots")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.input)
    metrics = tuple(item for item in args.metrics.split(",") if item)
    specs = build_figure_specs(rows, metrics=metrics)
    write_specs(specs, args.spec_json)
    written = []
    plot_error = None
    if args.plot_dir:
        try:
            written = plot_specs(specs, args.plot_dir)
        except Exception as exc:  # pragma: no cover - depends on remote plotting stack.
            plot_error = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"num_specs": len(specs), "plots": written}, indent=2, sort_keys=True))
    if plot_error:
        print(json.dumps({"plot_warning": plot_error}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
