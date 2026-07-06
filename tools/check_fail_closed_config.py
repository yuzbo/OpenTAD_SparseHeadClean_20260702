import argparse
import json
import sys
from pathlib import Path

from mmengine.config import Config


BLOCKED_KEY_PATTERNS = (
    "bata_allow_diagnostic_gt_cache",
    "allow_diagnostic_gt_cache",
    "diagnostic_gt",
    "oracle_diagnostic",
    "teacher_cache",
    "teacher_inputs",
    "teacher_path",
    "raw_prediction",
    "raw_predictions",
    "load_from_raw_predictions",
    "prediction_cache_shortcut",
    "cache_shortcut",
    "load_predictions",
    "load_from_predictions",
    "prediction_folder",
    "fuse_list",
)


def _plain_value(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def is_enabled_value(value):
    value = _plain_value(value)
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _iter_items(obj):
    obj = _plain_value(obj)
    if isinstance(obj, dict):
        return obj.items()
    return None


def scan_config_object(obj, path="cfg"):
    violations = []
    obj = _plain_value(obj)

    items = _iter_items(obj)
    if items is not None:
        for key, value in items:
            key_str = str(key)
            child_path = f"{path}.{key_str}"
            key_lower = key_str.lower()
            path_lower = child_path.lower()
            if any(pattern in key_lower or pattern in path_lower for pattern in BLOCKED_KEY_PATTERNS):
                if is_enabled_value(value):
                    violations.append(
                        {
                            "path": child_path,
                            "key": key_str,
                            "value": repr(_plain_value(value)),
                            "reason": "enabled diagnostic/cache/raw-prediction/post-processing shortcut",
                        }
                    )
            violations.extend(scan_config_object(value, child_path))
        return violations

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            violations.extend(scan_config_object(value, f"{path}[{idx}]"))
    return violations


def scan_config_file(config_path):
    cfg = Config.fromfile(str(config_path))
    return scan_config_object(cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Fail-closed scanner for evaluation-time shortcut config fields.")
    parser.add_argument("configs", nargs="+", help="Config file(s) to scan.")
    parser.add_argument("--json-out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_violations = []
    for config_path in args.configs:
        violations = scan_config_file(config_path)
        for violation in violations:
            violation["config"] = str(config_path)
        all_violations.extend(violations)

    report = {"ok": not all_violations, "violations": all_violations}
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if all_violations:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
