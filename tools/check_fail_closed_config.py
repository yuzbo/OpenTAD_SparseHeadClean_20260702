import argparse
import glob
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

LEGACY_COMPATIBILITY = "legacy_ablation_only"
CORRECTED_COMPATIBILITY = "dense_compatible_diagnostic_candidate"


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
        violations.extend(scan_route_contract_object(obj, path))
        violations.extend(scan_loadframes_contract_object(obj, path))
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


def _dict_get(obj, key, default=None):
    obj = _plain_value(obj)
    if isinstance(obj, dict):
        return _plain_value(obj.get(key, default))
    return default


def _is_bridge_head_like(obj):
    obj = _plain_value(obj)
    if not isinstance(obj, dict):
        return False
    head_type = str(obj.get("type", ""))
    return head_type == "IrregularActionFormerBridgeHead" or "allow_legacy_full_cell_span" in obj


def _is_v2_v3_sparse_head(obj):
    obj = _plain_value(obj)
    if not isinstance(obj, dict):
        return False
    return str(obj.get("type", "")) in {"IrregularActionFormerHeadV2", "IrregularActionFormerHeadV3"}


def _has_audited_hard_compatible_marker(obj, contract):
    return bool(_dict_get(obj, "audited_hard_compatible", False)) or bool(
        _dict_get(contract, "audited_hard_compatible", False)
    )


def _violation(path, key, value, reason):
    return {"path": f"{path}.{key}", "key": key, "value": repr(_plain_value(value)), "reason": reason}


def scan_route_contract_object(obj, path):
    obj = _plain_value(obj)
    if path.endswith(".route_contract"):
        return []
    if not isinstance(obj, dict):
        return []

    violations = []
    if _is_v2_v3_sparse_head(obj):
        contract = _dict_get(obj, "route_contract", {})
        contract = contract if isinstance(contract, dict) else {}
        dense_claim = bool(_dict_get(obj, "dense_equivalent_claim_allowed", False)) or bool(
            _dict_get(contract, "dense_equivalent_claim_allowed", False)
        )
        uses_center_fallback = bool(_dict_get(obj, "allow_center_fallback_inside_gt", False))
        assignment_mode = str(_dict_get(obj, "assignment_mode", "")).lower()
        explicitly_soft_or_weighted = "soft_assign_topk" in obj or assignment_mode in {"soft", "weighted"}
        uses_soft_route = explicitly_soft_or_weighted or not _has_audited_hard_compatible_marker(obj, contract)
        if dense_claim and (uses_soft_route or uses_center_fallback):
            violations.append(
                _violation(
                    path,
                    "route_contract.dense_equivalent_claim_allowed",
                    _dict_get(contract, "dense_equivalent_claim_allowed", _dict_get(obj, "dense_equivalent_claim_allowed")),
                    "dense-equivalent claim is forbidden for V2/V3 soft/weighted/default-assignment or missing-center fallback routes",
                )
            )

        if contract:
            contract_center_fallback = bool(_dict_get(contract, "allow_center_fallback_inside_gt", False))
            if contract_center_fallback != uses_center_fallback:
                violations.append(
                    _violation(
                        f"{path}.route_contract",
                        "allow_center_fallback_inside_gt",
                        contract_center_fallback,
                        "route_contract contradicts V2/V3 allow_center_fallback_inside_gt",
                    )
                )
        return violations

    if not _is_bridge_head_like(obj):
        return []

    contract = _dict_get(obj, "route_contract", {})
    contract = contract if isinstance(contract, dict) else {}
    uses_legacy_full_cell_span = bool(_dict_get(obj, "allow_legacy_full_cell_span", False))
    uses_center_fallback = bool(_dict_get(obj, "allow_center_fallback_inside_gt", False))
    legacy_scale_modes = {"full_cell_span"}
    uses_legacy_full_cell_span = uses_legacy_full_cell_span or _dict_get(obj, "center_radius_scale") in legacy_scale_modes
    uses_legacy_full_cell_span = uses_legacy_full_cell_span or _dict_get(obj, "reg_denom_mode") in legacy_scale_modes
    uses_legacy_route = uses_legacy_full_cell_span or uses_center_fallback

    dense_claim = bool(_dict_get(obj, "dense_equivalent_claim_allowed", False)) or bool(
        _dict_get(contract, "dense_equivalent_claim_allowed", False)
    )
    if uses_legacy_route and dense_claim:
        violations.append(
            _violation(
                path,
                "route_contract.dense_equivalent_claim_allowed",
                _dict_get(contract, "dense_equivalent_claim_allowed", _dict_get(obj, "dense_equivalent_claim_allowed")),
                "dense-equivalent claim is forbidden for legacy full-cell-span or missing-center fallback routes",
            )
        )

    if uses_legacy_route and not contract:
        violations.append(
            _violation(
                path,
                "route_contract",
                contract,
                "legacy full-cell-span or missing-center fallback route must declare route_contract metadata",
            )
        )
        return violations

    if not contract:
        return violations

    contract_legacy_full = bool(_dict_get(contract, "allow_legacy_full_cell_span", False))
    contract_center_fallback = bool(_dict_get(contract, "allow_center_fallback_inside_gt", False))
    compatibility = _dict_get(contract, "compatibility")

    if contract_legacy_full != uses_legacy_full_cell_span:
        violations.append(
            _violation(
                f"{path}.route_contract",
                "allow_legacy_full_cell_span",
                contract_legacy_full,
                "route_contract contradicts bridge allow_legacy_full_cell_span/scale modes",
            )
        )
    if contract_center_fallback != uses_center_fallback:
        violations.append(
            _violation(
                f"{path}.route_contract",
                "allow_center_fallback_inside_gt",
                contract_center_fallback,
                "route_contract contradicts bridge allow_center_fallback_inside_gt",
            )
        )
    expected_compatibility = LEGACY_COMPATIBILITY if uses_legacy_route else CORRECTED_COMPATIBILITY
    if compatibility not in {expected_compatibility, None}:
        violations.append(
            _violation(
                f"{path}.route_contract",
                "compatibility",
                compatibility,
                f"route_contract compatibility must be {expected_compatibility!r} for this bridge route",
            )
        )
    return violations


def scan_loadframes_contract_object(obj, path):
    obj = _plain_value(obj)
    if not isinstance(obj, dict) or str(obj.get("type", "")) != "LoadFrames":
        return []

    violations = []
    remaps_selected_gt = bool(_dict_get(obj, "remap_gt_to_selected_axis", True))
    allows_gt_drop = bool(_dict_get(obj, "allow_drop_selected_axis_gt", False))
    if remaps_selected_gt and allows_gt_drop:
        diagnostic_opt_in = bool(_dict_get(obj, "legacy_selected_axis_gt_drop_diagnostic", False)) or bool(
            _dict_get(obj, "diagnostic_only", False)
        )
        if not diagnostic_opt_in:
            violations.append(
                _violation(
                    path,
                    "allow_drop_selected_axis_gt",
                    allows_gt_drop,
                    "selected-axis GT drop/collapse opt-in requires an explicit legacy/diagnostic marker",
                )
            )
    return violations


def scan_config_file(config_path):
    cfg = Config.fromfile(str(config_path))
    return scan_config_object(cfg)


def expand_config_paths(configs):
    expanded = []
    for config in configs:
        matches = sorted(glob.glob(config))
        if matches:
            expanded.extend(Path(match) for match in matches)
        else:
            expanded.append(Path(config))
    return expanded


def parse_args():
    parser = argparse.ArgumentParser(description="Fail-closed scanner for evaluation-time shortcut config fields.")
    parser.add_argument("configs", nargs="+", help="Config file(s) to scan.")
    parser.add_argument("--json-out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_violations = []
    for config_path in expand_config_paths(args.configs):
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
