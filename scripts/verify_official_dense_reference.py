#!/usr/bin/env python
"""Compare local dense-reference files against upstream OpenTAD.

This script is intentionally read-only. Use it when dense AdaTAD / ActionFormer
behavior must be audited against the official OpenTAD source instead of the
copies in this sparse-head route repository.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys
from urllib.request import urlopen


REFERENCE_FILES = (
    "opentad/models/dense_heads/anchor_free_head.py",
    "opentad/models/dense_heads/prior_generator/point_generator.py",
    "opentad/models/necks/fpn.py",
)
OFFICIAL_URLS = {
    "opentad/models/dense_heads/anchor_free_head.py": "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/anchor_free_head.py",
    "opentad/models/dense_heads/prior_generator/point_generator.py": "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/prior_generator/point_generator.py",
    "opentad/models/necks/fpn.py": "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/necks/fpn.py",
}
EXPECTED_SELECTED_AXIS_SANITY_CONFIG = (
    "configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py"
)


class ConfigValidationError(ValueError):
    pass


def read_local(root: Path, rel_path: str) -> str:
    return (root / rel_path).read_text(encoding="utf-8")


def read_official(args: argparse.Namespace, rel_path: str) -> str:
    if args.official_root is not None:
        return read_local(args.official_root, rel_path)
    with urlopen(OFFICIAL_URLS[rel_path], timeout=args.timeout) as response:
        return response.read().decode("utf-8")


def compare_file(local_root: Path, args: argparse.Namespace, rel_path: str) -> str:
    local_text = read_local(local_root, rel_path)
    official_text = read_official(args, rel_path)
    if local_text == official_text:
        return ""
    diff = difflib.unified_diff(
        official_text.splitlines(keepends=True),
        local_text.splitlines(keepends=True),
        fromfile=f"official/{rel_path}",
        tofile=f"local/{rel_path}",
    )
    return "".join(diff)


def get_config_value(obj, key: str, default=None):
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def require_equal(errors: list[str], name: str, actual, expected) -> None:
    if actual != expected:
        errors.append(f"{name}: expected {expected!r}, got {actual!r}")


def require_true(errors: list[str], name: str, value) -> None:
    if not bool(value):
        errors.append(f"{name}: expected truthy, got {value!r}")


def validate_official_dense_selected_axis_config(local_root: Path, config_path: Path) -> list[str]:
    try:
        from mmengine.config import Config
    except Exception as exc:
        raise ConfigValidationError(f"mmengine is required to load config: {exc}") from exc

    resolved_config = config_path if config_path.is_absolute() else local_root / config_path
    cfg = Config.fromfile(str(resolved_config))
    errors: list[str] = []

    require_equal(errors, "model.type", cfg.model.type, "IrregularActionFormer")
    require_equal(
        errors,
        "model.projection.type",
        cfg.model.projection.type,
        "DensePassthroughConv1DTransformerProj",
    )
    require_equal(errors, "model.neck.type", cfg.model.neck.type, "DensePassthroughFPNIdentity")
    require_equal(errors, "model.rpn_head.type", cfg.model.rpn_head.type, "ActionFormerHead")
    require_equal(errors, "model.rpn_head.prior_generator.type", cfg.model.rpn_head.prior_generator.type, "PointGenerator")
    require_true(errors, "post_processing.save_dict", get_config_value(cfg.post_processing, "save_dict", False))

    forbidden_values = {
        "model.rpn_head.type": {"IrregularActionFormerHeadV2", "IrregularActionFormerBridgeHead"},
        "model.rpn_head.prior_generator.type": {"IrregularPointGeneratorV2"},
        "model.projection.type": {"GridAwareConv1DTransformerProj", "IrregularConvTransformerProj"},
        "model.neck.type": {"GridAwareFPNIdentity", "IrregularFPN"},
    }
    actual_values = {
        "model.rpn_head.type": cfg.model.rpn_head.type,
        "model.rpn_head.prior_generator.type": cfg.model.rpn_head.prior_generator.type,
        "model.projection.type": cfg.model.projection.type,
        "model.neck.type": cfg.model.neck.type,
    }
    for name, forbidden in forbidden_values.items():
        if actual_values[name] in forbidden:
            errors.append(f"{name}: forbidden native/sparse bridge component {actual_values[name]!r}")

    for split in ("train", "val", "test"):
        load_step = next(
            (step for step in getattr(cfg.dataset, split).pipeline if step.get("type") == "LoadFrames"),
            None,
        )
        if load_step is None:
            errors.append(f"dataset.{split}.pipeline: missing LoadFrames")
            continue
        require_equal(errors, f"dataset.{split}.LoadFrames.method", load_step.method, "uniform_fixed_subsample")
        require_equal(errors, f"dataset.{split}.LoadFrames.keep_ratio", float(load_step.keep_ratio), 0.5)
        require_true(
            errors,
            f"dataset.{split}.LoadFrames.remap_gt_to_selected_axis",
            load_step.remap_gt_to_selected_axis,
        )

    if "official_dense_selected_axis_sanity" not in cfg.work_dir:
        errors.append(f"work_dir: expected official_dense_selected_axis_sanity marker, got {cfg.work_dir!r}")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to this local OpenTAD route repository.",
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=None,
        help="Optional local checkout of official OpenTAD. If omitted, raw GitHub URLs are fetched.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    parser.add_argument(
        "--fail-on-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return non-zero when any reference file differs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional config to validate as the official dense selected-axis sanity route. "
            f"Default Stage-2 path is {EXPECTED_SELECTED_AXIS_SANITY_CONFIG}."
        ),
    )
    parser.add_argument(
        "--skip-reference-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only run the config contract check; do not fetch or diff official dense reference files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_root = args.local_root.resolve()
    if args.skip_reference_files and args.config is None:
        print("--skip-reference-files requires --config", file=sys.stderr)
        return 2

    config_errors = []
    if args.config is not None:
        try:
            config_errors = validate_official_dense_selected_axis_config(local_root, args.config)
        except ConfigValidationError as exc:
            config_errors = [str(exc)]
        if config_errors:
            print("Official dense selected-axis sanity config failed validation:", file=sys.stderr)
            for error in config_errors:
                print(f"- {error}", file=sys.stderr)

    diffs = []
    if not args.skip_reference_files:
        for rel_path in REFERENCE_FILES:
            diff = compare_file(local_root, args, rel_path)
            if diff:
                diffs.append(diff)

    if not diffs and not config_errors:
        if args.config is not None:
            print(f"Official dense selected-axis sanity config OK: {args.config}")
        if args.skip_reference_files:
            return 0
        print("Local dense reference files match upstream OpenTAD.")
        return 0

    if diffs:
        print("\n".join(diffs))
    if config_errors:
        return 1
    if args.fail_on_diff:
        print("Dense reference drift detected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
