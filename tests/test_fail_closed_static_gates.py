import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _is_selected_axis_to_dense_axis_call(node):
    if isinstance(node.func, ast.Name):
        return node.func.id == "selected_axis_to_dense_axis"
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "selected_axis_to_dense_axis"
    return False


def test_production_selected_axis_call_sites_pass_strict_true():
    source_paths = sorted((ROOT / "opentad/models").rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py"))

    calls = []
    offenders = []
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_selected_axis_to_dense_axis_call(node):
                continue
            calls.append((source_path, node))
            strict_keywords = [kw for kw in node.keywords if kw.arg == "strict"]
            strict_true = strict_keywords and isinstance(strict_keywords[0].value, ast.Constant) and strict_keywords[0].value.value is True
            if not strict_true:
                offenders.append(f"{source_path.relative_to(ROOT)}:{node.lineno}")

    assert calls, "production code should keep selected-axis conversion call coverage"
    assert offenders == []


def test_remote_run_execution_scripts_gate_fail_closed_config_before_commands():
    command_re = re.compile(
        r"\b("
        r"tools/(?:train|test)\.py|"
        r"tools/audit_sparse_head_assignment\.py|"
        r"python(?:3)?\s+-m\s+pytest|"
        r"\$PYTHON_BIN\"\s+-m\s+pytest"
        r")\b"
    )
    py_compile_re = re.compile(r"\bpy_compile\b")
    helper_re = re.compile(r"\b(?:bash|srun|nohup)\b.*\bremote_runs/")
    gate_re = re.compile(r"\btools/check_fail_closed_config\.py\b")

    offenders = []
    for script_path in sorted((ROOT / "remote_runs").glob("*.sh")):
        lines = script_path.read_text(encoding="utf-8").splitlines()
        gate_lines = [idx for idx, line in enumerate(lines, start=1) if gate_re.search(line)]
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if py_compile_re.search(stripped) or helper_re.search(stripped):
                continue
            if command_re.search(stripped) and not any(gate_lineno < lineno for gate_lineno in gate_lines):
                offenders.append(f"{script_path.relative_to(ROOT)}:{lineno}: {stripped}")

    assert offenders == []


def test_stage23_limited_runner_scripts_are_explicit_and_preflight_only():
    stage2 = ROOT / "remote_runs/run_stage2_selected_axis_dense_sanity_near63_near65_20260706.sh"
    stage3 = ROOT / "remote_runs/run_stage3_bridge_selected_native_short_audit_20260706.sh"

    assert stage2.exists()
    assert stage3.exists()

    stage2_text = stage2.read_text(encoding="utf-8")
    stage3_text = stage3.read_text(encoding="utf-8")

    for text in (stage2_text, stage3_text):
        assert "set -euo pipefail" in text
        assert "tools/check_fail_closed_config.py" in text
        assert "FAIL_CLOSED_JSON" in text
        assert "python.exe" in text
        assert "tools/train.py" not in text
        assert "torchrun" not in text
        assert "nohup" not in text

    assert "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py" in stage2_text
    assert "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py" in stage2_text
    assert "near63_random_fixed_selected_axis_dense_control" in stage2_text
    assert "near65_uniform_even_spacing_official_dense" in stage2_text

    assert "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py" in stage3_text
    assert "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py" in stage3_text
    assert "RUN_ASSIGNMENT_AUDIT" in stage3_text
    assert "tools/audit_sparse_head_assignment.py" in stage3_text
    assert "--configs \"$SELECTED_BRIDGE_CFG\"" in stage3_text
    assert "--configs \"$NATIVE_BRIDGE_CFG\"" in stage3_text
    assert '--configs "${CONFIGS[@]}"' not in stage3_text
