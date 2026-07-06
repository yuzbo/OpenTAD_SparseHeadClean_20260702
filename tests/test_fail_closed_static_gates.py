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
