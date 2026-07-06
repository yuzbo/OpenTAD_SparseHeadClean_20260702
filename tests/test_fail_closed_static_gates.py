import ast
import re
import subprocess
import sys
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
    assert '"neck": "IrregularFPNDenseAdapter"' in stage3_text
    assert '"neck": "GridAwareFPNIdentity"' in stage3_text
    assert '"dense_compat_mode": "official_actionformer"' in stage3_text
    assert '"compatibility": "irregular_geometry_diagnostic_candidate"' in stage3_text
    assert "RUN_ASSIGNMENT_AUDIT" in stage3_text
    assert "tools/audit_sparse_head_assignment.py" in stage3_text
    assert "--configs \"$SELECTED_BRIDGE_CFG\"" in stage3_text
    assert "--configs \"$NATIVE_BRIDGE_CFG\"" in stage3_text
    assert '--configs "${CONFIGS[@]}"' not in stage3_text


def test_gpu1_same_batch_runner_compares_open_abs_levelstride_not_radiuslevel():
    runner = (ROOT / "remote_runs/run_gpu1_same_batch_audit_20260705.sh").read_text(encoding="utf-8")

    assert "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py" in runner
    assert "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py" in runner
    assert "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py" in runner
    assert "input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py" not in runner


def test_stage2_resource_boundary_short_gpu1_but_long_slurm_only():
    short_runner = (ROOT / "remote_runs/run_gpu1_stage2_dense_selected_axis_short_20260706.sh").read_text(
        encoding="utf-8"
    )
    short_launcher = (ROOT / "remote_runs/launch_gpu1_stage2_dense_selected_axis_short_20260706.sh").read_text(
        encoding="utf-8"
    )
    slurm_body = (ROOT / "remote_runs/sbatch_stage2_dense_selected_axis_train_20260706.sh").read_text(
        encoding="utf-8"
    )
    submitter = (ROOT / "remote_runs/submit_stage2_dense_selected_axis_long_slurm_20260706.sh").read_text(
        encoding="utf-8"
    )

    for text in (short_runner, slurm_body, submitter):
        assert "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py" in text
        assert "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py" in text

    assert "ALLOW_GPU1_SHORT_VALIDATION" in short_runner
    assert 'SLURM_JOB_ID:-}" != "1118197"' in short_runner
    assert "workflow.end_epoch=2" in short_runner
    assert "workflow.val_start_epoch=1" in short_runner
    assert "work_dir=$short_work_dir" in short_runner
    assert "RUN_STAGE4_AFTER" in short_runner
    assert "run_stage4_detection_quality_stage2_dense_20260706.sh" in short_runner
    assert "unset CUDA_VISIBLE_DEVICES" in short_runner
    assert "REQUIRE_RESULTS=1" in short_runner
    assert "ALLOW_OVERWRITE_STAGE2_OUTPUT" in short_runner
    assert "prepare_stage2_output_dir" in short_runner
    assert '"$ROOT"/exps/thumos/adatad/*/gpu1_id*' in short_runner
    assert 'rm -rf "$run_dir"' in short_runner
    assert "srun --jobid=1118197 --overlap -w g0030" in short_launcher
    assert "export CUDA_VISIBLE_DEVICES=1" in short_launcher

    assert "#SBATCH --gpus=1" in slurm_body
    assert "#SBATCH --exclude=g0030" in slurm_body
    assert "tools/check_fail_closed_config.py" in slurm_body
    assert "tools/train.py" in slurm_body
    assert "torchrun" in slurm_body
    assert "RUN_STAGE4_AFTER" in slurm_body
    assert "run_stage4_detection_quality_stage2_dense_20260706.sh" in slurm_body
    assert "unset CUDA_VISIBLE_DEVICES" in slurm_body
    assert "REQUIRE_RESULTS=1" in slurm_body
    assert "ALLOW_OVERWRITE_STAGE2_OUTPUT" in slurm_body
    assert "prepare_stage2_output_dir" in slurm_body
    assert '"$ROOT"/exps/thumos/adatad/*/gpu1_id*' in slurm_body
    assert 'rm -rf "$run_dir"' in slurm_body
    assert "srun --jobid=1118197" not in slurm_body
    assert "--overlap" not in slurm_body
    assert "export CUDA_VISIBLE_DEVICES=1" not in slurm_body
    assert 'SLURM_JOB_ID:-}" == "1118197"' in slurm_body
    assert '"$(hostname -s)" == "g0030"' in slurm_body

    assert "sbatch" in submitter
    assert '--exclude="${EXCLUDE_NODE:-g0030}"' in submitter
    assert "srun --jobid=1118197" not in submitter
    assert "--overlap" not in submitter
    assert "CUDA_VISIBLE_DEVICES=1" not in submitter


def test_legacy_gpu1_long_training_entrypoints_are_disabled():
    legacy_scripts = [
        "run_gpu1_bridge_openrange_long_20260705.sh",
        "launch_gpu1_bridge_openrange_long_20260705.sh",
        "run_gpu1_bridge_absrange_long_20260706.sh",
        "launch_gpu1_bridge_absrange_long_20260706.sh",
        "run_gpu1_bridge_absrange_expanded_long_20260706.sh",
        "launch_gpu1_bridge_absrange_expanded_long_20260706.sh",
        "run_gpu1_uniform_fixed_bridge_absrange_expanded_long_20260706.sh",
        "launch_gpu1_uniform_fixed_bridge_absrange_expanded_long_20260706.sh",
        "run_gpu1_uniform_fixed_dense_control_long_20260706.sh",
        "launch_gpu1_uniform_fixed_dense_control_long_20260706.sh",
        "watch_and_launch_gpu1_bridge_absrange_expanded_20260706.sh",
        "launch_watch_gpu1_bridge_absrange_expanded_20260706.sh",
    ]

    for script in legacy_scripts:
        text = (ROOT / "remote_runs" / script).read_text(encoding="utf-8")
        assert "DEPRECATED_GPU1_LONG_TRAINING_DISABLED" in text, script
        assert "exit 64" in text, script
        assert "tools/train.py" not in text, script
        assert "torchrun" not in text, script
        assert "srun --jobid=1118197" not in text, script
        assert "CUDA_VISIBLE_DEVICES=1" not in text, script


def test_stage2_gpu1_short_waiter_requires_true_idle_gpu1_before_launch():
    waiter = (ROOT / "remote_runs/watch_and_launch_gpu1_stage2_dense_selected_axis_short_20260706.sh").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "remote_runs/launch_watch_gpu1_stage2_dense_selected_axis_short_20260706.sh").read_text(
        encoding="utf-8"
    )

    assert "GPU_INDEX=\"${GPU_INDEX:-1}\"" in waiter
    assert "GPU_MEM_FREE_MAX_MIB=\"${GPU_MEM_FREE_MAX_MIB:-100}\"" in waiter
    assert "SSH_TIMEOUT_SECONDS=\"${SSH_TIMEOUT_SECONDS:-60}\"" in waiter
    assert "ALLOWED_EXISTING_STEP_REGEX=" in waiter
    assert 'timeout "$SSH_TIMEOUT_SECONDS" /usr/bin/ssh "$NODE"' in waiter
    assert "--query-gpu=index,uuid,memory.used" in waiter
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_memory" in waiter
    assert "other_active_steps_count" in waiter
    assert "compute_apps=$apps" in waiter
    assert '[[ "$mem" -le "$GPU_MEM_FREE_MAX_MIB" && "$other_steps" -eq 0 ]]' in waiter
    assert "short_validation_already_active" in waiter
    assert "bash \"$TARGET_LAUNCHER\"" in waiter
    assert "tools/train.py" not in waiter
    assert "torchrun" not in waiter
    assert "srun --jobid=1118197" not in waiter
    assert "CUDA_VISIBLE_DEVICES=1" not in waiter

    assert "nohup bash" in launcher
    assert "watch_and_launch_gpu1_stage2_dense_selected_axis_short_20260706.sh" in launcher
    assert "pgrep -f" in launcher
    assert "waiter already running" in launcher
    assert "tools/train.py" not in launcher
    assert "torchrun" not in launcher
    assert "srun --jobid=1118197" not in launcher
    assert "CUDA_VISIBLE_DEVICES=1" not in launcher


def test_stage4_detection_quality_runner_is_cpu_only_and_stage2_scoped():
    runner = (ROOT / "remote_runs/run_stage4_detection_quality_stage2_dense_20260706.sh").read_text(
        encoding="utf-8"
    )

    assert "tools/check_fail_closed_config.py" in runner
    assert "tools/analyze_detection_quality.py" in runner
    assert "tools/summarize_stage2_dense_gate.py" in runner
    assert "input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py" in runner
    assert "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py" in runner
    assert "input_random_fixed_50pct_adapter_densehead_selected_axis_control_shortgate_n16r4" in runner
    assert "input_uniform_fixed_50pct_official_dense_selected_axis_sanity_shortgate_n16r4" in runner
    assert "result_detection.json" in runner
    assert "detection_quality_summary_" in runner
    assert "detection_quality_rows_" in runner
    assert "stage2_dense_gate.json" in runner
    assert "--brief" in runner
    assert "torchrun" not in runner
    assert "tools/train.py" not in runner
    assert "srun --jobid=1118197" not in runner
    assert "CUDA_VISIBLE_DEVICES" not in runner


def test_train_and_test_entrypoints_run_fail_closed_scan_after_cfg_options():
    for entrypoint, failure_text in (
        ("tools/train.py", "before training"),
        ("tools/test.py", "before testing"),
    ):
        source = (ROOT / entrypoint).read_text(encoding="utf-8")
        assert "from tools.check_fail_closed_config import scan_config_object" in source
        assert "violations = scan_config_object(cfg)" in source
        assert failure_text in source
        merge_pos = source.index("cfg.merge_from_dict(args.cfg_options)")
        scan_pos = source.index("enforce_fail_closed_config(cfg, args.config)")
        torch_pos = source.index("import torch")
        opentad_pos = source.index("from opentad.models import build_detector")
        ddp_pos = source.index("# DDP init")
        assert merge_pos < scan_pos < torch_pos < ddp_pos, entrypoint
        assert scan_pos < opentad_pos < ddp_pos, entrypoint


def test_train_and_test_entrypoints_fail_before_ddp_on_blocked_runtime_config(tmp_path):
    bad_config = tmp_path / "bad_runtime_shortcut.py"
    bad_config.write_text(
        "inference = dict(load_from_raw_predictions=True, fuse_list=['cached.json'])\n",
        encoding="utf-8",
    )

    for entrypoint, failure_text in (
        ("tools/train.py", "before training"),
        ("tools/test.py", "before testing"),
    ):
        completed = subprocess.run(
            [sys.executable, str(ROOT / entrypoint), str(bad_config)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode != 0
        assert "load_from_raw_predictions" in output
        assert failure_text in output
        assert "LOCAL_RANK" not in output
