import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_audit_with_stubs(name):
    torch_stub = ModuleType("torch")
    torch_stub.no_grad = lambda: (lambda func: func)
    torch_stub.is_tensor = lambda obj: False
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
    torch_stub.manual_seed = lambda seed: None
    torch_stub.device = lambda value: value

    mmengine_stub = ModuleType("mmengine")
    mmengine_config_stub = ModuleType("mmengine.config")
    mmengine_config_stub.Config = SimpleNamespace(fromfile=lambda path: None)
    mmengine_stub.config = mmengine_config_stub

    opentad_stub = ModuleType("opentad")
    opentad_datasets_stub = ModuleType("opentad.datasets")
    opentad_datasets_stub.build_dataloader = lambda *args, **kwargs: None
    opentad_datasets_stub.build_dataset = lambda *args, **kwargs: None
    opentad_models_stub = ModuleType("opentad.models")
    opentad_models_stub.build_detector = lambda *args, **kwargs: None
    opentad_utils_stub = ModuleType("opentad.models.utils")
    post_processing_stub = ModuleType("opentad.models.utils.post_processing")
    post_processing_stub.convert_to_seconds = lambda segments, meta, **kwargs: segments
    post_processing_stub.selected_axis_to_dense_axis = lambda segments, meta, **kwargs: segments

    inserted = {
        "torch": torch_stub,
        "mmengine": mmengine_stub,
        "mmengine.config": mmengine_config_stub,
        "opentad": opentad_stub,
        "opentad.datasets": opentad_datasets_stub,
        "opentad.models": opentad_models_stub,
        "opentad.models.utils": opentad_utils_stub,
        "opentad.models.utils.post_processing": post_processing_stub,
    }
    old_modules = {module_name: sys.modules.get(module_name) for module_name in inserted}
    sys.modules.update(inserted)
    try:
        module_path = ROOT / "tools/audit_sparse_head_assignment.py"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_value


def cfg_with_loadframes(remap_gt_to_selected_axis, expected_axis_contract):
    axis_contract = {"expected_axis_contract": expected_axis_contract}
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train=SimpleNamespace(
                pipeline=[
                    {"type": "PrepareVideoInfo"},
                    {
                        "type": "LoadFrames",
                        "method": "random_fixed_subsample",
                        "remap_gt_to_selected_axis": remap_gt_to_selected_axis,
                    },
                ]
            )
        ),
        model=SimpleNamespace(rpn_head=SimpleNamespace(route_contract=axis_contract)),
    )


def test_same_batch_audit_rejects_mixed_native_and_selected_axis_configs():
    audit = load_audit_with_stubs("audit_contract_mixed_axis")
    native_cfg = cfg_with_loadframes(
        False,
        {"gt_axis": "native", "proposal_axis": "native", "postprocess_axis": "native"},
    )
    selected_cfg = cfg_with_loadframes(
        True,
        {"gt_axis": "selected", "proposal_axis": "selected", "postprocess_axis": "native"},
    )

    contracts = [
        audit.same_batch_config_contract(native_cfg, "train", "native_cfg.py"),
        audit.same_batch_config_contract(selected_cfg, "train", "selected_cfg.py"),
    ]

    with pytest.raises(ValueError, match="same-batch.*incompatible.*native_cfg.py.*selected_cfg.py"):
        audit.assert_same_batch_axis_compatible(contracts)


def test_same_batch_contract_rejects_route_expected_axis_that_disagrees_with_loader_remap():
    audit = load_audit_with_stubs("audit_contract_route_mismatch")
    cfg = cfg_with_loadframes(
        False,
        {"gt_axis": "selected", "proposal_axis": "selected", "postprocess_axis": "native"},
    )

    with pytest.raises(ValueError, match="expected_axis_contract.*remap_gt_to_selected_axis"):
        audit.same_batch_config_contract(cfg, "train", "bad_cfg.py")


def test_same_batch_audit_rejects_sampled_batch_axis_that_disagrees_with_config_contract():
    audit = load_audit_with_stubs("audit_contract_batch_mismatch")
    selected_cfg = cfg_with_loadframes(
        True,
        {"gt_axis": "selected", "proposal_axis": "selected", "postprocess_axis": "native"},
    )
    contract = audit.same_batch_config_contract(selected_cfg, "train", "selected_cfg.py")
    native_batch = [
        (
            0,
            {
                "metas": [
                    {
                        "video_name": "video_native_batch",
                        "irregular_native_axis": True,
                        "irregular_gt_axis": "native",
                        "irregular_proposal_axis": "native",
                        "irregular_postprocess_axis": "native",
                    }
                ]
            },
        )
    ]

    with pytest.raises(ValueError, match="sampled batch.*selected_cfg.py.*video_native_batch"):
        audit.assert_sampled_batches_match_same_batch_contract(native_batch, contract)


def test_sample_fingerprint_records_stable_hashes_for_gt_selected_positions_and_grid():
    audit = load_audit_with_stubs("audit_contract_fingerprint")
    meta = {
        "video_name": "video_001",
        "window_start": 32,
        "irregular_selected_positions": [0.0, 2.0, 5.0, 9.0],
        "irregular_selected_valid_len": 12.0,
        "irregular_native_axis": False,
        "irregular_gt_axis": "selected",
        "irregular_proposal_axis": "selected",
        "irregular_postprocess_axis": "native",
    }
    temporal_grid_list = [
        {
            "center": [[0.0, 2.0, 5.0, 9.0]],
            "valid_mask": [[True, True, True, False]],
            "fresh_mask": [[True, False, True, False]],
            "cell_left": [[1.0, 1.0, 1.5, 2.0]],
            "cell_right": [[1.0, 1.5, 2.0, 2.0]],
        }
    ]

    fingerprint = audit.make_sample_fingerprint(
        meta=meta,
        gt_segment=[[1.0, 3.5], [6.0, 8.0]],
        gt_label=[4, 7],
        temporal_grid_list=temporal_grid_list,
        sample_idx=0,
        axes={"gt_axis": "selected", "proposal_axis": "selected", "postprocess_axis": "native"},
    )
    repeat = audit.make_sample_fingerprint(
        meta=meta,
        gt_segment=[[1.0, 3.5], [6.0, 8.0]],
        gt_label=[4, 7],
        temporal_grid_list=temporal_grid_list,
        sample_idx=0,
        axes={"gt_axis": "selected", "proposal_axis": "selected", "postprocess_axis": "native"},
    )

    assert fingerprint["sha256"] == repeat["sha256"]
    assert fingerprint["gt"]["segments_sha256"]
    assert fingerprint["gt"]["labels_sha256"]
    assert fingerprint["selected_axis"]["positions_sha256"]
    assert fingerprint["selected_axis"]["positions_count"] == 4
    assert fingerprint["temporal_grid"]["levels"][0]["center_sha256"]
    assert fingerprint["temporal_grid"]["levels"][0]["valid_count"] == 3
