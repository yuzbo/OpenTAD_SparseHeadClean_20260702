from pathlib import Path
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path, name):
    module_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_torch_or_skip():
    if sys.platform.startswith("win"):
        pytest.skip("torch tensor tests run on the Linux training environment")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch import unavailable: {exc}")
    return torch


def load_end_to_end_with_test_stubs():
    class FakeTensor(list):
        def bool(self):
            return self

    torch_stub = ModuleType("torch")
    torch_stub.ones = lambda count: FakeTensor([1] * int(count))
    torch_stub.zeros = lambda count: FakeTensor([0] * int(count))
    torch_stub.cat = lambda tensors: FakeTensor([item for tensor in tensors for item in tensor])

    torch_nn_stub = ModuleType("torch.nn")
    torch_nn_functional_stub = ModuleType("torch.nn.functional")

    package_stub = ModuleType("strict_loadframes_pkg")
    package_stub.__path__ = []
    transforms_stub = ModuleType("strict_loadframes_pkg.transforms")
    transforms_stub.__path__ = []

    builder_stub = ModuleType("strict_loadframes_pkg.builder")
    builder_stub.PIPELINES = SimpleNamespace(register_module=lambda: (lambda cls: cls))

    pseudo_boundary_stub = ModuleType("strict_loadframes_pkg.transforms.pseudo_boundary")
    pseudo_boundary_stub.load_boundary_scores = lambda *args, **kwargs: None
    pseudo_boundary_stub.select_pseudo_boundary_hybrid_positions = lambda *args, **kwargs: None
    pseudo_boundary_stub.select_pseudo_boundary_snap_positions = lambda *args, **kwargs: None
    pseudo_boundary_stub.slice_global_scores_for_window = lambda *args, **kwargs: None

    boundary_acq_stub = ModuleType("strict_loadframes_pkg.transforms.boundary_acquisition")
    boundary_acq_stub.BcaConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    boundary_acq_stub.load_bata_boundary_scores = lambda *args, **kwargs: (None, {})
    boundary_acq_stub.select_bata_boundary_acquisition_positions = lambda *args, **kwargs: None
    boundary_acq_stub.slice_global_scores_for_window = lambda *args, **kwargs: None
    boundary_acq_stub.validate_bata_cache_manifest_for_loader = lambda *args, **kwargs: False

    inserted = {
        "torch": torch_stub,
        "torch.nn": torch_nn_stub,
        "torch.nn.functional": torch_nn_functional_stub,
        "strict_loadframes_pkg": package_stub,
        "strict_loadframes_pkg.transforms": transforms_stub,
        "strict_loadframes_pkg.builder": builder_stub,
        "strict_loadframes_pkg.transforms.pseudo_boundary": pseudo_boundary_stub,
        "strict_loadframes_pkg.transforms.boundary_acquisition": boundary_acq_stub,
    }
    old_modules = {name: sys.modules.get(name) for name in inserted}
    sys.modules.update(inserted)
    try:
        module_path = ROOT / "opentad/datasets/transforms/end_to_end.py"
        spec = importlib.util.spec_from_file_location(
            "strict_loadframes_pkg.transforms.end_to_end",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def load_post_utils_with_test_stubs(name):
    torch_stub = ModuleType("torch")
    torch_nn_stub = ModuleType("torch.nn")
    torch_nn_functional_stub = ModuleType("torch.nn.functional")
    inserted = {
        "torch": torch_stub,
        "torch.nn": torch_nn_stub,
        "torch.nn.functional": torch_nn_functional_stub,
    }
    old_modules = {module_name: sys.modules.get(module_name) for module_name in inserted}
    sys.modules.update(inserted)
    try:
        return load_module("opentad/models/utils/post_processing/utils.py", name)
    finally:
        for module_name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_value


def test_selected_axis_to_dense_axis_strict_rejects_missing_metadata():
    torch = import_torch_or_skip()
    post_utils = load_module("opentad/models/utils/post_processing/utils.py", "strict_post_utils_missing")
    coords = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="strict.*irregular_selected_positions"):
        post_utils.selected_axis_to_dense_axis(coords, {}, strict=True)


def test_selected_axis_to_dense_axis_none_meta_preserves_legacy_non_strict_behavior():
    post_utils = load_post_utils_with_test_stubs("strict_post_utils_none_meta")
    coords = object()

    assert post_utils.selected_axis_to_dense_axis(coords, None) is coords
    with pytest.raises(ValueError, match="strict.*metadata"):
        post_utils.selected_axis_to_dense_axis(coords, None, strict=True)


def test_selected_axis_to_dense_axis_strict_rejects_inconsistent_metadata():
    torch = import_torch_or_skip()
    post_utils = load_module("opentad/models/utils/post_processing/utils.py", "strict_post_utils_bad_meta")
    coords = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    meta = dict(
        irregular_selected_positions=[0.0, 3.0, 2.0],
        irregular_selected_valid_len=4.0,
        irregular_native_axis=False,
    )

    with pytest.raises(ValueError, match="monotonic"):
        post_utils.selected_axis_to_dense_axis(coords, meta, strict=True)


def test_convert_to_seconds_rejects_auto_for_irregular_metadata_unless_allowed():
    torch = import_torch_or_skip()
    post_utils = load_module("opentad/models/utils/post_processing/utils.py", "strict_post_utils_auto")
    segments = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    meta = dict(
        fps=1.0,
        snippet_stride=1.0,
        offset_frames=0.0,
        window_start_frame=0.0,
        duration=10.0,
        irregular_selected_positions=[0.0, 2.0],
        irregular_selected_valid_len=4.0,
        irregular_native_axis=False,
    )

    with pytest.raises(ValueError, match="source_axis='auto'"):
        post_utils.convert_to_seconds(segments.clone(), meta, source_axis="auto")

    allowed = post_utils.convert_to_seconds(
        segments.clone(),
        meta,
        source_axis="auto",
        allow_auto_axis=True,
    )
    explicit = post_utils.convert_to_seconds(segments.clone(), meta, source_axis="selected")
    assert torch.allclose(allowed, explicit)


def test_irregular_actionformer_requires_metas_for_axis_contracts():
    if sys.platform.startswith("win"):
        pytest.skip("detector import pulls torch on Windows")
    detector_mod = pytest.importorskip("opentad.models.detectors.irregular_actionformer")
    model = object.__new__(detector_mod.IrregularActionFormer)

    with pytest.raises(ValueError, match="requires metas"):
        model._assert_axis_contracts(None, stage="train")


def test_irregular_actionformer_selected_to_native_uses_strict_metadata():
    if sys.platform.startswith("win"):
        pytest.skip("detector import pulls torch on Windows")
    torch = import_torch_or_skip()
    detector_mod = pytest.importorskip("opentad.models.detectors.irregular_actionformer")
    model = object.__new__(detector_mod.IrregularActionFormer)
    segments = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    meta = dict(
        irregular_gt_axis="selected",
        irregular_proposal_axis="selected",
        irregular_postprocess_axis="native",
        irregular_selected_positions=[0.0, 3.0, 2.0],
        irregular_selected_valid_len=4.0,
        irregular_native_axis=False,
    )

    with pytest.raises(ValueError, match="monotonic"):
        model._segments_to_axis(segments, meta, source_axis="selected", target_axis="native")


def test_loadframes_selected_axis_gt_drop_fails_closed_by_default():
    import numpy as np

    end_to_end = load_end_to_end_with_test_stubs()
    loader = object.__new__(end_to_end.LoadFrames)
    loader.allow_drop_selected_axis_gt = False
    loader._current_video_name = "video_drop_contract"

    with pytest.raises(ValueError, match="video_drop_contract"):
        loader._remap_gt_to_selected_axis(
            gt_segments=np.asarray([[12.0, 13.0], [0.0, 5.0]], dtype=np.float32),
            gt_labels=np.asarray([7, 3], dtype=np.int32),
            kept_positions=np.asarray([0, 5], dtype=np.int64),
            valid_len=10,
        )


def test_loadframes_selected_axis_gt_drop_allows_explicit_legacy_opt_in():
    import numpy as np

    end_to_end = load_end_to_end_with_test_stubs()
    loader = object.__new__(end_to_end.LoadFrames)
    loader.allow_drop_selected_axis_gt = True
    loader._current_video_name = "video_legacy_drop"

    segments, labels = loader._remap_gt_to_selected_axis(
        gt_segments=np.asarray([[12.0, 13.0], [0.0, 5.0]], dtype=np.float32),
        gt_labels=np.asarray([7, 3], dtype=np.int32),
        kept_positions=np.asarray([0, 5], dtype=np.int64),
        valid_len=10,
    )

    assert segments.tolist() == [[0.0, 1.0]]
    assert labels.tolist() == [3]
    assert loader._last_selected_axis_gt_drop_count == 1


def test_fail_closed_config_scanner_rejects_v2_soft_fallback_dense_claim():
    scanner = load_module("tools/check_fail_closed_config.py", "strict_scanner_v2")

    violations = scanner.scan_config_object(
        dict(
            model=dict(
                rpn_head=dict(
                    type="IrregularActionFormerHeadV2",
                    soft_assign_topk=9,
                    allow_center_fallback_inside_gt=True,
                    route_contract=dict(dense_equivalent_claim_allowed=True),
                )
            )
        )
    )

    assert any("dense-equivalent claim" in item["reason"] for item in violations)


def test_fail_closed_config_scanner_treats_loadframes_selected_remap_default_as_true():
    scanner = load_module("tools/check_fail_closed_config.py", "strict_scanner_loadframes_default")

    default_remap_violations = scanner.scan_config_object(
        dict(dataset=dict(train=dict(pipeline=[dict(type="LoadFrames", allow_drop_selected_axis_gt=True)])))
    )
    native_axis_violations = scanner.scan_config_object(
        dict(
            dataset=dict(
                train=dict(
                    pipeline=[
                        dict(
                            type="LoadFrames",
                            remap_gt_to_selected_axis=False,
                            allow_drop_selected_axis_gt=True,
                        )
                    ]
                )
            )
        )
    )

    assert any("selected-axis GT drop" in item["reason"] for item in default_remap_violations)
    assert not native_axis_violations
