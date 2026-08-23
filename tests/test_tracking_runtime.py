from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACKING = _load("h3_face_refine_tracking_runtime_base_test", "tracking_fixes.py")
RUNTIME = _load("h3_face_refine_tracking_runtime_test", "tracking_runtime.py")


def _box(cx):
    return (cx - 30.0, 60.0, cx + 30.0, 140.0)


def test_long_stable_multiface_run_uses_periodic_crowd_checkpoints():
    detections = [[_box(80), _box(300)] for _ in range(30)]

    crowded = RUNTIME._checkpointed_crowded_frames(
        detections,
        TRACKING._crowded_frames,
        radius=2,
        stride=12,
    )

    assert [i for i, value in enumerate(crowded) if value] == [0, 12, 24]


def test_new_multiface_run_gets_immediate_checkpoint_after_single_frame_gap():
    detections = [
        [_box(80), _box(300)],
        [_box(80), _box(300)],
        [_box(80)],
        [_box(80), _box(300)],
        [_box(80), _box(300)],
    ]

    crowded = RUNTIME._checkpointed_crowded_frames(
        detections,
        TRACKING._crowded_frames,
        radius=2,
        stride=12,
    )

    assert crowded[0] is True
    assert crowded[2] is True  # singleton transition safety is preserved
    assert crowded[3] is True  # a new crowd run is checked immediately


def test_expanded_singleton_transition_guard_is_preserved():
    detections = [
        [_box(80)],
        [_box(80)],
        [_box(80), _box(300)],
        [_box(80)],
        [_box(80)],
    ]
    baseline = TRACKING._crowded_frames(detections, radius=2)

    crowded = RUNTIME._checkpointed_crowded_frames(
        detections,
        TRACKING._crowded_frames,
        radius=2,
        stride=12,
    )

    for i, boxes in enumerate(detections):
        if len(boxes) == 1:
            assert crowded[i] is baseline[i]


def test_identity_provider_state_does_not_require_onnxruntime(monkeypatch):
    # The module is intentionally optional in CPU-only CI.  A failed import/query must
    # remain diagnostic rather than making FaceRefine itself unloadable.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError("onnxruntime unavailable in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    backend, providers, error = RUNTIME._identity_provider_state()

    assert backend == "unavailable"
    assert providers == ()
    assert "onnxruntime unavailable" in error
