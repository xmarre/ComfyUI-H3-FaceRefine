from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACKING = _load("h3_face_refine_tracking_runtime_integration_base", "tracking_fixes.py")
PERF = _load("h3_face_refine_tracking_runtime_integration_perf", "tracking_performance.py")
RUNTIME = _load("h3_face_refine_tracking_runtime_integration_policy", "tracking_runtime.py")


def _box(cx, cy=100, h=80, w=60):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


class _IdentityModule:
    _iou = staticmethod(_iou)

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.calls = []

    @staticmethod
    def _to_bgr_u8(frame):
        return frame

    def _embed_faces(self, _app, frame):
        self.calls.append(frame)
        return self.embeddings[frame]


def _runtime_tracking_dict():
    tracking = dict(TRACKING.__dict__)
    original = tracking["_crowded_frames"]

    def checkpointed(detections, radius=2):
        return RUNTIME._checkpointed_crowded_frames(
            detections, original, radius=radius, stride=12
        )

    tracking["_crowded_frames"] = checkpointed
    return tracking


def test_clear_long_crowd_does_not_run_identity_every_frame():
    frames = 30
    target_boxes = [_box(80 + frame * 2) for frame in range(frames)]
    bystander_boxes = [_box(310) for _ in range(frames)]
    detections = [
        [target_boxes[frame], bystander_boxes[frame]] for frame in range(frames)
    ]
    embeddings = {
        frame: [
            (target_boxes[frame], np.asarray([1.0, 0.0], dtype=np.float32)),
            (bystander_boxes[frame], np.asarray([0.0, 1.0], dtype=np.float32)),
        ]
        for frame in range(frames)
    }
    module = _IdentityModule(embeddings)

    track, stats = PERF._associate_boxes_sparse(
        _runtime_tracking_dict(),
        module,
        detections,
        width=400,
        height=300,
        select="largest",
        images=list(range(frames)),
        app=object(),
        ref_emb=np.asarray([1.0, 0.0], dtype=np.float32),
        identity_threshold=0.28,
        embedding_cache={},
    )

    assert all(track[i] == tuple(float(v) for v in target_boxes[i]) for i in range(frames))
    assert len(module.calls) <= 4
    assert stats["rejected"] == 0


def test_ambiguous_crossing_still_forces_identity_between_checkpoints():
    target0 = _box(80)
    other0 = _box(300)
    target1 = _box(100)
    # Put the second candidate close enough to the predicted target that the normal
    # ambiguity margin fires even though frame 1 is not a scheduled crowd checkpoint.
    other1 = _box(135)
    detections = [[target0, other0], [target1, other1]]
    embeddings = {
        0: [
            (target0, np.asarray([1.0, 0.0], dtype=np.float32)),
            (other0, np.asarray([0.0, 1.0], dtype=np.float32)),
        ],
        1: [
            (target1, np.asarray([1.0, 0.0], dtype=np.float32)),
            (other1, np.asarray([0.0, 1.0], dtype=np.float32)),
        ],
    }
    module = _IdentityModule(embeddings)

    track, stats = PERF._associate_boxes_sparse(
        _runtime_tracking_dict(),
        module,
        detections,
        width=400,
        height=300,
        select="largest",
        images=[0, 1],
        app=object(),
        ref_emb=np.asarray([1.0, 0.0], dtype=np.float32),
        identity_threshold=0.28,
        embedding_cache={},
    )

    assert module.calls == [0, 1]
    assert stats["ambiguous"] >= 1
    assert track[1] == tuple(float(v) for v in target1)
