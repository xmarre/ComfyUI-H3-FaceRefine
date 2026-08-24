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


TRACKING = _load("h3_face_refine_tracking_sparse_crowd_base", "tracking_fixes.py")
PERF = _load("h3_face_refine_tracking_sparse_crowd_perf", "tracking_performance.py")


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


def _tracking_with_latched_crowd_flag(frame_count: int):
    tracking = dict(TRACKING.__dict__)

    # Reproduces the production failure mode where overlapping expanded crowd-transition
    # windows keep singleton frames marked crowded for a long target-loss stretch.
    tracking["_crowded_frames"] = lambda _detections, radius=2: [True] * frame_count
    return tracking


def _run(detections, embeddings):
    module = _IdentityModule(embeddings)
    track, stats = PERF._associate_boxes_sparse(
        _tracking_with_latched_crowd_flag(len(detections)),
        module,
        detections,
        width=400,
        height=300,
        select="largest",
        images=list(range(len(detections))),
        app=object(),
        ref_emb=np.asarray([1.0, 0.0], dtype=np.float32),
        identity_threshold=0.28,
        embedding_cache={},
    )
    return module, track, stats


def test_latched_crowd_flag_does_not_disable_negative_tracklet_sparse_probing():
    target = _box(50)
    bystander = _box(300)
    frames = 40
    detections = [[target]] + [[bystander] for _ in range(frames - 1)]
    embeddings = {
        0: [(target, np.asarray([1.0, 0.0], dtype=np.float32))]
    }
    embeddings.update(
        {
            frame: [(bystander, np.asarray([0.0, 1.0], dtype=np.float32))]
            for frame in range(1, frames)
        }
    )

    module, track, stats = _run(detections, embeddings)

    assert track[0] == tuple(float(v) for v in target)
    assert all(box is None for box in track[1:])
    assert stats["identity_skipped"] >= 30
    assert stats["reacquire_probes"] <= 4
    assert len(module.calls) <= 5


def test_latched_crowd_sparse_probe_backfills_exact_target_return_boundary():
    target = _box(50)
    shared_geometry = _box(300)
    frames = 30
    return_frame = 20
    detections = [[target]] + [[shared_geometry] for _ in range(frames - 1)]
    embeddings = {
        0: [(target, np.asarray([1.0, 0.0], dtype=np.float32))]
    }
    for frame in range(1, frames):
        identity = (
            np.asarray([1.0, 0.0], dtype=np.float32)
            if frame >= return_frame
            else np.asarray([0.0, 1.0], dtype=np.float32)
        )
        embeddings[frame] = [(shared_geometry, identity)]

    module, track, stats = _run(detections, embeddings)

    assert all(box is None for box in track[1:return_frame])
    assert all(
        track[frame] == tuple(float(v) for v in shared_geometry)
        for frame in range(return_frame, frames)
    )
    assert stats["backfilled"] >= 5
    assert stats["identity_skipped"] > 0
    assert len(module.calls) < frames
