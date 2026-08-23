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


TRACKING = _load("h3_face_refine_tracking_fixes_perf_test", "tracking_fixes.py")
PERF = _load("h3_face_refine_tracking_performance_test", "tracking_performance.py")


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


def _box(cx, cy=100, h=80, w=60):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


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


def _run(detections, embeddings):
    module = _IdentityModule(embeddings)
    track, stats = PERF._associate_boxes_sparse(
        TRACKING.__dict__,
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


def test_long_rejected_bystander_is_not_identity_analyzed_every_frame():
    target = _box(50)
    bystander = _box(300)
    frames = 50
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
    assert stats["rejected"] == frames - 1
    assert stats["identity_skipped"] >= 40
    # Old behaviour ran InsightFace on all 50 frames. Sparse refreshes should need only
    # frame 0 plus the first reject and periodic probes at the fixed refresh cadence.
    assert len(module.calls) <= 6
    assert stats["identity_clip_frames"] == len(module.calls)


def test_sparse_probe_backfills_exact_contiguous_reappearance_boundary():
    target = _box(50)
    same_geometry = _box(300)
    frames = 14
    detections = [[target]] + [[same_geometry] for _ in range(frames - 1)]
    embeddings = {
        0: [(target, np.asarray([1.0, 0.0], dtype=np.float32))]
    }
    # The same YOLO tracklet is a wrong person through frame 9, then the target occupies
    # that geometry from frame 10 onward. The sparse refresh occurs later, so backfill must
    # recover frames 10-12 rather than introducing output tracking delay.
    for frame in range(1, frames):
        identity = (
            np.asarray([1.0, 0.0], dtype=np.float32)
            if frame >= 10
            else np.asarray([0.0, 1.0], dtype=np.float32)
        )
        embeddings[frame] = [(same_geometry, identity)]

    module, track, stats = _run(detections, embeddings)

    assert all(box is None for box in track[1:10])
    assert all(track[frame] == tuple(float(v) for v in same_geometry) for frame in range(10, 14))
    assert stats["backfilled"] == 3
    assert stats["rejected"] == 9
    # The successful sparse probe plus a bounded reverse scan is still far cheaper than
    # analyzing every frame in the lost stretch.
    assert len(module.calls) == 7


def test_new_second_candidate_forces_immediate_identity_reacquisition():
    target0 = _box(50)
    bystander = _box(300)
    target_return = _box(180)
    detections = [
        [target0],
        [bystander],
        [bystander],
        [bystander],
        [bystander, target_return],
    ]
    embeddings = {
        0: [(target0, np.asarray([1.0, 0.0], dtype=np.float32))],
        1: [(bystander, np.asarray([0.0, 1.0], dtype=np.float32))],
        2: [(bystander, np.asarray([0.0, 1.0], dtype=np.float32))],
        3: [(bystander, np.asarray([0.0, 1.0], dtype=np.float32))],
        4: [
            (bystander, np.asarray([0.0, 1.0], dtype=np.float32)),
            (target_return, np.asarray([1.0, 0.0], dtype=np.float32)),
        ],
    }

    module, track, stats = _run(detections, embeddings)

    assert track[4] == tuple(float(v) for v in target_return)
    assert 4 in module.calls
    assert stats["identity"] >= 2


def test_negative_tracklet_geometry_change_forces_probe_before_refresh_stride():
    target = _box(50)
    bystander = _box(300)
    changed = _box(180)
    detections = [[target], [bystander], [bystander], [changed]]
    embeddings = {
        0: [(target, np.asarray([1.0, 0.0], dtype=np.float32))],
        1: [(bystander, np.asarray([0.0, 1.0], dtype=np.float32))],
        2: [(bystander, np.asarray([0.0, 1.0], dtype=np.float32))],
        3: [(changed, np.asarray([1.0, 0.0], dtype=np.float32))],
    }

    module, track, stats = _run(detections, embeddings)

    # Forward tracking skips frame 2, sees the material geometry change at frame 3 and
    # probes immediately. The successful frame-3 probe then performs the bounded offline
    # backfill, which is why frame 2 appears later in the call log.
    assert module.calls[:3] == [0, 1, 3]
    assert module.calls.index(3) < module.calls.index(2)
    assert stats["identity_skipped"] >= 1
    assert track[2] is None
    assert track[3] == tuple(float(v) for v in changed)
