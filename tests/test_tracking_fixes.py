from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3_face_refine_tracking_fixes_test", ROOT / "tracking_fixes.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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


class _GeometryModule:
    _iou = staticmethod(_iou)


def _box(cx, cy=100, h=80, w=60):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_velocity_prediction_does_not_choose_stale_nearby_face():
    detections = [
        [_box(0)],
        [_box(60), _box(140)],
        [_box(120), _box(65)],
    ]
    track, stats = MODULE._associate_boxes(
        _GeometryModule,
        detections,
        width=400,
        height=300,
        select="largest",
    )
    assert track[2] == tuple(float(v) for v in _box(120))
    assert stats["rejected"] == 0


def test_unrelated_lone_detection_is_rejected_after_target_disappears():
    detections = [
        [_box(50)],
        [_box(55), _box(250)],
        [_box(250)],
    ]
    track, stats = MODULE._associate_boxes(
        _GeometryModule,
        detections,
        width=400,
        height=300,
        select="largest",
    )
    assert track[0] is not None
    assert track[1] is not None
    assert track[2] is None
    assert stats["rejected"] == 1


def test_identity_chooses_yolo_box_without_substituting_insightface_geometry():
    yolo_a = _box(60)
    yolo_b = _box(220)
    insight_a = (
        yolo_a[0] + 3,
        yolo_a[1] + 6,
        yolo_a[2] - 2,
        yolo_a[3] + 5,
    )
    insight_b = (
        yolo_b[0] - 4,
        yolo_b[1] + 7,
        yolo_b[2] + 2,
        yolo_b[3] + 4,
    )

    class _IdentityModule:
        _iou = staticmethod(_iou)
        _to_bgr_u8 = staticmethod(
            lambda _image: np.zeros((2, 2, 3), dtype=np.uint8)
        )
        _embed_faces = staticmethod(
            lambda _app, _bgr: [
                (insight_a, np.asarray([0.0, 1.0], dtype=np.float32)),
                (insight_b, np.asarray([1.0, 0.0], dtype=np.float32)),
            ]
        )

    track, stats = MODULE._associate_boxes(
        _IdentityModule,
        [[yolo_a, yolo_b]],
        width=400,
        height=300,
        select="largest",
        images=[object()],
        app=object(),
        ref_emb=np.asarray([1.0, 0.0], dtype=np.float32),
        identity_threshold=0.28,
    )
    assert track[0] == tuple(float(v) for v in yolo_b)
    assert track[0] != tuple(float(v) for v in insight_b)
    assert stats["identity"] == 1


def test_face_rect_maps_actual_offset_inside_top_clamped_crop():
    rect = MODULE._face_rect_in_canvas(
        face_cx=100,
        face_cy=40,
        face_w=60,
        face_h=80,
        crop_box=(0, 0, 200, 200),
        canvas_width=512,
        canvas_height=512,
    )
    left, top, width, height = rect
    assert math.isclose(left, 179.2, abs_tol=1e-6)
    assert math.isclose(top, 0.0, abs_tol=1e-6)
    assert math.isclose(width, 153.6, abs_tol=1e-6)
    assert math.isclose(height, 204.8, abs_tol=1e-6)
    # Old behaviour forced the face rectangle to canvas centre.
    assert not math.isclose(top, 512 * 0.5 - height * 0.5, abs_tol=1e-6)


def test_center_lag_guard_bounds_fast_motion_error_by_face_size():
    raw_x = np.asarray([0.0, 0.0, 100.0, 200.0])
    raw_y = np.zeros_like(raw_x)
    smooth_x = np.asarray([0.0, 20.0, 60.0, 130.0])
    smooth_y = np.zeros_like(raw_x)
    face_h = np.full_like(raw_x, 100.0)
    out_x, out_y, before, after = MODULE._bound_center_lag(
        raw_x, raw_y, smooth_x, smooth_y, face_h
    )
    assert before == 70.0
    assert after <= 12.0 + 1e-9
    assert np.allclose(out_y, 0.0)
    assert math.isclose(out_x[-1], 188.0, abs_tol=1e-9)


def test_center_lag_guard_does_not_follow_isolated_detector_spike():
    raw_x = np.asarray([0.0, 0.0, 100.0, 0.0, 0.0])
    raw_y = np.zeros_like(raw_x)
    smooth_x = np.zeros_like(raw_x)
    smooth_y = np.zeros_like(raw_x)
    face_h = np.full_like(raw_x, 100.0)
    out_x, _out_y, _before, after = MODULE._bound_center_lag(
        raw_x, raw_y, smooth_x, smooth_y, face_h
    )
    assert np.allclose(out_x, 0.0)
    assert after == 0.0


def test_body_fallback_tracks_expected_subject_instead_of_largest_person():
    target_body = (90.0, 40.0, 150.0, 240.0)
    large_bystander = (280.0, 20.0, 390.0, 300.0)
    chosen = MODULE._choose_body_for_track(
        [large_bystander, target_body],
        expected_cx=120.0,
        expected_cy=80.0,
        face_h=80.0,
        head_frac=0.5,
    )
    assert chosen == target_body


def test_identity_activates_when_multiple_faces_appear_after_frame_zero():
    detections = [[_box(50)], [_box(60)], [_box(70), _box(220)]]
    assert MODULE._should_use_identity(detections, None, True) is True
    assert MODULE._should_use_identity(detections, None, False) is False


def test_failed_identity_reacquisition_does_not_switch_to_lone_bystander():
    target0 = _box(50)
    target1 = _box(60)
    bystander1 = _box(170)
    bystander2 = _box(72)
    per_frame = {
        0: [(target0, np.asarray([1.0, 0.0], dtype=np.float32))],
        1: [
            (target1, np.asarray([1.0, 0.0], dtype=np.float32)),
            (bystander1, np.asarray([0.0, 1.0], dtype=np.float32)),
        ],
        2: [(bystander2, np.asarray([0.0, 1.0], dtype=np.float32))],
    }

    class _IdentityModule:
        _iou = staticmethod(_iou)
        _to_bgr_u8 = staticmethod(lambda frame: frame)
        _embed_faces = staticmethod(lambda _app, frame: per_frame[frame])

    track, stats = MODULE._associate_boxes(
        _IdentityModule,
        [[target0], [target1, bystander1], [bystander2]],
        width=400,
        height=300,
        select="largest",
        images=[0, 1, 2],
        app=object(),
        ref_emb=np.asarray([1.0, 0.0], dtype=np.float32),
        identity_threshold=0.28,
    )
    assert track[0] == tuple(float(v) for v in target0)
    assert track[1] == tuple(float(v) for v in target1)
    assert track[2] is None
    assert stats["rejected"] == 1
