"""Tracking and face-region geometry correctness fixes for H3 FaceRefine.

The tracker needs two related trajectories with different jobs:

* a stabilized crop trajectory, so H3 sees steady context;
* responsive face geometry, so the refine/SAM mask stays on the face inside that crop.

The original implementation collapsed those into one smoothed centre and also mixed
YOLO and InsightFace boxes. This module keeps the configured YOLO detector as the
sole geometry source, uses InsightFace only for identity selection, predicts motion
for association, bounds crop lag, and maps responsive face geometry through the
actual (possibly frame-edge-clamped) crop transform.
"""

from __future__ import annotations

from functools import wraps
import math
import sys
from typing import Mapping

import numpy as np
import torch


_INSTALL_MARKER = "_h3_tracking_fixes_installed"
_MAX_ANCHOR_SAMPLES = 24
_ANCHOR_INLIER_SIM = 0.30
_CENTER_LAG_FRACTION = 0.12
_CENTER_LAG_MIN_PX = 1.0


def _box_state(box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (
        (x0 + x1) * 0.5,
        (y0 + y1) * 0.5,
        max(y1 - y0, 1.0),
        max(x1 - x0, 1.0),
    )


def _select_initial_box(boxes, width: int, height: int, select: str):
    if select == "most_central":
        fc = (float(width) * 0.5, float(height) * 0.5)
        return min(
            boxes,
            key=lambda q: (
                (_box_state(q)[0] - fc[0]) ** 2 + (_box_state(q)[1] - fc[1]) ** 2
            ),
        )
    return max(boxes, key=lambda q: _box_state(q)[2])


def _predict_track_state(
    history: list[tuple[int, tuple[float, float, float, float]]], frame: int
):
    """Predict centre from recent accepted YOLO boxes; keep last measured size."""
    last_i, last_box = history[-1]
    last_cx, last_cy, last_h, last_w = _box_state(last_box)
    gap = max(1, int(frame) - int(last_i))
    if len(history) < 2:
        return last_cx, last_cy, last_h, last_w, gap

    vx: list[float] = []
    vy: list[float] = []
    recent = history[-4:]
    for (ia, ba), (ib, bb) in zip(recent[:-1], recent[1:]):
        dt = max(1, int(ib) - int(ia))
        ax, ay, _, _ = _box_state(ba)
        bx, by, _, _ = _box_state(bb)
        vx.append((bx - ax) / dt)
        vy.append((by - ay) / dt)
    dx = float(np.median(vx)) * gap if vx else 0.0
    dy = float(np.median(vy)) * gap if vy else 0.0

    # Never let one noisy velocity sample project arbitrarily far through a dropout.
    max_disp = last_h * (1.5 + 0.5 * min(gap, 4))
    disp = math.hypot(dx, dy)
    if disp > max_disp > 0.0:
        scale = max_disp / disp
        dx *= scale
        dy *= scale
    return last_cx + dx, last_cy + dy, last_h, last_w, gap


def _tracking_cost(box, predicted) -> float:
    pcx, pcy, ph, _pw, _gap = predicted
    cx, cy, h, _w = _box_state(box)
    scale = max(0.5 * (ph + h), 8.0)
    distance = math.hypot(cx - pcx, cy - pcy) / scale
    size = abs(math.log(max(h, 1.0) / max(ph, 1.0)))
    return distance + 0.60 * size


def _inside_motion_gate(box, predicted) -> bool:
    pcx, pcy, ph, _pw, gap = predicted
    cx, cy, h, _w = _box_state(box)
    scale = max(0.5 * (ph + h), 8.0)
    distance = math.hypot(cx - pcx, cy - pcy) / scale
    size_ratio = max(h, ph) / max(min(h, ph), 1.0)
    extra = min(max(gap - 1, 0), 4)
    max_distance = 1.25 + 0.60 * extra
    max_size_ratio = 2.0 + 0.35 * extra
    return distance <= max_distance and size_ratio <= max_size_ratio


def _should_use_identity(detections, identity_reference, identity_track: bool) -> bool:
    return bool(identity_track) and (
        identity_reference is not None or any(len(boxes) > 1 for boxes in detections)
    )


def _crowded_frames(detections, radius: int = 2) -> list[bool]:
    crowded = [len(boxes) > 1 for boxes in detections]
    if radius <= 0:
        return crowded
    out = crowded[:]
    for i, value in enumerate(crowded):
        if not value:
            continue
        lo, hi = max(0, i - radius), min(len(out), i + radius + 1)
        for j in range(lo, hi):
            out[j] = True
    return out


def _match_embedding_to_box(module, candidates, box):
    """Map InsightFace identity data onto a YOLO box without adopting its geometry."""
    if not candidates:
        return None
    overlaps = [float(module._iou(cand_box, box)) for cand_box, _emb in candidates]
    best = int(np.argmax(overlaps))
    if overlaps[best] >= 0.10:
        return candidates[best][1]

    cx, cy, h, _w = _box_state(box)
    distances = []
    for cand_box, _emb in candidates:
        ccx, ccy, ch, _cw = _box_state(cand_box)
        scale = max(0.5 * (h + ch), 8.0)
        distances.append(math.hypot(ccx - cx, ccy - cy) / scale)
    best = int(np.argmin(distances))
    return candidates[best][1] if distances[best] <= 0.75 else None


def _frame_embeddings(module, app, images, frame: int, cache: dict[int, list]):
    if frame not in cache:
        cache[frame] = module._embed_faces(app, module._to_bgr_u8(images[frame]))
    return cache[frame]


def _identity_scores_for_boxes(module, app, images, frame: int, boxes, ref_emb, cache):
    candidates = _frame_embeddings(module, app, images, frame, cache)
    scores: list[float] = []
    for box in boxes:
        emb = _match_embedding_to_box(module, candidates, box)
        scores.append(float(np.dot(emb, ref_emb)) if emb is not None else -1.0)
    return scores


def _build_anchor_from_track(
    module,
    app,
    images,
    track,
    cache,
    max_samples: int = _MAX_ANCHOR_SAMPLES,
):
    indices = [i for i, box in enumerate(track) if box is not None]
    if not indices:
        return None, 0
    if len(indices) > max_samples:
        positions = np.linspace(0, len(indices) - 1, max_samples)
        indices = [indices[int(round(p))] for p in positions]
        indices = list(dict.fromkeys(indices))

    embeddings = []
    for i in indices:
        candidates = _frame_embeddings(module, app, images, i, cache)
        emb = _match_embedding_to_box(module, candidates, track[i])
        if emb is not None:
            embeddings.append(np.asarray(emb, dtype=np.float32))
    if not embeddings:
        return None, 0
    if len(embeddings) == 1:
        return embeddings[0], 1

    # Anchor to the initially selected identity. A later accidental crossing must not
    # redefine the target merely because more samples came from the wrong person.
    stack = np.stack(embeddings)
    seed = stack[0]
    keep = (stack @ seed) >= _ANCHOR_INLIER_SIM
    if not bool(np.any(keep)):
        keep[0] = True
    anchor = np.mean(stack[keep], axis=0)
    norm = float(np.linalg.norm(anchor))
    if norm > 0.0:
        anchor = anchor / norm
    return anchor.astype(np.float32), int(np.sum(keep))


def _associate_boxes(
    module,
    detections,
    *,
    width: int,
    height: int,
    select: str,
    images=None,
    app=None,
    ref_emb=None,
    identity_threshold: float = 0.28,
    embedding_cache: dict[int, list] | None = None,
):
    """Associate one configured-detector box per frame without changing geometry source."""
    cache = embedding_cache if embedding_cache is not None else {}
    track: list[tuple[float, float, float, float] | None] = [None] * len(detections)
    history: list[tuple[int, tuple[float, float, float, float]]] = []
    crowded = _crowded_frames(detections)
    stats = {"continuity": 0, "identity": 0, "ambiguous": 0, "rejected": 0}

    for i, raw_boxes in enumerate(detections):
        boxes = [tuple(float(v) for v in box) for box in raw_boxes]
        if not boxes:
            continue

        chosen = None
        if not history:
            if ref_emb is not None and app is not None and images is not None:
                scores = _identity_scores_for_boxes(
                    module, app, images, i, boxes, ref_emb, cache
                )
                best_i = int(np.argmax(scores)) if scores else -1
                if best_i >= 0 and scores[best_i] >= float(identity_threshold):
                    chosen = boxes[best_i]
                    stats["identity"] += 1
            if chosen is None:
                chosen = tuple(_select_initial_box(boxes, width, height, select))
                stats["continuity"] += 1
        else:
            predicted = _predict_track_state(history, i)
            ranked = sorted(boxes, key=lambda q: _tracking_cost(q, predicted))
            best = ranked[0]
            best_cost = _tracking_cost(best, predicted)
            gate_ok = _inside_motion_gate(best, predicted)

            ambiguous = False
            if len(ranked) > 1:
                second = ranked[1]
                second_cost = _tracking_cost(second, predicted)
                ambiguous = (
                    (second_cost - best_cost) < 0.75
                    or float(module._iou(best, second)) > 0.15
                )
                if ambiguous:
                    stats["ambiguous"] += 1

            safety_identity = (
                (not gate_ok)
                or predicted[-1] > 1
                or (crowded[i] and len(boxes) == 1)
            )
            need_identity = (
                ref_emb is not None
                and app is not None
                and images is not None
                and (ambiguous or safety_identity or crowded[i])
            )
            identity_passed = False
            if need_identity:
                scores = _identity_scores_for_boxes(
                    module, app, images, i, boxes, ref_emb, cache
                )
                best_i = int(np.argmax(scores)) if scores else -1
                if best_i >= 0 and scores[best_i] >= float(identity_threshold):
                    chosen = boxes[best_i]
                    identity_passed = True
                    stats["identity"] += 1

            # Identity failure may fall back to continuity only while continuity itself
            # is trustworthy. Across gaps/crowd transitions, a miss is safer than silently
            # replacing the subject with a nearby bystander.
            if (
                chosen is None
                and gate_ok
                and not (need_identity and safety_identity and not identity_passed)
            ):
                chosen = best
                stats["continuity"] += 1
            elif chosen is None:
                stats["rejected"] += 1
                continue

        chosen = tuple(float(v) for v in chosen)
        track[i] = chosen
        history.append((i, chosen))

    return track, stats


def _choose_body_for_track(
    people,
    expected_cx: float,
    expected_cy: float,
    face_h: float,
    head_frac: float,
):
    scale = max(float(face_h), 8.0)

    def cost(box):
        x0, y0, x1, _y1 = (float(v) for v in box)
        hx = 0.5 * (x0 + x1)
        hy = y0 + float(head_frac) * scale
        return math.hypot(hx - expected_cx, hy - expected_cy) / scale

    return min(people, key=cost)


def _median3(values):
    """One-frame impulse rejection that preserves sustained movement and step changes."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.median(
        np.stack((padded[:-2], padded[1:-1], padded[2:])),
        axis=0,
    )


def _bound_center_lag(raw_cx, raw_cy, smooth_cx, smooth_cy, face_h):
    """Stabilize the crop while retaining a separate responsive face trajectory.

    The responsive median-3 trajectory is the geometry truth used for face masks. The
    smoothed trajectory is only the crop centre and may deviate by at most 12% of face
    height, leaving enough stability to suppress box jitter without visibly trailing a
    real head movement.
    """
    raw_cx = np.asarray(raw_cx, dtype=np.float64)
    raw_cy = np.asarray(raw_cy, dtype=np.float64)
    smooth_cx = np.asarray(smooth_cx, dtype=np.float64).copy()
    smooth_cy = np.asarray(smooth_cy, dtype=np.float64).copy()
    face_h = np.asarray(face_h, dtype=np.float64)

    responsive_cx = _median3(raw_cx)
    responsive_cy = _median3(raw_cy)
    dx = responsive_cx - smooth_cx
    dy = responsive_cy - smooth_cy
    distance = np.hypot(dx, dy)
    limit = np.maximum(
        _CENTER_LAG_MIN_PX,
        np.maximum(face_h, 1.0) * _CENTER_LAG_FRACTION,
    )
    correction = np.zeros_like(distance)
    active = distance > limit
    correction[active] = (
        (distance[active] - limit[active]) / np.maximum(distance[active], 1e-12)
    )
    smooth_cx += dx * correction
    smooth_cy += dy * correction
    guarded = np.hypot(responsive_cx - smooth_cx, responsive_cy - smooth_cy)
    return (
        smooth_cx,
        smooth_cy,
        responsive_cx,
        responsive_cy,
        float(distance.max(initial=0.0)),
        float(guarded.max(initial=0.0)),
    )


def _face_rect_in_canvas(
    face_cx: float,
    face_cy: float,
    face_w: float,
    face_h: float,
    crop_box,
    canvas_width: int,
    canvas_height: int,
):
    """Map source-space face geometry through the exact source->crop->canvas transform."""
    x, y, bw, bh = (float(v) for v in crop_box)
    if bw <= 0.0 or bh <= 0.0:
        raise ValueError(f"invalid crop box for face mapping: {crop_box!r}")
    left = (
        (float(face_cx) - 0.5 * float(face_w) - x)
        / bw
        * float(canvas_width)
    )
    top = (
        (float(face_cy) - 0.5 * float(face_h) - y)
        / bh
        * float(canvas_height)
    )
    width = float(face_w) / bw * float(canvas_width)
    height = float(face_h) / bh * float(canvas_height)
    return left, top, width, height


def _run_fixed(
    module,
    images,
    detector,
    confidence,
    crop_factor,
    canvas_width,
    canvas_height,
    canvas_mode,
    smooth_window,
    size_smooth_window,
    smooth_method,
    size_mode,
    select="largest",
    fallback_detector="none",
    fallback_head_frac=0.5,
    identity_reference=None,
    identity_threshold=0.28,
    identity_track=True,
):
    model = module._load_detector(detector)
    B, H, W, _ = images.shape
    import comfy.model_management as _mm

    # Detect once. Identity logic consumes these boxes instead of running a second YOLO
    # pass, and every accepted frame retains the configured detector's exact geometry.
    detections = []
    for i in range(B):
        _mm.throw_exception_if_processing_interrupted()
        result = model.predict(
            module._to_bgr_u8(images[i]), conf=confidence, verbose=False
        )[0]
        detections.append(result.boxes.xyxy.tolist() if len(result.boxes) else [])

    geometric_track, geometric_stats = _associate_boxes(
        module,
        detections,
        width=W,
        height=H,
        select=select,
    )

    embedding_cache: dict[int, list] = {}
    ref_emb, app = None, None
    anchor_count = 0
    if _should_use_identity(detections, identity_reference, identity_track):
        try:
            app = module._face_recogniser()
            if identity_reference is not None:
                candidates = module._embed_faces(
                    app, module._to_bgr_u8(identity_reference[0])
                )
                if candidates:
                    j = max(
                        range(len(candidates)),
                        key=lambda k: _box_state(candidates[k][0])[2],
                    )
                    ref_emb = np.asarray(candidates[j][1], dtype=np.float32)
                    anchor_count = 1
                    print("[H3FaceRefine] identity anchor from the supplied reference")
            if ref_emb is None:
                ref_emb, anchor_count = _build_anchor_from_track(
                    module,
                    app,
                    images,
                    geometric_track,
                    embedding_cache,
                )
                if ref_emb is not None:
                    print(
                        f"[H3FaceRefine] identity anchor built from the tracked clip "
                        f"({anchor_count} consistent frames)"
                    )
        except Exception as exc:
            print(f"[H3FaceRefine] identity matching unavailable ({exc})")
            ref_emb, app = None, None

    if ref_emb is not None and app is not None:
        track, track_stats = _associate_boxes(
            module,
            detections,
            width=W,
            height=H,
            select=select,
            images=images,
            app=app,
            ref_emb=ref_emb,
            identity_threshold=identity_threshold,
            embedding_cache=embedding_cache,
        )
    else:
        track, track_stats = geometric_track, geometric_stats

    detected_cx = np.zeros(B, dtype=np.float64)
    detected_cy = np.zeros(B, dtype=np.float64)
    face_h_raw = np.zeros(B, dtype=np.float64)
    face_w_raw = np.zeros(B, dtype=np.float64)
    valid = np.zeros(B, dtype=bool)
    via_body = np.zeros(B, dtype=bool)
    for i, box in enumerate(track):
        if box is None:
            continue
        bx, by, bh, bw = _box_state(box)
        detected_cx[i], detected_cy[i] = bx, by
        face_h_raw[i], face_w_raw[i] = bh, bw
        valid[i] = True

    found = int(valid.sum())
    if found == 0:
        # Keep the exact prefix consumed by the existing no-face passthrough policy.
        raise ValueError(
            "No face detected in any frame. Lower `confidence`, or this clip has no "
            "usable face and should be skipped."
        )

    # Body fallback follows the expected tracked head rather than the largest person.
    face_h_seed = module._interp_gaps(face_h_raw, valid)
    expected_cx = module._interp_gaps(detected_cx, valid)
    expected_cy = module._interp_gaps(detected_cy, valid)
    fallback_cx = detected_cx.copy()
    fallback_cy = detected_cy.copy()
    if fallback_detector != "none" and (~valid).any():
        try:
            body_model = module._load_detector(fallback_detector)
            for i in np.nonzero(~valid)[0]:
                _mm.throw_exception_if_processing_interrupted()
                result = body_model.predict(
                    module._to_bgr_u8(images[i]), conf=confidence, verbose=False
                )[0]
                if not len(result.boxes):
                    continue
                body_boxes = result.boxes.xyxy.tolist()
                classes = (
                    result.boxes.cls.tolist()
                    if getattr(result.boxes, "cls", None) is not None
                    else [0] * len(body_boxes)
                )
                people = [
                    q for q, cls_id in zip(body_boxes, classes) if int(cls_id) == 0
                ] or body_boxes
                person = _choose_body_for_track(
                    people,
                    expected_cx=float(expected_cx[i]),
                    expected_cy=float(expected_cy[i]),
                    face_h=float(face_h_seed[i]),
                    head_frac=float(fallback_head_frac),
                )
                fallback_cx[i] = 0.5 * (float(person[0]) + float(person[2]))
                fallback_cy[i] = float(person[1]) + float(fallback_head_frac) * max(
                    float(face_h_seed[i]), 8.0
                )
                via_body[i] = True
        except Exception as exc:
            print(f"[H3FaceRefine] body fallback '{fallback_detector}' failed: {exc}")

    known = valid | via_body
    raw_cx = module._interp_gaps(fallback_cx, known)
    raw_cy = module._interp_gaps(fallback_cy, known)
    raw_face_h = module._interp_gaps(face_h_raw, valid)
    raw_face_w = module._interp_gaps(face_w_raw, valid)

    # Crop centre: heavily smoothed for stable H3 context, then bounded against real motion.
    smooth_cx = module._smooth(raw_cx, smooth_window, smooth_method)
    smooth_cy = module._smooth(raw_cy, smooth_window, smooth_method)
    (
        crop_cx,
        crop_cy,
        face_cx,
        face_cy,
        max_lag_before,
        max_lag_after,
    ) = _bound_center_lag(raw_cx, raw_cy, smooth_cx, smooth_cy, raw_face_h)

    # Face-region geometry: responsive median-3 trajectory. This is deliberately separate
    # from crop smoothing: a stable crop may trail slightly, while the mask must remain on
    # the actual face *inside* that crop. Median-3 removes isolated detector impulses.
    mask_face_h = np.maximum(_median3(raw_face_h), 1.0)
    mask_face_w = np.maximum(_median3(raw_face_w), 1.0)

    # Crop size may stay more heavily smoothed because breathing changes resample scale.
    crop_face_h = module._smooth(raw_face_h, size_smooth_window, smooth_method)
    if size_mode == "max_of_clip":
        crop_face_h[:] = crop_face_h.max()

    def _jit(values):
        return float(np.abs(np.diff(values)).mean()) if len(values) > 1 else 0.0

    jit_before = (_jit(raw_cx) + _jit(raw_cy)) * 0.5
    jit_after = (_jit(crop_cx) + _jit(crop_cy)) * 0.5
    size_before, size_after = _jit(raw_face_h), _jit(crop_face_h)

    if canvas_mode != "manual":
        need = float(min(crop_face_h.max() * crop_factor, H))
        snapped = int(np.ceil(need / 32.0) * 32)
        if canvas_mode == "auto_capped_768":
            snapped = min(snapped, 768)
        snapped = max(128, min(snapped, 1344))
        if snapped != canvas_height:
            print(
                f"[H3FaceRefine] canvas_mode={canvas_mode}: "
                f"{canvas_width}x{canvas_height} -> {snapped}x{snapped} "
                f"(largest crop {need:.0f}px)"
            )
        canvas_width = canvas_height = snapped

    aspect = canvas_width / float(canvas_height)
    boxes: list[tuple[float, float, float, float]] = []
    crops = torch.zeros(
        (B, canvas_height, canvas_width, 3), dtype=images.dtype
    )
    preview = images[..., :3].clone()

    for i in range(B):
        bh = float(crop_face_h[i]) * float(crop_factor)
        bw = bh * aspect
        if bw > W:
            bw, bh = float(W), float(W) / aspect
        if bh > H:
            bh, bw = float(H), float(H) * aspect
        x = min(max(float(crop_cx[i]) - bw * 0.5, 0.0), max(0.0, W - bw))
        y = min(max(float(crop_cy[i]) - bh * 0.5, 0.0), max(0.0, H - bh))
        box = (float(x), float(y), float(bw), float(bh))
        boxes.append(box)
        crops[i : i + 1] = module._affine_crop(
            images[i : i + 1], box, canvas_width, canvas_height
        ).to(crops.dtype)

        xi, yi = int(round(x)), int(round(y))
        wi, hi = max(4, int(round(bw))), max(4, int(round(bh)))
        xi = min(xi, W - wi)
        yi = min(yi, H - hi)
        if valid[i]:
            red, green = 0.0, 1.0
        elif via_body[i]:
            red, green = 1.0, 1.0
        else:
            red, green = 1.0, 0.0
        for yy0, yy1, xx0, xx1 in (
            (yi, yi + 2, xi, xi + wi),
            (yi + hi - 2, yi + hi, xi, xi + wi),
            (yi, yi + hi, xi, xi + 2),
            (yi, yi + hi, xi + wi - 2, xi + wi),
        ):
            preview[i, yy0:yy1, xx0:xx1, 0] = red
            preview[i, yy0:yy1, xx0:xx1, 1] = green
            preview[i, yy0:yy1, xx0:xx1, 2] = 0.0

    weights = module._smooth(
        valid.astype(np.float64), max(9, smooth_window // 2), "gaussian"
    )
    weights = np.clip(weights, 0.0, 1.0)

    runs, current = [], 0
    for value in known:
        if value:
            if current:
                runs.append(current)
            current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    longest_gap = max(runs) if runs else 0

    # This is the key placement invariant: face geometry is mapped from its responsive
    # source-space position through the *actual* crop box. It is not assumed to be at the
    # crop centre, so both smoothing displacement and frame-edge clamping are represented.
    face_rects = [
        _face_rect_in_canvas(
            float(face_cx[i]),
            float(face_cy[i]),
            float(mask_face_w[i]),
            float(mask_face_h[i]),
            boxes[i],
            int(canvas_width),
            int(canvas_height),
        )
        for i in range(B)
    ]
    source_face_rects = [
        (
            float(face_cx[i] - 0.5 * mask_face_w[i]),
            float(face_cy[i] - 0.5 * mask_face_h[i]),
            float(mask_face_w[i]),
            float(mask_face_h[i]),
        )
        for i in range(B)
    ]
    mags = [canvas_height / float(box[3]) for box in boxes]
    transform = {
        "boxes": boxes,
        "canvas": (int(canvas_width), int(canvas_height)),
        "src_size": (int(W), int(H)),
        "frames": int(B),
        "weights": [float(weight) for weight in weights],
        "detected": [bool(value) for value in valid],
        "face_rect": face_rects,
        "face_rect_source": source_face_rects,
        "crop_factor": float(crop_factor),
        "tracking_geometry": "configured_detector_only",
        "face_geometry": "responsive_median3",
    }

    gapwarn = ""
    if longest_gap >= 12:
        gapwarn = (
            f"\n!! longest dropout is {longest_gap} frames "
            f"({longest_gap / 24.0:.1f}s). The crop is interpolated through the gap; "
            "detector gating prevents sampler-2 refinement on missing-face frames."
        )

    n_down = sum(1 for mag in mags if mag < 1.0)
    warn = ""
    if n_down:
        need = max(box[3] for box in boxes)
        warn = (
            f"\n!! {n_down}/{B} frames ({n_down / B * 100:.0f}%) have "
            "magnification < 1.0x - their crops are downscaled into the canvas.\n"
            f"   Raise canvas to >= {need}px, lower crop_factor, or skip close-up clips."
        )

    box_jit = (
        float(
            np.mean(
                [
                    abs(boxes[i][0] - boxes[i - 1][0])
                    + abs(boxes[i][1] - boxes[i - 1][1])
                    for i in range(1, len(boxes))
                ]
            )
        )
        if len(boxes) > 1
        else 0.0
    )
    report = (
        f"tracking: {track_stats['continuity']} by motion continuity, "
        f"{track_stats['ambiguous']} ambiguous, "
        f"{track_stats['identity']} by face identity, "
        f"{track_stats['rejected']} implausible detections rejected\n"
        f"identity anchor frames={anchor_count}; geometry source=configured detector only\n"
        f"frames={B}  face={found} ({found / B * 100:.0f}%)  "
        f"body-fallback={int(via_body.sum())}  "
        f"interpolated={B - int(known.sum())}\n"
        f"face height  min={mask_face_h.min():.0f}px  "
        f"mean={mask_face_h.mean():.0f}px  max={mask_face_h.max():.0f}px\n"
        f"face fills   ~{100.0 / crop_factor:.0f}% of every crop "
        f"(crop_factor={crop_factor})\n"
        f"crop box     min={min(box[3] for box in boxes):.0f}px  "
        f"max={max(box[3] for box in boxes):.0f}px\n"
        f"magnification into {canvas_width}x{canvas_height}: "
        f"min={min(mags):.2f}x  mean={sum(mags) / len(mags):.2f}x  "
        f"max={max(mags):.2f}x\n"
        f"jitter ({smooth_method}) crop centre {jit_before:.2f} -> "
        f"{jit_after:.2f} px/frame   crop size {size_before:.2f} -> "
        f"{size_after:.2f} px/frame\n"
        f"crop lag guard: max responsive-to-smoothed "
        f"{max_lag_before:.1f}px -> {max_lag_after:.1f}px "
        f"(<= {_CENTER_LAG_FRACTION * 100:.0f}% face height)\n"
        f"box movement {box_jit:.2f} px/frame; responsive face geometry mapped "
        "inside actual clamped crop\n"
        f"dropout runs: {len(runs)}  longest={longest_gap} frames "
        f"({longest_gap / 24.0:.1f}s at 24fps)"
        f"{gapwarn}{warn}"
    )
    print("[H3FaceRefine] " + report.replace("\n", "\n[H3FaceRefine] "))
    return (
        crops,
        transform,
        preview,
        report,
        int(canvas_width),
        int(canvas_height),
    )


def install_tracking_fixes(node_class_mappings: Mapping[str, type]) -> None:
    cls = node_class_mappings.get("H3FaceTrackCrop")
    if cls is None:
        raise ImportError(
            "FaceRefine node mappings are incomplete; cannot install tracking fixes"
        )
    if getattr(cls, _INSTALL_MARKER, False):
        return
    module = sys.modules.get(cls.__module__)
    if module is None:
        raise ImportError(
            f"could not locate FaceRefine nodes module {cls.__module__!r}"
        )
    original = cls.run

    @wraps(original)
    def run(
        self,
        images,
        detector,
        confidence,
        crop_factor,
        canvas_width,
        canvas_height,
        canvas_mode,
        smooth_window,
        size_smooth_window,
        smooth_method,
        size_mode,
        select="largest",
        fallback_detector="none",
        fallback_head_frac=0.5,
        identity_reference=None,
        identity_threshold=0.28,
        identity_track=True,
    ):
        return _run_fixed(
            module,
            images,
            detector,
            confidence,
            crop_factor,
            canvas_width,
            canvas_height,
            canvas_mode,
            smooth_window,
            size_smooth_window,
            smooth_method,
            size_mode,
            select=select,
            fallback_detector=fallback_detector,
            fallback_head_frac=fallback_head_frac,
            identity_reference=identity_reference,
            identity_threshold=identity_threshold,
            identity_track=identity_track,
        )

    cls.run = run
    cls.DESCRIPTION = (
        str(getattr(cls, "DESCRIPTION", "")).rstrip()
        + " Motion-predicted association, identity-stable geometry, bounded crop lag, "
        "and responsive crop-aware face-mask placement are applied automatically."
    ).strip()
    setattr(cls, _INSTALL_MARKER, True)


__all__ = [
    "install_tracking_fixes",
    "_associate_boxes",
    "_bound_center_lag",
    "_choose_body_for_track",
    "_face_rect_in_canvas",
    "_median3",
    "_predict_track_state",
    "_should_use_identity",
]
