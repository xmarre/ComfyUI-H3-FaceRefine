"""Performance policy for identity-aware H3 FaceRefine tracking.

A rejected target must stay rejected until identity proves that it returned.  The
tracking correctness fix originally enforced that by running InsightFace on every
subsequent frame while the target was lost.  A long-lived bystander therefore
turned one dropout into hundreds of full InsightFace passes.

This policy keeps the same identity-safety invariant while treating a smoothly
continuing rejected YOLO box as a negative tracklet.  Rejected tracklets are
re-probed sparsely; a geometry change, crowd/ambiguity, or new candidate still
forces an immediate identity check.  When a sparse probe finds the target again,
the short interval since the previous probe is scanned backwards so the exact
contiguous reappearance boundary is restored offline.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import math
import time
from typing import Any, Mapping

import numpy as np


_INSTALL_MARKER = "_h3_tracking_performance_installed"
_REACQUIRE_PROBE_STRIDE = 12
_NEGATIVE_TRACK_MAX_GAP = 2
_NEGATIVE_TRACK_MAX_DISTANCE = 0.75
_NEGATIVE_TRACK_MAX_SIZE_RATIO = 1.60
_TELEMETRY: ContextVar[dict[str, Any] | None] = ContextVar(
    "h3_face_refine_tracking_perf", default=None
)


def _box_state(box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (
        (x0 + x1) * 0.5,
        (y0 + y1) * 0.5,
        max(y1 - y0, 1.0),
        max(x1 - x0, 1.0),
    )


def _same_negative_tracklet(current, previous, frame_gap: int) -> bool:
    """Cheap YOLO-geometry test for a continuing already-rejected bystander."""
    if previous is None or int(frame_gap) < 1 or int(frame_gap) > _NEGATIVE_TRACK_MAX_GAP:
        return False
    cx, cy, h, _w = _box_state(current)
    px, py, ph, _pw = _box_state(previous)
    scale = max(0.5 * (h + ph), 8.0)
    distance = math.hypot(cx - px, cy - py) / scale
    size_ratio = max(h, ph) / max(min(h, ph), 1.0)
    return (
        distance <= _NEGATIVE_TRACK_MAX_DISTANCE
        and size_ratio <= _NEGATIVE_TRACK_MAX_SIZE_RATIO
    )


def _score_identity(tracking, module, app, images, frame, boxes, ref_emb, cache, stats):
    """Run/cache one frame identity analysis and account for its actual inference cost."""
    before = len(cache)
    started = time.perf_counter()
    scores = tracking["_identity_scores_for_boxes"](
        module, app, images, frame, boxes, ref_emb, cache
    )
    elapsed = time.perf_counter() - started
    new_frames = max(0, len(cache) - before)
    if new_frames:
        stats["identity_frame_evals"] += new_frames
        stats["identity_seconds"] += elapsed
    return scores


def _backfill_reappearance(
    tracking,
    module,
    detections,
    *,
    start_exclusive: int,
    stop_exclusive: int,
    images,
    app,
    ref_emb,
    identity_threshold: float,
    cache,
    stats,
):
    """Recover contiguous target frames immediately preceding a successful sparse probe."""
    recovered: list[tuple[int, tuple[float, float, float, float]]] = []
    for frame in range(int(stop_exclusive) - 1, int(start_exclusive), -1):
        raw_boxes = detections[frame]
        if not raw_boxes:
            break
        boxes = [tuple(float(v) for v in box) for box in raw_boxes]
        scores = _score_identity(
            tracking, module, app, images, frame, boxes, ref_emb, cache, stats
        )
        best_i = int(np.argmax(scores)) if scores else -1
        if best_i < 0 or scores[best_i] < float(identity_threshold):
            break
        recovered.append((frame, boxes[best_i]))
    recovered.reverse()
    return recovered


def _associate_boxes_sparse(
    tracking,
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
    """Identity-safe association without re-running InsightFace on one rejected face each frame."""
    cache = embedding_cache if embedding_cache is not None else {}
    track: list[tuple[float, float, float, float] | None] = [None] * len(detections)
    history: list[tuple[int, tuple[float, float, float, float]]] = []
    crowded = tracking["_crowded_frames"](detections)
    stats = {
        "continuity": 0,
        "identity": 0,
        "ambiguous": 0,
        "rejected": 0,
        "identity_skipped": 0,
        "identity_frame_evals": 0,
        "identity_seconds": 0.0,
        "reacquire_probes": 0,
        "backfilled": 0,
    }
    last_negative_box = None
    last_negative_frame = None
    last_reacquire_probe = None

    for i, raw_boxes in enumerate(detections):
        boxes = [tuple(float(v) for v in box) for box in raw_boxes]
        if not boxes:
            continue

        chosen = None
        if not history:
            if ref_emb is not None and app is not None and images is not None:
                scores = _score_identity(
                    tracking, module, app, images, i, boxes, ref_emb, cache, stats
                )
                best_i = int(np.argmax(scores)) if scores else -1
                if best_i >= 0 and scores[best_i] >= float(identity_threshold):
                    chosen = boxes[best_i]
                    stats["identity"] += 1
            if chosen is None:
                chosen = tuple(
                    tracking["_select_initial_box"](boxes, width, height, select)
                )
                stats["continuity"] += 1
        else:
            predicted = tracking["_predict_track_state"](history, i)
            ranked = sorted(
                boxes, key=lambda box: tracking["_tracking_cost"](box, predicted)
            )
            best = ranked[0]
            best_cost = tracking["_tracking_cost"](best, predicted)
            gate_ok = tracking["_inside_motion_gate"](best, predicted)

            ambiguous = False
            if len(ranked) > 1:
                second = ranked[1]
                second_cost = tracking["_tracking_cost"](second, predicted)
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
            identity_required = (
                ref_emb is not None
                and app is not None
                and images is not None
                and (ambiguous or safety_identity or crowded[i])
            )

            # Once a singleton candidate has already failed identity and continues as the
            # same YOLO tracklet, a broad crowd-transition flag must not force another
            # expensive identity pass on every frame.  The first failed probe establishes
            # the negative tracklet; geometry changes, ambiguity, or an additional candidate
            # still bypass this path and force identity immediately.  Successful later probes
            # retain exact output timing through the existing bounded reverse backfill.
            continuing_negative = (
                identity_required
                and safety_identity
                and len(boxes) == 1
                and not ambiguous
                and last_negative_frame is not None
                and _same_negative_tracklet(
                    best, last_negative_box, i - int(last_negative_frame)
                )
            )
            refresh_due = (
                last_reacquire_probe is None
                or i - int(last_reacquire_probe) >= _REACQUIRE_PROBE_STRIDE
            )
            sparse_skip = continuing_negative and not refresh_due

            identity_passed = False
            previous_probe = last_reacquire_probe
            if identity_required and not sparse_skip:
                if safety_identity:
                    stats["reacquire_probes"] += 1
                scores = _score_identity(
                    tracking, module, app, images, i, boxes, ref_emb, cache, stats
                )
                best_i = int(np.argmax(scores)) if scores else -1
                if best_i >= 0 and scores[best_i] >= float(identity_threshold):
                    chosen = boxes[best_i]
                    identity_passed = True
                    stats["identity"] += 1

                    if (
                        safety_identity
                        and previous_probe is not None
                        and i - int(previous_probe) > 1
                    ):
                        recovered = _backfill_reappearance(
                            tracking,
                            module,
                            detections,
                            start_exclusive=int(previous_probe),
                            stop_exclusive=i,
                            images=images,
                            app=app,
                            ref_emb=ref_emb,
                            identity_threshold=identity_threshold,
                            cache=cache,
                            stats=stats,
                        )
                        for frame, recovered_box in recovered:
                            if track[frame] is not None:
                                continue
                            track[frame] = recovered_box
                            history.append((frame, recovered_box))
                            stats["identity"] += 1
                            stats["backfilled"] += 1
                            stats["rejected"] = max(0, stats["rejected"] - 1)

                    last_negative_box = None
                    last_negative_frame = None
                    last_reacquire_probe = None
                elif safety_identity and len(boxes) == 1:
                    last_negative_box = best
                    last_negative_frame = i
                    last_reacquire_probe = i
            elif sparse_skip:
                stats["identity_skipped"] += 1
                last_negative_box = best
                last_negative_frame = i

            # A skipped identity check is still identity-required. Never weaken the
            # correctness gate merely because a continuing bystander was cached negative.
            if (
                chosen is None
                and gate_ok
                and not (identity_required and safety_identity and not identity_passed)
            ):
                chosen = best
                stats["continuity"] += 1
            elif chosen is None:
                stats["rejected"] += 1
                continue

        chosen = tuple(float(v) for v in chosen)
        track[i] = chosen
        history.append((i, chosen))
        last_negative_box = None
        last_negative_frame = None
        last_reacquire_probe = None

    stats["identity_clip_frames"] = len(cache)
    _TELEMETRY.set(dict(stats))
    return track, stats


def install_tracking_performance_fixes(node_class_mappings: Mapping[str, type]) -> None:
    """Patch the already-installed tracking policy with sparse lost-target identity probes."""
    cls = node_class_mappings.get("H3FaceTrackCrop")
    if cls is None:
        raise ImportError(
            "FaceRefine node mappings are incomplete; cannot install tracking performance fixes"
        )
    if getattr(cls, _INSTALL_MARKER, False):
        return

    original = cls.run
    tracking = getattr(original, "__globals__", None)
    if not isinstance(tracking, dict) or "_associate_boxes" not in tracking:
        raise ImportError(
            "tracking performance fixes must be installed after install_tracking_fixes"
        )

    def patched_associate(module, detections, **kwargs):
        return _associate_boxes_sparse(tracking, module, detections, **kwargs)

    patched_associate._h3_sparse_identity_reacquisition = True
    tracking["_associate_boxes"] = patched_associate

    @wraps(original)
    def run(self, *args, **kwargs):
        token = _TELEMETRY.set(None)
        started = time.perf_counter()
        try:
            result = original(self, *args, **kwargs)
            elapsed = time.perf_counter() - started
            telemetry = _TELEMETRY.get() or {}
            if not isinstance(result, tuple) or len(result) < 4:
                return result

            items = list(result)
            transform = items[1]
            frames = int(transform.get("frames", 0)) if isinstance(transform, dict) else 0
            clip_identity = int(telemetry.get("identity_clip_frames", 0))
            skipped = int(telemetry.get("identity_skipped", 0))
            backfilled = int(telemetry.get("backfilled", 0))
            probes = int(telemetry.get("reacquire_probes", 0))
            identity_seconds = float(telemetry.get("identity_seconds", 0.0))
            perf_line = (
                f"tracking performance: total={elapsed:.3f}s  "
                f"clip-identity-frames={clip_identity}/{frames}  "
                f"reacquire-probes={probes}  sparse-skips={skipped}  "
                f"backfilled={backfilled}  association-identity={identity_seconds:.3f}s"
            )
            print("[H3FaceRefine] " + perf_line)

            if isinstance(transform, dict):
                transform["tracking_identity_clip_frames"] = clip_identity
                transform["tracking_identity_sparse_skips"] = skipped
                transform["tracking_identity_backfilled"] = backfilled
                transform["tracking_total_seconds"] = float(elapsed)
            items[3] = str(items[3]) + "\n" + perf_line
            return tuple(items)
        finally:
            _TELEMETRY.reset(token)

    cls.run = run
    cls.DESCRIPTION = (
        str(getattr(cls, "DESCRIPTION", "")).rstrip()
        + " Lost-target identity reacquisition caches continuing rejected faces and probes "
        "them sparsely with offline boundary backfill."
    ).strip()
    setattr(cls, _INSTALL_MARKER, True)


__all__ = [
    "install_tracking_performance_fixes",
    "_associate_boxes_sparse",
    "_same_negative_tracklet",
]
