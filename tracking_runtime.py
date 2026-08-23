"""Runtime policy for fast, explicit identity-aware face tracking.

Two independent costs matter for identity tracking:

* InsightFace may silently run on CPU when ONNX Runtime has no CUDA provider;
* the conservative crowd policy used to request identity on every frame in an
  expanded crowd window, even when motion association was unambiguous.

This layer keeps the hard safety gates for ambiguous motion, dropouts, and crowd
transitions.  During a stable multi-face run it turns the broad ``crowded`` flag
into periodic identity checkpoints; ambiguity still forces an immediate check.
It also reports the actual ONNX Runtime provider and emits one explicit warning
when identity matching is CPU-backed.
"""

from __future__ import annotations

from functools import wraps
import sys
from typing import Mapping


_INSTALL_MARKER = "_h3_tracking_runtime_fixes_installed"
_RECOGNISER_MARKER = "_h3_identity_provider_diagnostics_installed"
_CROWD_CHECKPOINT_STRIDE = 12
_CPU_PROVIDER_WARNING_EMITTED = False


def _identity_provider_state():
    """Return (backend, providers, error) without requiring ONNX Runtime in tests."""
    try:
        import onnxruntime as ort

        providers = tuple(str(p) for p in ort.get_available_providers())
    except Exception as exc:
        return "unavailable", (), str(exc)

    if "CUDAExecutionProvider" in providers:
        return "cuda", providers, None
    if "CPUExecutionProvider" in providers:
        return "cpu", providers, None
    return "other", providers, None


def _checkpointed_crowded_frames(
    detections,
    original_crowded_frames,
    radius: int = 2,
    stride: int = _CROWD_CHECKPOINT_STRIDE,
):
    """Preserve transition safety while sparsifying stable multi-face checkpoints.

    ``tracking_performance`` already forces identity for ambiguous candidates,
    implausible motion, and tracking gaps.  The only redundant part is the broad
    ``or crowded[i]`` clause on every non-ambiguous multi-face frame.  Singleton
    frames in the expanded crowd radius are intentionally kept True because that
    is the guard which prevents a disappearing target from being replaced by the
    one bystander left on screen.
    """
    expanded = list(original_crowded_frames(detections, radius=radius))
    if not expanded:
        return expanded

    stride = max(1, int(stride))
    out = [False] * len(expanded)
    in_multi_run = False
    last_checkpoint = None

    for i, boxes in enumerate(detections):
        is_multi = len(boxes) > 1
        if not is_multi:
            # Keep the existing +/- radius transition guard for lone detections.
            out[i] = bool(expanded[i])
            in_multi_run = False
            last_checkpoint = None
            continue

        # A newly-entered crowd is checked immediately, then periodically while the
        # multi-face geometry remains stable.  Any ambiguity between checkpoints is
        # still handled by the association code and forces identity immediately.
        if (
            not in_multi_run
            or last_checkpoint is None
            or i - int(last_checkpoint) >= stride
        ):
            out[i] = True
            last_checkpoint = i
        in_multi_run = True

    return out


def _warn_if_cpu_identity_backend() -> None:
    global _CPU_PROVIDER_WARNING_EMITTED
    if _CPU_PROVIDER_WARNING_EMITTED:
        return

    backend, providers, error = _identity_provider_state()
    if backend != "cpu":
        return

    provider_text = ",".join(providers) if providers else "none"
    print(
        "[H3FaceRefine] WARNING: InsightFace identity backend=CPU; "
        "CUDAExecutionProvider is unavailable. Identity tracking can take tens of "
        "seconds on long clips. Remove the CPU-only 'onnxruntime' distribution, "
        "install a CUDA-compatible 'onnxruntime-gpu', and verify "
        "ort.get_available_providers() contains CUDAExecutionProvider. "
        f"available providers={provider_text}"
    )
    _CPU_PROVIDER_WARNING_EMITTED = True


def install_tracking_runtime_fixes(node_class_mappings: Mapping[str, type]) -> None:
    """Install crowd-checkpoint throttling and explicit identity backend diagnostics."""
    cls = node_class_mappings.get("H3FaceTrackCrop")
    if cls is None:
        raise ImportError(
            "FaceRefine node mappings are incomplete; cannot install tracking runtime fixes"
        )
    if getattr(cls, _INSTALL_MARKER, False):
        return

    # tracking_performance wraps tracking_fixes with functools.wraps.  Its wrapped
    # function therefore gives us the tracking_fixes module globals used by the
    # sparse association closure installed in the previous layer.
    performance_run = cls.run
    tracking_run = getattr(performance_run, "__wrapped__", None)
    tracking = getattr(tracking_run, "__globals__", None)
    if not isinstance(tracking, dict) or "_crowded_frames" not in tracking:
        raise ImportError(
            "tracking runtime fixes must be installed after tracking performance fixes"
        )

    original_crowded_frames = tracking["_crowded_frames"]
    if not getattr(original_crowded_frames, "_h3_checkpointed_crowd_policy", False):

        def checkpointed_crowded_frames(detections, radius=2):
            return _checkpointed_crowded_frames(
                detections,
                original_crowded_frames,
                radius=radius,
                stride=_CROWD_CHECKPOINT_STRIDE,
            )

        checkpointed_crowded_frames._h3_checkpointed_crowd_policy = True
        tracking["_crowded_frames"] = checkpointed_crowded_frames

    # Warn at the moment InsightFace is actually requested, before the expensive
    # FaceAnalysis construction/inference path starts.  Do not turn the fallback
    # into a hard error: an already-generated upstream clip should still be allowed
    # to finish on CPU if the user chooses not to repair the environment immediately.
    node_module = sys.modules.get(cls.__module__)
    if node_module is None or not hasattr(node_module, "_face_recogniser"):
        raise ImportError(
            f"could not locate FaceRefine recogniser in module {cls.__module__!r}"
        )
    original_recogniser = node_module._face_recogniser
    if not getattr(original_recogniser, _RECOGNISER_MARKER, False):

        @wraps(original_recogniser)
        def face_recogniser(*args, **kwargs):
            _warn_if_cpu_identity_backend()
            return original_recogniser(*args, **kwargs)

        setattr(face_recogniser, _RECOGNISER_MARKER, True)
        node_module._face_recogniser = face_recogniser

    # Add the provider directly to the node report so future performance logs do not
    # require finding ONNX Runtime's initialization warning hundreds of lines earlier.
    original_run = cls.run

    @wraps(original_run)
    def run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 4:
            return result

        backend, providers, error = _identity_provider_state()
        providers_text = ",".join(providers) if providers else "none"
        runtime_line = (
            f"identity runtime: backend={backend} providers={providers_text} "
            f"crowd-checkpoint-stride={_CROWD_CHECKPOINT_STRIDE}"
        )
        if error:
            runtime_line += f" provider-query-error={error}"

        items = list(result)
        transform = items[1]
        if isinstance(transform, dict):
            transform["tracking_identity_backend"] = backend
            transform["tracking_identity_providers"] = list(providers)
            transform["tracking_crowd_checkpoint_stride"] = int(
                _CROWD_CHECKPOINT_STRIDE
            )
        items[3] = str(items[3]) + "\n" + runtime_line
        print("[H3FaceRefine] " + runtime_line)
        return tuple(items)

    cls.run = run
    cls.DESCRIPTION = (
        str(getattr(cls, "DESCRIPTION", "")).rstrip()
        + " Stable crowd frames use periodic identity checkpoints; ambiguous motion and "
        "crowd transitions still force immediate identity. The active ONNX identity "
        "provider is reported explicitly."
    ).strip()
    setattr(cls, _INSTALL_MARKER, True)


__all__ = [
    "install_tracking_runtime_fixes",
    "_checkpointed_crowded_frames",
    "_identity_provider_state",
]
