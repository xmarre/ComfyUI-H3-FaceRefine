"""Runtime policy for fast, explicit identity-aware face tracking.

Two independent costs matter for identity tracking:

* InsightFace may silently run on CPU when ONNX Runtime has no CUDA provider;
* the conservative crowd policy used to request identity on every frame in an
  expanded crowd window, even when motion association was unambiguous.

This layer keeps the hard safety gates for ambiguous motion, dropouts, and crowd
transitions.  During a stable multi-face run it turns the broad ``crowded`` flag
into periodic identity checkpoints; ambiguity still forces an immediate check.
It also reports the actual ONNX Runtime provider. On CUDA ComfyUI hosts, an
accidental CPU-only identity backend is rejected before expensive FaceAnalysis work.
"""

from __future__ import annotations

from functools import wraps
import os
import sys
from typing import Mapping


_INSTALL_MARKER = "_h3_tracking_runtime_fixes_installed"
_RECOGNISER_MARKER = "_h3_identity_provider_diagnostics_installed"
_IDENTITY_REQUIREMENT_MARKER = "_h3_identity_requirement_guard_installed"
_CROWD_CHECKPOINT_STRIDE = 12
_CPU_PROVIDER_WARNING_EMITTED = False
_ALLOW_CPU_ENV = "H3FACEREFINE_ALLOW_CPU_IDENTITY"


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


def _cuda_host() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and getattr(torch.version, "cuda", None))
    except Exception:
        return False


def _cpu_identity_allowed() -> bool:
    return os.environ.get(_ALLOW_CPU_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cpu_backend_is_misconfigured(
    backend: str, *, cuda_host: bool, allow_cpu: bool
) -> bool:
    return backend == "cpu" and bool(cuda_host) and not bool(allow_cpu)


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


def _validate_identity_backend() -> None:
    """Refuse accidental CPU fallback on a CUDA host before FaceAnalysis construction."""
    global _CPU_PROVIDER_WARNING_EMITTED

    backend, providers, error = _identity_provider_state()
    cuda_host = _cuda_host()
    allow_cpu = _cpu_identity_allowed()
    provider_text = ",".join(providers) if providers else "none"

    if _cpu_backend_is_misconfigured(
        backend, cuda_host=cuda_host, allow_cpu=allow_cpu
    ):
        raise RuntimeError(
            "InsightFace identity backend is CPU on a CUDA ComfyUI host. FaceRefine's "
            "prestartup repair should restore the GPU ONNX Runtime payload before custom "
            "nodes load. Refusing motion-only degradation. Restart ComfyUI after updating "
            "FaceRefine and verify onnxruntime.get_available_providers() contains "
            f"CUDAExecutionProvider. providers={provider_text}"
        )

    if backend != "cpu" or _CPU_PROVIDER_WARNING_EMITTED:
        return

    suffix = f" provider-query-error={error}" if error else ""
    print(
        "[H3FaceRefine] WARNING: InsightFace identity backend=CPU; "
        f"available providers={provider_text}.{suffix}"
    )
    if allow_cpu and cuda_host:
        print(
            f"[H3FaceRefine] CPU identity fallback was explicitly allowed by {_ALLOW_CPU_ENV}."
        )
    _CPU_PROVIDER_WARNING_EMITTED = True


def _guard_identity_requirement(original_should_use_identity):
    """Validate ORT before tracking_fixes enters its recoverable InsightFace try/except.

    ``tracking_fixes`` intentionally degrades when optional identity extraction itself
    fails.  Backend misconfiguration is different: silently converting an identity-
    required crowd/reacquisition pass to motion-only tracking can switch subjects.  This
    wrapper runs outside that recoverable block, so a broken CUDA backend remains fatal.
    """
    if getattr(original_should_use_identity, _IDENTITY_REQUIREMENT_MARKER, False):
        return original_should_use_identity

    @wraps(original_should_use_identity)
    def should_use_identity(*args, **kwargs):
        required = bool(original_should_use_identity(*args, **kwargs))
        if required:
            _validate_identity_backend()
        return required

    setattr(should_use_identity, _IDENTITY_REQUIREMENT_MARKER, True)
    return should_use_identity


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

    original_should_use_identity = tracking.get("_should_use_identity")
    if not callable(original_should_use_identity):
        raise ImportError(
            "tracking runtime fixes could not locate the identity-requirement policy"
        )
    tracking["_should_use_identity"] = _guard_identity_requirement(
        original_should_use_identity
    )

    # Keep a second provider check directly on recogniser construction as defence in
    # depth.  The requirement guard above is the one that intentionally lives outside
    # tracking_fixes' recoverable InsightFace exception boundary.
    node_module = sys.modules.get(cls.__module__)
    if node_module is None or not hasattr(node_module, "_face_recogniser"):
        raise ImportError(
            f"could not locate FaceRefine recogniser in module {cls.__module__!r}"
        )
    original_recogniser = node_module._face_recogniser
    if not getattr(original_recogniser, _RECOGNISER_MARKER, False):

        @wraps(original_recogniser)
        def face_recogniser(*args, **kwargs):
            _validate_identity_backend()
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
        "crowd transitions still force immediate identity. CUDA hosts reject accidental "
        "CPU-only ONNX Runtime identity fallback."
    ).strip()
    setattr(cls, _INSTALL_MARKER, True)


__all__ = [
    "install_tracking_runtime_fixes",
    "_checkpointed_crowded_frames",
    "_cpu_backend_is_misconfigured",
    "_guard_identity_requirement",
    "_identity_provider_state",
]
