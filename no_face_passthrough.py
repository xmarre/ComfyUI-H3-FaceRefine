"""Graceful no-face handling for FaceRefine pipelines.

A clip without a detectable human face is a valid input for a general video
workflow.  FaceRefine must therefore become a no-op for that clip instead of
turning a completed upstream generation into a failed prompt.

The package already installs small runtime quality/compatibility policies from
``__init__.py``.  Keep this behaviour isolated in the same style so the core
tracking/refinement implementations stay focused on the face-present path.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import torch


PASSTHROUGH_KEY = "h3_face_refine_passthrough"
PASSTHROUGH_REASON_KEY = "h3_face_refine_passthrough_reason"
NO_FACE_REASON = "no_face_detected"
_NO_FACE_ERROR_PREFIX = "No face detected in any frame."
_INSTALL_MARKER = "_h3_no_face_passthrough_installed"


def _single(value: Any) -> Any:
    """Unwrap ComfyUI INPUT_IS_LIST singletons without accepting ambiguous lists."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int, default=None):
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def is_no_face_passthrough(transform: Any) -> bool:
    transform = _single(transform)
    return bool(
        isinstance(transform, Mapping)
        and transform.get(PASSTHROUGH_KEY) is True
        and transform.get(PASSTHROUGH_REASON_KEY) == NO_FACE_REASON
    )


def _passthrough_report(stage: str) -> str:
    return (
        f"{stage}: no human face detected; FaceRefine skipped automatically. "
        "Original video frames are preserved unchanged."
    )


def _make_passthrough_track_result(
    images: torch.Tensor,
    *,
    crop_factor: float,
    canvas_width: int,
    canvas_height: int,
):
    if not torch.is_tensor(images) or images.ndim != 4:
        raise TypeError("FaceRefine no-face passthrough requires IMAGE [frames,H,W,C]")
    frames, height, width, _ = images.shape
    transform = {
        PASSTHROUGH_KEY: True,
        PASSTHROUGH_REASON_KEY: NO_FACE_REASON,
        "boxes": [],
        "canvas": (int(canvas_width), int(canvas_height)),
        "src_size": (int(width), int(height)),
        "frames": int(frames),
        "weights": [0.0] * int(frames),
        "detected": [False] * int(frames),
        "face_rect": [],
        "crop_factor": float(crop_factor),
    }
    report = _passthrough_report("tracking")
    print("[H3FaceRefine] " + report)
    # The downstream Continuum refine policy recognises the sentinel before doing
    # any VAE/sampler-2 work.  Returning the source timeline here also gives every
    # diagnostic/output branch a useful image rather than a synthetic placeholder.
    rgb = images[..., :3]
    return (
        rgb,
        transform,
        rgb.clone(),
        report,
        int(canvas_width),
        int(canvas_height),
    )


def _install_track_policy(cls: type) -> None:
    if getattr(cls, _INSTALL_MARKER, False):
        return
    original = cls.run

    @wraps(original)
    def run(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except ValueError as exc:
            # Only convert the deliberate all-frames-missed condition.  Detector,
            # tensor, topology, and configuration errors must remain hard failures.
            if not str(exc).startswith(_NO_FACE_ERROR_PREFIX):
                raise
            images = _argument(args, kwargs, "images", 0)
            crop_factor = _argument(args, kwargs, "crop_factor", 3, 2.5)
            canvas_width = _argument(args, kwargs, "canvas_width", 4, 512)
            canvas_height = _argument(args, kwargs, "canvas_height", 5, 512)
            return _make_passthrough_track_result(
                images,
                crop_factor=float(crop_factor),
                canvas_width=int(canvas_width),
                canvas_height=int(canvas_height),
            )

    cls.run = run
    cls.DESCRIPTION = (
        str(getattr(cls, "DESCRIPTION", "")).rstrip()
        + " Clips with no detectable human face pass through automatically."
    ).strip()
    setattr(cls, _INSTALL_MARKER, True)


def _install_continuum_policy(cls: type) -> None:
    if getattr(cls, _INSTALL_MARKER, False):
        return
    original = cls.refine

    @wraps(original)
    def refine(self, *args, **kwargs):
        transform = _argument(args, kwargs, "transform", 1)
        if is_no_face_passthrough(transform):
            crops = _single(_argument(args, kwargs, "crops", 0))
            if not torch.is_tensor(crops) or crops.ndim != 4:
                raise TypeError("FaceRefine passthrough crops must be IMAGE [frames,H,W,C]")
            report = _passthrough_report("Continuum FaceRefine")
            print("[H3FaceRefine] " + report)
            # Crucially return before validating/rebuilding physical groups or running
            # sampler 2.  A wildlife/landscape clip pays only the detector cost.
            return crops, report
        return original(self, *args, **kwargs)

    cls.refine = refine
    cls.DESCRIPTION = (
        str(getattr(cls, "DESCRIPTION", "")).rstrip()
        + " No-face timelines bypass sampler 2 and pass through unchanged."
    ).strip()
    setattr(cls, _INSTALL_MARKER, True)


def _install_stitch_policy(cls: type) -> None:
    if getattr(cls, _INSTALL_MARKER, False):
        return
    original = cls.run

    @wraps(original)
    def run(self, *args, **kwargs):
        transform = _argument(args, kwargs, "transform", 2)
        if is_no_face_passthrough(transform):
            base_images = _argument(args, kwargs, "base_images", 0)
            if not torch.is_tensor(base_images) or base_images.ndim != 4:
                raise TypeError("FaceRefine passthrough base_images must be IMAGE [frames,H,W,C]")
            print("[H3FaceRefine] stitch: passthrough active; returning original frames unchanged")
            return (base_images,)
        return original(self, *args, **kwargs)

    cls.run = run
    setattr(cls, _INSTALL_MARKER, True)


def _install_mask_policy(cls: type) -> None:
    if getattr(cls, _INSTALL_MARKER, False):
        return
    original = cls.run

    @wraps(original)
    def run(self, *args, **kwargs):
        transform = _argument(args, kwargs, "transform", 2)
        if is_no_face_passthrough(transform):
            crops = _argument(args, kwargs, "crops", 0)
            if not torch.is_tensor(crops) or crops.ndim != 4:
                raise TypeError("FaceRefine passthrough crops must be IMAGE [frames,H,W,C]")
            masks = torch.zeros(
                (int(crops.shape[0]), int(crops.shape[1]), int(crops.shape[2])),
                dtype=torch.float32,
                device=crops.device,
            )
            report = _passthrough_report("SAM mask")
            print("[H3FaceRefine] " + report)
            return masks, report
        return original(self, *args, **kwargs)

    cls.run = run
    setattr(cls, _INSTALL_MARKER, True)


def _install_denoise_policy(cls: type) -> None:
    if getattr(cls, _INSTALL_MARKER, False):
        return
    original = cls.run

    @wraps(original)
    def run(self, *args, **kwargs):
        transform = _argument(args, kwargs, "transform", 1)
        if is_no_face_passthrough(transform):
            av_latent = _argument(args, kwargs, "av_latent", 0)
            report = _passthrough_report("per-frame denoise")
            print("[H3FaceRefine] " + report)
            return av_latent, report
        return original(self, *args, **kwargs)

    cls.run = run
    setattr(cls, _INSTALL_MARKER, True)


def _install_info_policy(cls: type) -> None:
    if getattr(cls, _INSTALL_MARKER, False):
        return
    original = cls.run

    @wraps(original)
    def run(self, *args, **kwargs):
        transform = _argument(args, kwargs, "transform", 0)
        if is_no_face_passthrough(transform):
            text = _passthrough_report("transform info")
            print("[H3FaceRefine] " + text)
            return (text,)
        return original(self, *args, **kwargs)

    cls.run = run
    setattr(cls, _INSTALL_MARKER, True)


def install_no_face_passthrough(node_mappings: Mapping[str, type]) -> None:
    """Install the no-face no-op contract on every FaceRefine stage that can see it."""
    policies = {
        "H3FaceTrackCrop": _install_track_policy,
        "H3ContinuumFaceRefine": _install_continuum_policy,
        "H3FaceStitch": _install_stitch_policy,
        "H3FaceMaskSAM": _install_mask_policy,
        "H3PerFrameDenoise": _install_denoise_policy,
        "H3FaceTransformInfo": _install_info_policy,
    }
    for name, installer in policies.items():
        cls = node_mappings.get(name)
        if cls is not None:
            installer(cls)


__all__ = [
    "NO_FACE_REASON",
    "PASSTHROUGH_KEY",
    "PASSTHROUGH_REASON_KEY",
    "install_no_face_passthrough",
    "is_no_face_passthrough",
]
