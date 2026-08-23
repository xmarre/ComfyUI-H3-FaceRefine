"""Residual-only stitch-back and Continuum runtime guidance for H3 FaceRefine.

Sampler 2 emits source-looking crops whose only difference from the tracked
source crop is ``decoded_refined - decoded_clean``. Stitch Back therefore must
transfer only that residual onto the untouched full-resolution source frame.

Important invariants:

* there is no magnification hard gate: downscaled crops are allowed to contribute
  a learned correction without replacing the higher-resolution source baseline;
* a zero sampler-2 correction is an exact no-op after stitch;
* the moving crop boundary carries zero correction outside its mask, so it cannot
  reveal synthetic black or an absolute resized-patch rectangle;
* colour_match removes only residual per-channel DC bias; it never rescales and
  repastes the absolute crop;
* 512x512 is the validated Continuum baseline. Larger canvases remain an explicit
  quality/performance A/B option, not a required fix for sub-1x crop magnification.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


CONTINUUM_CANVAS_BASELINE = 512


if __package__:
    from . import quality_fixes as _q1
else:
    _path = Path(__file__).with_name("quality_fixes.py")
    _spec = importlib.util.spec_from_file_location("h3_face_refine_quality_fixes_v1", _path)
    if _spec is None or _spec.loader is None:
        raise ImportError("could not load sibling quality_fixes.py")
    _q1 = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _q1
    _spec.loader.exec_module(_q1)


def _crop_grid(
    boxes,
    start: int,
    stop: int,
    *,
    width: int,
    height: int,
    crop_w: int,
    crop_h: int,
    device,
):
    """Exact batched equivalent of nodes._affine_crop for align_corners=False."""

    n = stop - start
    theta = torch.empty((n, 2, 3), dtype=torch.float32, device=device)
    for row, index in enumerate(range(start, stop)):
        x, y, bw, bh = (float(v) for v in boxes[index])
        theta[row, 0, 0] = bw / float(width)
        theta[row, 0, 1] = 0.0
        theta[row, 0, 2] = (2.0 * x + bw) / float(width) - 1.0
        theta[row, 1, 0] = 0.0
        theta[row, 1, 1] = bh / float(height)
        theta[row, 1, 2] = (2.0 * y + bh) / float(height) - 1.0
    return F.affine_grid(theta, (n, 3, int(crop_h), int(crop_w)), align_corners=False)


def _warp_grid(boxes, start: int, stop: int, *, width: int, height: int, device):
    """Exact inverse of :func:`_crop_grid` for align_corners=False."""

    n = stop - start
    theta = torch.empty((n, 2, 3), dtype=torch.float32, device=device)
    for row, index in enumerate(range(start, stop)):
        x, y, bw, bh = (float(v) for v in boxes[index])
        theta[row, 0, 0] = float(width) / bw
        theta[row, 0, 1] = 0.0
        theta[row, 0, 2] = (float(width) - 2.0 * x) / bw - 1.0
        theta[row, 1, 0] = 0.0
        theta[row, 1, 1] = float(height) / bh
        theta[row, 1, 2] = (float(height) - 2.0 * y) / bh - 1.0
    return F.affine_grid(theta, (n, 3, int(height), int(width)), align_corners=False)


def _remove_residual_bias(delta: torch.Tensor, mask: torch.Tensor, amount: float) -> torch.Tensor:
    """Remove patch-wide residual colour/tone bias without touching absolute source pixels."""

    amount = float(amount)
    if amount <= 0.0:
        return delta
    weight = mask.clamp(0, 1)
    wsum = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    mean = (delta * weight).sum(dim=(2, 3), keepdim=True) / wsum
    return delta - mean * amount


def _guidance_transform(value):
    """Soft-normalize the same singleton-list contract as Continuum's INPUT_IS_LIST node."""

    while isinstance(value, list):
        if len(value) != 1:
            return None
        value = value[0]
    return value if isinstance(value, Mapping) else None


def continuum_canvas_guidance(transform) -> str:
    """Return Continuum-specific canvas guidance from a tracker transform.

    The tracker is also used by the original standalone absolute-paste workflow, where
    sub-1x crop magnification really can mean replacing source detail with a lower-resolution
    reconstruction. Continuum's residual stitch has a different contract: the untouched
    full-resolution frame remains the baseline and only the learned residual is transferred.
    Therefore sub-1x magnification is informational on the Continuum path, not a reason to
    grow the canvas until it matches the largest crop.

    ``H3ContinuumFaceRefine`` declares ``INPUT_IS_LIST = True``. The outer guidance
    wrapper therefore sees ``transform`` as a singleton list before the production refine
    implementation calls its strict ``_single`` normalizer. Diagnostics must understand
    that same shape without modifying the value delegated downstream.
    """

    transform = _guidance_transform(transform)
    if transform is None:
        return (
            "Continuum canvas guidance unavailable: H3FACEXFORM must resolve to exactly "
            "one transform mapping"
        )

    canvas = transform.get("canvas")
    boxes = transform.get("boxes")
    if not isinstance(canvas, (tuple, list)) or len(canvas) != 2:
        return "Continuum canvas guidance unavailable: H3FACEXFORM is missing canvas geometry"

    width, height = int(canvas[0]), int(canvas[1])
    baseline = int(CONTINUUM_CANVAS_BASELINE)
    area_ratio = (width * height) / float(baseline * baseline)

    n_down = 0
    total = 0
    if isinstance(boxes, (list, tuple)):
        total = len(boxes)
        for box in boxes:
            try:
                crop_h = float(box[3])
            except (TypeError, ValueError, IndexError):
                continue
            if crop_h > 0.0 and height / crop_h < 1.0:
                n_down += 1

    down_text = (
        f"; {n_down}/{total} crop frame(s) are sub-1x"
        if total
        else ""
    )

    if width == baseline and height == baseline:
        return (
            f"Continuum canvas: {baseline}x{baseline} validated baseline{down_text}. "
            "Sub-1x crops are allowed here because residual stitch preserves the untouched "
            "full-resolution source; do not raise the canvas just to eliminate that statistic."
        )

    return (
        f"Continuum canvas: {width}x{height} uses ~{area_ratio:.2f}x the spatial token area "
        f"of {baseline}x{baseline}{down_text}. {baseline}x{baseline} is the validated baseline; "
        "use a larger canvas only when direct visual A/B testing shows a worthwhile refinement "
        "benefit, not merely to make magnification >= 1.0x."
    )


def build_continuum_canvas_guidance(base_class):
    class H3ContinuumFaceRefineGuided(base_class):
        def refine(self, crops, transform, *args, **kwargs):
            print(f"[H3FaceRefine] {continuum_canvas_guidance(transform)}")
            return super().refine(crops, transform, *args, **kwargs)

    H3ContinuumFaceRefineGuided.__name__ = base_class.__name__
    H3ContinuumFaceRefineGuided.__qualname__ = base_class.__qualname__
    H3ContinuumFaceRefineGuided._h3_continuum_canvas_guidance_v2 = True
    return H3ContinuumFaceRefineGuided


def build_residual_stitch(base_class):
    class H3FaceStitchResidual(base_class):
        def run(
            self,
            base_images,
            refined_crops,
            transform,
            paste_region,
            mask_dilation,
            feather,
            colour_match,
            blend,
            undetected_frames="fade_out",
            masks=None,
            feather_scales_with_crop=False,
        ):
            boxes = transform.get("boxes")
            frames = int(base_images.shape[0])
            if not isinstance(boxes, list) or len(boxes) != frames:
                raise ValueError("H3 Face Stitch Back requires one crop box per source frame")
            if int(refined_crops.shape[0]) != frames:
                raise ValueError(
                    "H3 Face Stitch Back requires exact frame alignment: "
                    f"base={frames}, refined={int(refined_crops.shape[0])}, transform={len(boxes)}"
                )

            cw, ch = (int(v) for v in transform["canvas"])
            width, height = (int(v) for v in transform["src_size"])
            if tuple(base_images.shape[1:3]) != (height, width):
                raise ValueError("H3 Face Stitch Back base image geometry differs from H3FACEXFORM")
            if tuple(refined_crops.shape[1:3]) != (ch, cw):
                raise ValueError("H3 Face Stitch Back refined crop geometry differs from H3FACEXFORM")

            detected = transform.get("detected")
            if not isinstance(detected, (list, tuple)) or len(detected) != frames:
                raise ValueError("H3 Face Stitch Back requires the aligned per-frame detected mask")
            detection_weights = _q1.safe_stitch_weights(
                [bool(value) for value in detected], str(undetected_frames)
            )
            face_rects = transform.get("face_rect")

            import comfy.model_management as mm

            try:
                device = mm.get_torch_device()
            except Exception:
                device = base_images.device

            dtype = base_images.dtype
            output = base_images[..., :3].clone()
            per_frame_mb = (height * width * 3 * 4) / 2**20
            chunk = max(1, min(32, int(768 / max(per_frame_mb, 1e-6))))

            print(
                "[H3FaceRefine] stitch mode: VAE-roundtrip-cancelled residual transfer; "
                "no magnification hard gate"
            )

            for start in range(0, frames, chunk):
                mm.throw_exception_if_processing_interrupted()
                stop = min(start + chunk, frames)
                n = stop - start

                base = base_images[start:stop, ..., :3].to(device).movedim(-1, 1).float()
                crop_grid = _crop_grid(
                    boxes,
                    start,
                    stop,
                    width=width,
                    height=height,
                    crop_w=cw,
                    crop_h=ch,
                    device=device,
                )
                source_canvas = F.grid_sample(
                    base,
                    crop_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                corrected_canvas = (
                    refined_crops[start:stop, ..., :3].to(device).movedim(-1, 1).float()
                )

                mask_canvas = _q1._canvas_mask_batch(
                    start=start,
                    stop=stop,
                    height=ch,
                    width=cw,
                    paste_region=str(paste_region),
                    face_rects=face_rects,
                    dilation=int(mask_dilation),
                    masks=masks,
                    device=device,
                )
                if feather_scales_with_crop and int(feather) > 0:
                    mask_canvas = _q1._gaussian_blur(mask_canvas, int(feather)).clamp(0, 1)

                # corrected_canvas was constructed as:
                # source_crop + (decoded_refined - decoded_clean).
                # Subtracting the exact source crop here recovers only sampler 2's
                # VAE-cancelled learned correction. No absolute 512px patch survives.
                delta_canvas = corrected_canvas - source_canvas
                delta_canvas = _remove_residual_bias(
                    delta_canvas, mask_canvas, float(colour_match)
                )

                inverse_grid = _warp_grid(
                    boxes, start, stop, width=width, height=height, device=device
                )
                delta = F.grid_sample(
                    delta_canvas,
                    inverse_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                mask = F.grid_sample(
                    mask_canvas,
                    inverse_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).clamp(0, 1)
                if not feather_scales_with_crop and int(feather) > 0:
                    mask = _q1._gaussian_blur(mask, int(feather)).clamp(0, 1)

                opacity = torch.full(
                    (n, 1, 1, 1), float(blend), device=device, dtype=torch.float32
                )
                if detection_weights is not None:
                    opacity *= torch.tensor(
                        detection_weights[start:stop], device=device, dtype=torch.float32
                    ).view(n, 1, 1, 1)

                composed = (base + delta * mask * opacity).clamp(0, 1)
                output[start:stop] = composed.movedim(1, -1).to(output.device, dtype)

            return (output,)

    H3FaceStitchResidual.__name__ = base_class.__name__
    H3FaceStitchResidual.__qualname__ = base_class.__qualname__
    return H3FaceStitchResidual


def install_quality_fixes_v2(node_class_mappings: dict[str, type]) -> None:
    stitch = node_class_mappings.get("H3FaceStitch")
    if stitch is None:
        raise ImportError("FaceRefine node mappings are incomplete; cannot install residual stitch")
    node_class_mappings["H3FaceStitch"] = build_residual_stitch(stitch)

    # Continuum is an additive integration. Keep this installer usable by the original
    # standalone/partial mapping contract and attach path-specific canvas guidance only
    # when the Continuum node is actually registered.
    continuum = node_class_mappings.get("H3ContinuumFaceRefine")
    if continuum is not None and not getattr(
        continuum, "_h3_continuum_canvas_guidance_v2", False
    ):
        node_class_mappings["H3ContinuumFaceRefine"] = build_continuum_canvas_guidance(continuum)
