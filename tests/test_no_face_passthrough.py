from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3_face_refine_no_face_passthrough_test", ROOT / "no_face_passthrough.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Track:
    DESCRIPTION = "track"

    def run(self, images, detector, confidence, crop_factor, canvas_width, canvas_height,
            canvas_mode, smooth_window, size_smooth_window, smooth_method, size_mode,
            **kwargs):
        raise ValueError(
            "No face detected in any frame. Lower `confidence`, or this clip has no usable face and should be skipped."
        )


class _Continuum:
    DESCRIPTION = "continuum"

    def refine(self, *args, **kwargs):
        raise AssertionError("sampler-2 refinement must not run for a no-face timeline")


class _Stitch:
    def run(self, *args, **kwargs):
        raise AssertionError("stitch implementation must not run for a no-face timeline")


class _Mask:
    def run(self, *args, **kwargs):
        raise AssertionError("SAM must not run for a no-face timeline")


class _Denoise:
    def run(self, *args, **kwargs):
        raise AssertionError("per-frame denoise must not run for a no-face timeline")


class _Info:
    def run(self, *args, **kwargs):
        raise AssertionError("normal transform formatting must not run for a no-face timeline")


def _mapping():
    # Fresh subclasses keep installer idempotence markers isolated between tests.
    return {
        "H3FaceTrackCrop": type("Track", (_Track,), {}),
        "H3ContinuumFaceRefine": type("Continuum", (_Continuum,), {}),
        "H3FaceStitch": type("Stitch", (_Stitch,), {}),
        "H3FaceMaskSAM": type("Mask", (_Mask,), {}),
        "H3PerFrameDenoise": type("Denoise", (_Denoise,), {}),
        "H3FaceTransformInfo": type("Info", (_Info,), {}),
    }


def test_no_face_becomes_end_to_end_noop():
    mapping = _mapping()
    MODULE.install_no_face_passthrough(mapping)

    images = torch.rand((8, 64, 96, 3), dtype=torch.float32)
    track = mapping["H3FaceTrackCrop"]()
    crops, transform, preview, report, canvas_w, canvas_h = track.run(
        images=images,
        detector="dummy.pt",
        confidence=0.35,
        crop_factor=2.5,
        canvas_width=512,
        canvas_height=512,
        canvas_mode="manual",
        smooth_window=21,
        size_smooth_window=51,
        smooth_method="gaussian",
        size_mode="per_frame",
    )

    assert MODULE.is_no_face_passthrough(transform)
    assert torch.equal(crops, images)
    assert torch.equal(preview, images)
    assert transform["frames"] == 8
    assert transform["boxes"] == []
    assert transform["detected"] == [False] * 8
    assert (canvas_w, canvas_h) == (512, 512)
    assert "skipped automatically" in report

    refined, refine_report = mapping["H3ContinuumFaceRefine"]().refine(
        crops=[crops],
        transform=[transform],
        video_latents=[object()],
        audio_latents=[object()],
        assembly_plan=[object()],
        refine_state=[object()],
        vae=[object()],
        noise=[object()],
        sampler=[object()],
        sigmas=[object()],
        strength_small_face=[1.0],
        strength_large_face=[0.35],
        scale_mode=["absolute_px"],
        face_px_small=[30.0],
        face_px_large=[120.0],
        gamma=[1.0],
        smooth_frames=[9],
    )
    assert torch.equal(refined, images)
    assert "Original video frames are preserved unchanged" in refine_report

    base = torch.rand_like(images)
    stitched, = mapping["H3FaceStitch"]().run(
        base_images=base,
        refined_crops=refined,
        transform=transform,
        paste_region="face_only",
        mask_dilation=16,
        feather=6,
        colour_match=1.0,
        blend=1.0,
    )
    assert stitched is base


def test_passthrough_skips_auxiliary_face_work():
    mapping = _mapping()
    MODULE.install_no_face_passthrough(mapping)
    transform = {
        MODULE.PASSTHROUGH_KEY: True,
        MODULE.PASSTHROUGH_REASON_KEY: MODULE.NO_FACE_REASON,
    }
    crops = torch.rand((3, 32, 48, 3))

    masks, mask_report = mapping["H3FaceMaskSAM"]().run(
        crops=crops,
        sam_model=object(),
        transform=transform,
        threshold=0.93,
        dilation=0,
        temporal_smooth=5,
    )
    assert tuple(masks.shape) == (3, 32, 48)
    assert torch.count_nonzero(masks) == 0
    assert "skipped automatically" in mask_report

    latent = {"samples": object()}
    returned, denoise_report = mapping["H3PerFrameDenoise"]().run(
        av_latent=latent,
        transform=transform,
        strength_small_face=1.0,
        strength_large_face=0.35,
        face_px_small=30.0,
        face_px_large=120.0,
        gamma=1.0,
        smooth_frames=9,
    )
    assert returned is latent
    assert "skipped automatically" in denoise_report

    info, = mapping["H3FaceTransformInfo"]().run(transform=transform, max_rows=12)
    assert "skipped automatically" in info


def test_unrelated_tracker_value_error_remains_hard_failure():
    class BrokenTrack:
        DESCRIPTION = "track"

        def run(self, *args, **kwargs):
            raise ValueError("detector output tensor is malformed")

    mapping = {"H3FaceTrackCrop": BrokenTrack}
    MODULE.install_no_face_passthrough(mapping)

    with pytest.raises(ValueError, match="malformed"):
        BrokenTrack().run(
            images=torch.rand((1, 16, 16, 3)),
            detector="dummy.pt",
            confidence=0.35,
            crop_factor=2.5,
            canvas_width=512,
            canvas_height=512,
            canvas_mode="manual",
            smooth_window=21,
            size_smooth_window=51,
            smooth_method="gaussian",
            size_mode="per_frame",
        )


def test_installer_is_idempotent():
    mapping = _mapping()
    MODULE.install_no_face_passthrough(mapping)
    first = mapping["H3FaceTrackCrop"].run
    MODULE.install_no_face_passthrough(mapping)
    assert mapping["H3FaceTrackCrop"].run is first
