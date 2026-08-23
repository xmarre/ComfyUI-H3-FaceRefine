from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).resolve().parents[1] / "quality_fixes_v2.py"
SPEC = importlib.util.spec_from_file_location("h3_face_quality_fix_v2_tests", MODULE_PATH)
assert SPEC and SPEC.loader
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


def _install_fake_comfy(monkeypatch):
    comfy = types.ModuleType("comfy")
    management = types.ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    management.throw_exception_if_processing_interrupted = lambda: None
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)


def _source_crop(base_images, boxes, crop_w, crop_h):
    grid = quality._crop_grid(
        boxes,
        0,
        len(boxes),
        width=int(base_images.shape[2]),
        height=int(base_images.shape[1]),
        crop_w=crop_w,
        crop_h=crop_h,
        device=torch.device("cpu"),
    )
    src = base_images[..., :3].movedim(-1, 1).float()
    return F.grid_sample(
        src, grid, mode="bilinear", padding_mode="border", align_corners=False
    ).movedim(1, -1)


def _run(Fixed, base, refined, boxes, *, feather=0, colour_match=0.0):
    crop_h, crop_w = int(refined.shape[1]), int(refined.shape[2])
    transform = {
        "boxes": boxes,
        "canvas": (crop_w, crop_h),
        "src_size": (int(base.shape[2]), int(base.shape[1])),
        "detected": [True] * int(base.shape[0]),
        "face_rect": [(0.0, 0.0, float(crop_w), float(crop_h))] * int(base.shape[0]),
    }
    return Fixed().run(
        base,
        refined,
        transform,
        "full_crop",
        0,
        feather,
        colour_match,
        1.0,
        "fade_out",
    )[0]


def test_residual_stitch_is_exact_noop_when_corrected_equals_exact_source_crop(monkeypatch):
    _install_fake_comfy(monkeypatch)

    class FakeStitch:
        pass

    Fixed = quality.build_residual_stitch(FakeStitch)
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    base = torch.stack(
        (xx.float() / 7.0, yy.float() / 7.0, (xx + yy).float() / 14.0), dim=-1
    ).unsqueeze(0)
    boxes = [(2.2, 1.7, 3.0, 3.0)]
    corrected = _source_crop(base, boxes, 4, 4)
    out = _run(Fixed, base, corrected, boxes)
    assert torch.allclose(out, base, atol=1e-7, rtol=0.0)


def test_downscaled_crop_still_transfers_learned_delta_without_replacing_source(monkeypatch):
    _install_fake_comfy(monkeypatch)

    class FakeStitch:
        pass

    Fixed = quality.build_residual_stitch(FakeStitch)
    base = torch.full((1, 8, 8, 3), 0.5)
    boxes = [(1.0, 1.0, 6.0, 6.0)]  # 4/6 < 1x: deliberately downscaled crop
    source = _source_crop(base, boxes, 4, 4)
    corrected = source + 0.10
    out = _run(Fixed, base, corrected, boxes, feather=1)
    # No magnification hard gate: the learned correction is retained. Because the
    # source baseline itself is never pasted, the full-resolution frame remains the base.
    assert float(out.max()) > 0.5
    assert float(out.min()) >= 0.5 - 1e-6


def test_positive_refinement_delta_cannot_create_a_dark_moving_box_edge(monkeypatch):
    _install_fake_comfy(monkeypatch)

    class FakeStitch:
        pass

    Fixed = quality.build_residual_stitch(FakeStitch)
    base = torch.full((2, 10, 10, 3), 0.5)
    boxes = [(1.2, 2.0, 4.0, 4.0), (2.3, 2.0, 4.0, 4.0)]
    source = _source_crop(base, boxes, 6, 6)
    corrected = source + 0.1
    out = _run(Fixed, base, corrected, boxes, feather=1)
    assert float(out.min()) >= 0.5 - 1e-6
    assert float(out.max()) > 0.5


def test_negative_refinement_delta_cannot_create_a_bright_moving_box_edge(monkeypatch):
    _install_fake_comfy(monkeypatch)

    class FakeStitch:
        pass

    Fixed = quality.build_residual_stitch(FakeStitch)
    base = torch.full((2, 10, 10, 3), 0.5)
    boxes = [(1.2, 2.0, 4.0, 4.0), (2.3, 2.0, 4.0, 4.0)]
    source = _source_crop(base, boxes, 6, 6)
    corrected = source - 0.1
    out = _run(Fixed, base, corrected, boxes, feather=1)
    assert float(out.max()) <= 0.5 + 1e-6
    assert float(out.min()) < 0.5


def test_residual_colour_match_removes_only_dc_bias():
    delta = torch.tensor(
        [[[[0.10, 0.20], [0.30, 0.40]], [[0.20, 0.30], [0.40, 0.50]], [[0.0, 0.10], [0.20, 0.30]]]],
        dtype=torch.float32,
    )
    mask = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    adjusted = quality._remove_residual_bias(delta, mask, 1.0)
    means = adjusted.mean(dim=(2, 3))
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-7)
    # Relative structure remains; there is no std/gain remapping of the patch.
    assert torch.allclose(
        adjusted[:, :, 1, 1] - adjusted[:, :, 0, 0],
        delta[:, :, 1, 1] - delta[:, :, 0, 0],
    )


def test_install_v2_replaces_stitch_only():
    class FakeStitch:
        pass

    untouched = object()
    mappings = {"H3FaceStitch": FakeStitch, "Other": untouched}
    quality.install_quality_fixes_v2(mappings)
    assert mappings["H3FaceStitch"] is not FakeStitch
    assert issubclass(mappings["H3FaceStitch"], FakeStitch)
    assert mappings["Other"] is untouched
