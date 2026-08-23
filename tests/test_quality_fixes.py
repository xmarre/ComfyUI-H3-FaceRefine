from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).resolve().parents[1] / "quality_fixes.py"
SPEC = importlib.util.spec_from_file_location("h3_face_quality_fix_tests", MODULE_PATH)
assert SPEC and SPEC.loader
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


def test_detection_gate_uses_only_real_face_detections():
    gate = quality.require_detection_gate({"detected": [True, False, False, True]}, 4)
    assert torch.equal(gate, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert quality.dropout_runs(gate) == [(1, 3)]


def test_detection_gate_fails_closed_when_tracker_contract_is_missing():
    with pytest.raises(ValueError, match="missing an aligned per-frame 'detected' mask"):
        quality.require_detection_gate({}, 3)
    with pytest.raises(ValueError, match="missing an aligned per-frame 'detected' mask"):
        quality.require_detection_gate({"detected": [True, False]}, 3)


def test_fade_out_never_prefades_valid_frames_before_dropout():
    detected = [True, True, True, False, False, True, True, True, True]
    weights = quality.safe_stitch_weights(detected, "fade_out", fade_frames=4)
    assert weights is not None
    assert weights[:3] == [1.0, 1.0, 1.0]
    assert weights[3:5] == [0.0, 0.0]
    assert weights[5:] == [0.25, 0.5, 0.75, 1.0]


def test_skip_is_exact_binary_detection_mask():
    detected = [True, False, True, False]
    assert quality.safe_stitch_weights(detected, "skip") == [1.0, 0.0, 1.0, 0.0]
    assert quality.safe_stitch_weights(detected, "composite_anyway") is None


def test_fractional_rect_mask_preserves_subpixel_edge_coverage():
    mask = quality.fractional_rect_mask(
        4, 6, (1.25, 1.0, 2.0, 2.0), 0.0, device=torch.device("cpu")
    )[0, 0]
    assert torch.allclose(mask[1, 1:4], torch.tensor([0.75, 1.0, 0.25]))
    assert torch.allclose(mask[2, 1:4], torch.tensor([0.75, 1.0, 0.25]))
    assert float(mask.sum()) == pytest.approx(4.0)


def test_fractional_rect_moves_smoothly_instead_of_one_pixel_hopping():
    first = quality.fractional_rect_mask(
        4, 6, (1.10, 1.0, 2.0, 2.0), 0.0, device=torch.device("cpu")
    )
    second = quality.fractional_rect_mask(
        4, 6, (1.20, 1.0, 2.0, 2.0), 0.0, device=torch.device("cpu")
    )
    delta = (second - first).abs()
    assert 0.0 < float(delta.max()) < 1.0
    assert float(delta.sum()) == pytest.approx(0.4, abs=1e-6)


def test_border_safe_patch_warp_cannot_create_black_fringe():
    boxes = [(1.25, 1.0, 2.5, 2.0)]
    grid = quality._warp_grid(boxes, 0, 1, width=6, height=4, device=torch.device("cpu"))
    white = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    zero_padded = F.grid_sample(
        white, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    border_padded = F.grid_sample(
        white, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    assert float(zero_padded.min()) < 1.0
    assert torch.allclose(border_padded, torch.ones_like(border_padded))


def test_source_space_feather_blurs_at_requested_pixel_scale():
    mask = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    mask[:, :, 3:6, 3:6] = 1.0
    blurred = quality._gaussian_blur(mask, 2)
    assert 0.0 < float(blurred[0, 0, 2, 4]) < 1.0
    assert float(blurred[0, 0, 4, 4]) > float(blurred[0, 0, 2, 4])


def test_partial_sigma_error_no_longer_recommends_four_steps():
    with pytest.raises(ValueError) as exc:
        quality.validate_partial_sigmas_current(torch.tensor([1.0, 0.0]))
    text = str(exc.value)
    assert "12 steps" in text
    assert "4 steps" not in text


def test_paired_decode_preserves_clean_and_refined_batches():
    class FakeVAE:
        def decode(self, latent):
            assert latent.shape[0] == 2
            out = torch.empty((2, 5, 2, 2, 3), dtype=torch.float32)
            out[0].fill_(0.25)
            out[1].fill_(0.35)
            return out

    clean = torch.zeros((1, 24, 2, 1, 1))
    refined = torch.ones_like(clean)
    a, b = quality._decode_pair_strict(
        FakeVAE(), clean, refined, total_frames=5, crop_height=2, crop_width=2
    )
    assert torch.allclose(a, torch.full_like(a, 0.25))
    assert torch.allclose(b, torch.full_like(b, 0.35))


def test_roundtrip_cancelled_group_outputs_only_sampler_delta_over_source_crop():
    module = types.SimpleNamespace()
    module.H3_VIDEO_CHANNELS = 24
    module.H3_AUDIO_CHANNELS = 32
    module._split_samples = lambda latent, **_kwargs: latent["samples"]
    module.encode_video_strict = lambda _vae, _crops, _t: torch.zeros((1, 24, 2, 1, 1))
    module.inherited_temporal_mask = lambda _mask, _source_t, target_t: torch.ones(target_t)
    module._mask_member = lambda _latent: None
    module.carry_previous_refined_prefix = lambda *_args, **_kwargs: 0
    module.compose_video_mask = lambda _inherited, _strengths, video: torch.ones(
        (1, 1, video.shape[2], video.shape[3], video.shape[4])
    )
    module._nested = lambda members: tuple(members)
    module.validate_refine_state = lambda _state: (object(), [])
    module.adapt_conditioning_to_crop = lambda positive, *_args: positive
    module.model_for_refinement = lambda model: model

    def sample_locked_audio(clean, **_kwargs):
        video, audio = clean["samples"]
        return video + 1.0, audio

    module.sample_locked_audio = sample_locked_audio

    class FakeVAE:
        def decode(self, latent):
            values = latent.mean(dim=(1, 2, 3, 4))
            out = torch.empty((latent.shape[0], 5, 2, 2, 3), dtype=torch.float32)
            for i, value in enumerate(values):
                out[i].fill_(0.25 + 0.10 * float(value))
            return out

    fn = quality.build_roundtrip_cancelled_physical_group(module)
    crops = torch.full((5, 2, 2, 3), 0.4)
    video_latent = {"samples": torch.zeros((1, 24, 2, 1, 1))}
    audio_latent = {"samples": torch.zeros((1, 32, 2, 3))}
    segment, _video, _context, _carried, report = fn(
        position=0,
        group={
            "expected_video_latent_t": 2,
            "expected_audio_latent_t": 3,
            "total_frames": 5,
            "trim_frames": 0,
            "net_frames": 5,
            "logical_chunk_indices": [1],
            "terminal_merged": False,
        },
        crops=crops,
        strengths=torch.ones(5),
        video_latent=video_latent,
        audio_latent=audio_latent,
        refine_state={},
        vae=FakeVAE(),
        noise=object(),
        sampler=object(),
        sigmas=torch.tensor([0.2, 0.0]),
        previous_refined_video=None,
    )
    assert torch.allclose(segment, torch.full_like(crops, 0.5), atol=1e-6)
    assert "vae_roundtrip_cancelled=True" in report


def test_install_quality_fixes_patches_continuum_group_without_touching_stitch():
    fake_module = types.ModuleType("fake_face_continuum")

    class FakeContinuum:
        pass

    FakeContinuum.__module__ = fake_module.__name__
    fake_module.FakeContinuum = FakeContinuum
    fake_module.refine_physical_group = lambda **_kwargs: None
    fake_module.validate_partial_sigmas = lambda value, **_kwargs: value
    sys.modules[fake_module.__name__] = fake_module

    class FakeStitch:
        pass

    mappings = {
        "H3FaceStitch": FakeStitch,
        "H3ContinuumFaceRefine": FakeContinuum,
    }
    quality.install_quality_fixes(mappings)
    assert mappings["H3FaceStitch"] is FakeStitch
    assert mappings["H3ContinuumFaceRefine"] is not FakeContinuum
    assert issubclass(mappings["H3ContinuumFaceRefine"], FakeContinuum)
    assert getattr(fake_module.refine_physical_group, "_h3_roundtrip_cancelled", False)
    assert fake_module.validate_partial_sigmas is quality.validate_partial_sigmas_current


def test_fixed_continuum_node_zeroes_interpolated_frames_before_physical_sampling():
    module = types.ModuleType("fake_face_continuum_runtime")
    captured = {}

    class FakeContinuum:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"sigmas": ("SIGMAS", {})}}

    FakeContinuum.__module__ = module.__name__
    module.FakeContinuum = FakeContinuum
    module._single = lambda value, _name: value
    module._group_list = lambda value, _name: list(value)
    module._value_at = lambda value, index: value[index] if isinstance(value, list) else value
    module.validate_physical_groups = lambda _plan: ([{"logical_chunk_indices": [1]}], 4)
    module.face_strength_timeline = lambda *_args, **_kwargs: torch.ones(4)
    module.partition_global_timeline = lambda values, _groups: [values]

    def fake_refine_physical_group(**kwargs):
        captured["strengths"] = kwargs["strengths"].clone()
        return kwargs["crops"].clone(), torch.zeros(1), None, 0, "group 1"

    module.refine_physical_group = fake_refine_physical_group
    sys.modules[module.__name__] = module

    Fixed = quality.build_fixed_continuum_refine(FakeContinuum)
    crops = torch.zeros((4, 2, 2, 3), dtype=torch.float32)
    transform = {
        "frames": 4,
        "boxes": [(0.0, 0.0, 2.0, 2.0)] * 4,
        "detected": [True, False, False, True],
    }
    output, report = Fixed().refine(
        crops,
        transform,
        [object()],
        [object()],
        object(),
        [object()],
        object(),
        object(),
        object(),
        torch.tensor([0.2, 0.0]),
        1.0,
        0.35,
        "absolute_px",
        30.0,
        120.0,
        1.0,
        1,
    )

    assert torch.equal(captured["strengths"], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.equal(output, crops)
    assert "2 interpolated/no-face frames forced to denoise mask 0" in report
    assert "protected dropout runs: 1-2 (2f)" in report
    tooltip = Fixed.INPUT_TYPES()["required"]["sigmas"][1]["tooltip"]
    assert "12 steps" in tooltip
    assert "Four-step refinement was visually rejected" in tooltip
