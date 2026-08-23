from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "continuum_refine.py"
SPEC = importlib.util.spec_from_file_location("h3_face_continuum_refine_tests", MODULE_PATH)
assert SPEC and SPEC.loader
refine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = refine
SPEC.loader.exec_module(refine)


def _chunk(index, total, trim, video_t=2, audio_t=8):
    return {
        "sequence_index": index,
        "chunk_index": index,
        "total_frames": total,
        "trim_frames": trim,
        "net_frames": total - trim,
        "context_frames": trim,
        "expected_video_latent_t": video_t,
        "expected_audio_latent_t": audio_t,
    }


def _plan(chunks, groups=None, *, target_frames=None):
    natural = sum(int(item["net_frames"]) for item in chunks)
    plan = {
        "magic": refine.ASSEMBLY_MAGIC,
        "schema_version": 1,
        "fps": 24,
        "width": 64,
        "height": 64,
        "chunk_seconds": 5.0,
        "target_frames": natural if target_frames is None else target_frames,
        "natural_frames": natural,
        "chunks": chunks,
    }
    if groups is not None:
        plan.update(
            {
                "decode_group_version": 1,
                "logical_chunk_count": len(chunks),
                "physical_decode_group_count": len(groups),
                "decode_groups": groups,
            }
        )
    return plan


def _normal_plan():
    first = _chunk(1, 5, 0, video_t=2, audio_t=8)
    second = _chunk(2, 22, 5, video_t=7, audio_t=10)
    return _plan([first, second])


def test_non_terminal_plan_uses_one_physical_group_per_logical_chunk():
    groups, frames = refine.validate_physical_groups(_normal_plan())
    assert frames == 22
    assert [item["logical_chunk_indices"] for item in groups] == [[1], [2]]
    assert [item["frame_start"] for item in groups] == [0, 5]


def test_two_by_five_terminal_merge_is_one_physical_group():
    chunks = [_chunk(1, 124, 0, 37, 200), _chunk(2, 141, 22, 42, 220)]
    group = {
        **chunks[0],
        "total_frames": 243,
        "trim_frames": 0,
        "net_frames": 243,
        "context_frames": 0,
        "expected_video_latent_t": 72,
        "expected_audio_latent_t": 405,
        "frame_start": 0,
        "frame_stop": 243,
        "logical_chunk_indices": [1, 2],
        "terminal_merged": True,
    }
    plan = _plan(chunks, [group])
    plan["natural_frames"] = 243
    groups, frames = refine.validate_physical_groups(plan)
    assert frames == 243
    assert groups[0]["logical_chunk_indices"] == [1, 2]
    assert groups[0]["terminal_merged"] is True


def test_three_chunk_terminal_tail_uses_two_physical_groups():
    chunks = [_chunk(1, 124, 0, 37, 200), _chunk(2, 141, 22, 42, 220), _chunk(3, 141, 22, 42, 220)]
    first = {**chunks[0], "frame_start": 0, "frame_stop": 124, "logical_chunk_indices": [1], "terminal_merged": False}
    tail = {
        **chunks[1],
        "total_frames": 260,
        "trim_frames": 22,
        "net_frames": 238,
        "context_frames": 22,
        "expected_video_latent_t": 77,
        "expected_audio_latent_t": 433,
        "frame_start": 124,
        "frame_stop": 362,
        "logical_chunk_indices": [2, 3],
        "terminal_merged": True,
    }
    plan = _plan(chunks, [first, tail])
    groups, frames = refine.validate_physical_groups(plan)
    assert frames == 362
    assert [g["logical_chunk_indices"] for g in groups] == [[1], [2, 3]]


def test_unknown_decode_group_version_fails_closed():
    plan = _normal_plan()
    plan.update({"decode_groups": [plan["chunks"][0]], "decode_group_version": 99})
    with pytest.raises(ValueError, match="decode_group_version"):
        refine.validate_physical_groups(plan)


def test_physical_groups_must_cover_logical_chunks_exactly_once():
    plan = _normal_plan()
    groups, _ = refine.validate_physical_groups(plan)
    groups[1]["logical_chunk_indices"] = [1]
    bad = _plan(plan["chunks"], groups)
    with pytest.raises(ValueError, match="cover logical chunks"):
        refine.validate_physical_groups(bad)


def test_partition_reconstructs_preceding_context_from_global_timeline():
    groups, _ = refine.validate_physical_groups(_normal_plan())
    timeline = torch.arange(22)
    physical = refine.partition_global_timeline(timeline, groups)
    assert physical[0].tolist() == list(range(5))
    assert physical[1].tolist() == list(range(22))


def test_partition_rejects_missing_natural_tail_instead_of_padding():
    groups, _ = refine.validate_physical_groups(_normal_plan())
    with pytest.raises(ValueError, match="reconstructed 21 crop frames"):
        refine.partition_global_timeline(torch.arange(21), groups)


def _transform(frames, heights=None):
    heights = heights or [30.0 + i for i in range(frames)]
    return {
        "frames": frames,
        "crop_factor": 2.0,
        "boxes": [(0.0, 0.0, 64.0, float(height) * 2.0) for height in heights],
    }


def test_face_strength_absolute_thresholds():
    strength = refine.face_strength_timeline(
        _transform(3, [30, 75, 120]),
        strength_small_face=1.0,
        strength_large_face=0.4,
        scale_mode="absolute_px",
        face_px_small=30,
        face_px_large=120,
        gamma=1.0,
        smooth_frames=1,
    )
    assert torch.allclose(strength, torch.tensor([1.0, 0.7, 0.4]))


def test_face_strength_relative_mode_uses_clip_extremes():
    strength = refine.face_strength_timeline(
        _transform(2, [80, 100]),
        strength_small_face=1.0,
        strength_large_face=0.25,
        scale_mode="relative_to_clip",
        face_px_small=1,
        face_px_large=2,
        gamma=1.0,
        smooth_frames=1,
    )
    assert torch.allclose(strength, torch.tensor([1.0, 0.25]))


def test_inherited_native_mask_requires_spatial_uniformity():
    mask = torch.ones((1, 1, 2, 2, 2))
    mask[..., 0, 0] = 0
    with pytest.raises(ValueError, match="spatially non-uniform"):
        refine.inherited_temporal_mask(mask, 2, 2)


def test_inherited_mask_nearest_maps_temporal_grid():
    mask = torch.tensor([0.0, 1.0]).view(1, 1, 2, 1, 1)
    values = refine.inherited_temporal_mask(mask, 2, 4)
    assert values.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_mask_multiplication_keeps_protected_zero():
    video = torch.zeros((1, 24, 3, 2, 2))
    composed = refine.compose_video_mask(
        torch.tensor([0.0, 1.0, 1.0]), torch.tensor([1.0, 0.5, 0.25]), video
    )
    assert torch.all(composed[:, :, 0] == 0)
    assert torch.allclose(composed[0, 0, :, 0, 0], torch.tensor([0.0, 0.5, 0.25]))


def test_carry_uses_only_contiguous_inherited_zero_prefix():
    current = torch.zeros((1, 24, 4, 2, 2))
    previous = torch.zeros_like(current)
    previous[:, :, -2:] = 9
    steps = refine.carry_previous_refined_prefix(
        current, previous, torch.tensor([0.0, 0.0, 1.0, 0.0])
    )
    assert steps == 2
    assert torch.all(current[:, :, :2] == 9)
    assert torch.all(current[:, :, 2:] == 0)


def test_carry_rejects_changed_crop_latent_geometry():
    with pytest.raises(ValueError, match="changed latent geometry"):
        refine.carry_previous_refined_prefix(
            torch.zeros((1, 24, 2, 2, 2)),
            torch.zeros((1, 24, 2, 3, 2)),
            torch.tensor([0.0, 1.0]),
        )


def test_unknown_refine_state_api_fails_closed():
    with pytest.raises(ValueError, match="Unsupported H3 Continuum refine_state API"):
        refine.validate_refine_state({"api": 999, "model": object(), "positive": []})


class FakeSampling:
    sigma_max = torch.tensor(1.0)


class FakeModel:
    def __init__(self, options=None):
        self.model_options = copy.deepcopy(options or {})
        self.sampling = FakeSampling()

    def clone(self):
        return FakeModel(self.model_options)

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self.sampling


def test_refinement_model_preserves_patch_stack_and_marks_sampler_two():
    source = FakeModel(
        {"transformer_options": {"h3_continuum": {"api": 1}, "spectrum": {"active": True}}}
    )
    marked = refine.model_for_refinement(source)
    assert marked is not source
    assert refine.H3_REFINEMENT_KEY not in source.model_options["transformer_options"]
    opts = marked.model_options["transformer_options"]
    assert opts["h3_continuum"] == {"api": 1}
    assert opts["spectrum"] == {"active": True}
    assert opts[refine.H3_REFINEMENT_KEY]["sigma_reference"] == 1.0


def test_refinement_model_accepts_missing_transformer_options():
    marked = refine.model_for_refinement(FakeModel({"transformer_options": None}))
    assert marked.model_options["transformer_options"][refine.H3_REFINEMENT_KEY]["api"] == 1


def test_refinement_model_requires_scalar_sigma_reference():
    class VectorSigmaModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.sampling.sigma_max = torch.tensor([1.0, 2.0])

        def clone(self):
            return VectorSigmaModel()

    source = VectorSigmaModel()
    with pytest.raises(ValueError, match=r"scalar model_sampling\.sigma_max"):
        refine.model_for_refinement(source)


def test_conditioning_resizes_target_keyframes_and_preserves_references():
    refs = [{"latent": torch.ones((1, 24, 1, 3, 5))}]
    keyframe = {"latent": torch.ones((1, 24, 1, 3, 5)), "latent_h": 3, "latent_w": 5}
    positive = [[torch.ones(1), {"minimax_refs": refs, "minimax_keyframes": [keyframe]}]]
    result = refine.resize_target_conditioning(positive, 4, 6)
    meta = result[0][1]
    assert meta["minimax_refs"] is refs
    assert tuple(meta["minimax_keyframes"][0]["latent"].shape[-2:]) == (4, 6)
    assert tuple(keyframe["latent"].shape[-2:]) == (3, 5)


class KeyframeVAE:
    def encode(self, images):
        value = float(images[0, 0, 0, 0])
        return torch.full((1, 24, 1, 2, 2), value)


def test_crop_conditioning_rebinds_visual_keyframes_and_preserves_hybrid_refs():
    refs = [
        {"role": "reference_image", "latent": torch.ones((1, 24, 1, 3, 5))},
        {"role": "video_context", "latent": torch.ones((1, 24, 2, 3, 5))},
    ]
    audio_keyframe = {"resolved_frame_index": 0, "audio_latent": torch.ones((1, 32, 2, 3))}
    positive = [[
        torch.ones(1),
        {
            "minimax_refs": refs,
            "minimax_keyframes": [
                {"resolved_frame_index": 4, "latent": torch.zeros((1, 24, 1, 3, 5))},
                audio_keyframe,
            ],
            "minimax_frame_count": 5,
        },
    ]]
    crops = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 8, 8, 3)
    result = refine.adapt_conditioning_to_crop(positive, KeyframeVAE(), crops, 4, 6)
    meta = result[0][1]
    visual = meta["minimax_keyframes"][0]["latent"]
    assert tuple(visual.shape) == (1, 24, 1, 4, 6)
    assert torch.all(visual == 4)
    assert meta["minimax_keyframes"][1]["audio_latent"] is audio_keyframe["audio_latent"]
    assert meta["minimax_refs"] is refs


class ShapeVAE:
    def encode(self, images):
        return torch.zeros((1, 24, 3, 2, 2))


def test_strict_encode_rejects_temporal_mismatch_without_trim_or_pad():
    with pytest.raises(ValueError, match="No temporal trim/pad"):
        refine.encode_video_strict(ShapeVAE(), torch.zeros((5, 8, 8, 3)), expected_t=2)


class NonTensorVAE:
    def encode(self, _images):
        return object()


def test_strict_encode_rejects_non_tensor_with_topology_error():
    with pytest.raises(ValueError, match=r"\[1,24,T,H,W\]"):
        refine.encode_video_strict(NonTensorVAE(), torch.zeros((5, 8, 8, 3)), expected_t=2)


def test_crop_keyframe_encode_rejects_non_tensor_with_topology_error():
    positive = [[
        torch.ones(1),
        {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": torch.zeros((1, 24, 1, 2, 2))}]},
    ]]
    with pytest.raises(ValueError, match=r"\[1,24,T,H,W\]"):
        refine.adapt_conditioning_to_crop(
            positive,
            NonTensorVAE(),
            torch.zeros((5, 8, 8, 3)),
            3,
            5,
        )


class FakeNested:
    is_nested = True

    def __init__(self, members):
        self.members = list(members)

    def unbind(self):
        return tuple(self.members)

    def to(self, device):
        return FakeNested([member.to(device) for member in self.members])


class FakeNoise:
    seed = 123

    def generate_noise(self, latent):
        video, audio = latent["samples"].unbind()
        return FakeNested([torch.ones_like(video), torch.ones_like(audio)])


class FakeGuider:
    calls = []

    def __init__(self, model):
        self.model_patcher = model
        self.conds = None

    def inner_set_conds(self, conds):
        self.conds = conds

    def sample(self, noise, latent, _sampler, _sigmas, **kwargs):
        video, audio = latent.unbind()
        noise_video, noise_audio = noise.unbind()
        FakeGuider.calls.append(
            {"video_before": video.clone(), "audio_before": audio.clone(), "mask": kwargs["denoise_mask"]}
        )
        return FakeNested([video + noise_video, audio + noise_audio])


@pytest.fixture(autouse=True)
def _reset_fake_guider_calls():
    FakeGuider.calls.clear()
    yield
    FakeGuider.calls.clear()


def _install_fake_comfy(monkeypatch):
    comfy = types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = FakeNested
    samplers = types.ModuleType("comfy.samplers")
    samplers.CFGGuider = FakeGuider
    sample = types.ModuleType("comfy.sample")
    sample.fix_empty_latent_channels = lambda _model, latent, *_ratios: latent
    management = types.ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    utils = types.ModuleType("comfy.utils")
    utils.PROGRESS_BAR_ENABLED = False
    comfy.nested_tensor = nested
    comfy.samplers = samplers
    comfy.sample = sample
    comfy.model_management = management
    comfy.utils = utils
    preview = types.ModuleType("latent_preview")
    preview.prepare_callback = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "latent_preview", preview)


def test_sampler_two_locks_and_restores_audio(monkeypatch):
    _install_fake_comfy(monkeypatch)
    clean_audio = torch.full((1, 32, 2, 8), 7.0)
    clean = {
        "samples": FakeNested([torch.zeros((1, 24, 2, 2, 2)), clean_audio.clone()]),
        "noise_mask": FakeNested(
            [torch.ones((1, 1, 2, 2, 2)), torch.zeros((1, 1, 2, 8))]
        ),
    }
    video, audio = refine.sample_locked_audio(
        clean,
        model=FakeModel(),
        positive=[[torch.ones(1), {}]],
        noise=FakeNoise(),
        sampler=object(),
        sigmas=torch.tensor([0.2, 0.0]),
    )
    assert torch.all(video == 1)
    assert torch.equal(audio, clean_audio)
    _, sent_audio_noise = FakeGuider.calls[-1]["mask"].unbind()
    assert torch.all(sent_audio_noise == 0)


def test_sampler_two_rejects_reused_full_pass_one_schedule_with_actionable_error(monkeypatch):
    _install_fake_comfy(monkeypatch)
    clean = {
        "samples": FakeNested(
            [torch.zeros((1, 24, 2, 2, 2)), torch.zeros((1, 32, 2, 8))]
        ),
        "noise_mask": FakeNested(
            [torch.ones((1, 1, 2, 2, 2)), torch.zeros((1, 1, 2, 8))]
        ),
    }
    with pytest.raises(
        ValueError,
        match=r"separate BasicScheduler.*12 steps.*denoise 0\.45.*Continuum sampler-1 SIGMAS",
    ):
        refine.sample_locked_audio(
            clean,
            model=FakeModel(),
            positive=[[torch.ones(1), {}]],
            noise=FakeNoise(),
            sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
        )


class FakeVideoVAE:
    def __init__(self):
        self.current = None

    def encode(self, images):
        self.current = images.clone()
        t = {5: 2, 22: 7}[int(images.shape[0])]
        value = float(images[0, 0, 0, 0])
        return torch.full((1, 24, t, 2, 2), value)

    def decode(self, _video):
        return (self.current + 0.1).unsqueeze(0)


def test_strict_decode_normalizes_core_video_vae_batch_axis():
    vae = FakeVideoVAE()
    vae.current = torch.zeros((5, 8, 8, 3))
    decoded = refine.decode_video_strict(
        vae,
        torch.zeros((1, 24, 2, 2, 2)),
        total_frames=5,
        crop_height=8,
        crop_width=8,
    )
    assert tuple(decoded.shape) == (5, 8, 8, 3)


def test_strict_decode_rejects_multiple_video_batches():
    class MultiBatchVAE:
        def decode(self, _video):
            return torch.zeros((2, 5, 8, 8, 3))

    with pytest.raises(ValueError, match="multiple batches"):
        refine.decode_video_strict(
            MultiBatchVAE(),
            torch.zeros((1, 24, 2, 2, 2)),
            total_frames=10,
            crop_height=8,
            crop_width=8,
        )


def _video_latent(t, mask=None):
    value = {"samples": torch.zeros((1, 24, t, 4, 4))}
    if mask is not None:
        value["noise_mask"] = mask.view(1, 1, t, 1, 1).expand(1, 1, t, 4, 4).clone()
    return value


def _audio_latent(t, value):
    return {"samples": torch.full((1, 32, 2, t), float(value))}


def test_complete_node_refines_physical_groups_and_reassembles_global_order(monkeypatch):
    _install_fake_comfy(monkeypatch)
    plan = _normal_plan()
    crops = torch.arange(22, dtype=torch.float32).view(22, 1, 1, 1).expand(22, 8, 8, 3) / 100
    states = [
        {"api": 1, "model": FakeModel({"transformer_options": {"spectrum": {"active": True}}}), "positive": [[torch.ones(1), {}]]},
        {"api": 1, "model": FakeModel(), "positive": [[torch.ones(1), {}]]},
    ]
    output, report = refine.H3ContinuumFaceRefine().refine(
        crops,
        _transform(22),
        [_video_latent(2), _video_latent(7, torch.tensor([0, 0, 1, 1, 1, 1, 1]))],
        [_audio_latent(8, 5), _audio_latent(10, 7)],
        plan,
        states,
        FakeVideoVAE(),
        FakeNoise(),
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
    assert output.shape == crops.shape
    assert torch.allclose(output, crops + 0.1)
    assert len(FakeGuider.calls) == 2
    assert torch.all(FakeGuider.calls[1]["video_before"][:, :, :2] == 1)
    assert "logical=[1]" in report and "logical=[2]" in report
    assert "audio: locked" in report


def test_complete_node_rejects_run_storage_style_state_misalignment(monkeypatch):
    _install_fake_comfy(monkeypatch)
    with pytest.raises(ValueError, match="Regenerate from Chunk 1"):
        refine.H3ContinuumFaceRefine().refine(
            torch.zeros((22, 8, 8, 3)),
            _transform(22),
            [_video_latent(2), _video_latent(7)],
            [_audio_latent(8, 1), _audio_latent(10, 1)],
            _normal_plan(),
            [{"api": 1, "model": FakeModel(), "positive": []}],
            FakeVideoVAE(),
            FakeNoise(),
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


def test_complete_node_rejects_already_exact_duration_trimmed_timeline(monkeypatch):
    _install_fake_comfy(monkeypatch)
    plan = _normal_plan()
    plan["target_frames"] = 20
    with pytest.raises(ValueError, match="H3 Continuum Finalize Duration V3.4"):
        refine.H3ContinuumFaceRefine().refine(
            torch.zeros((20, 8, 8, 3)),
            _transform(20),
            [_video_latent(2), _video_latent(7)],
            [_audio_latent(8, 1), _audio_latent(10, 1)],
            plan,
            [{"api": 1, "model": FakeModel(), "positive": []}] * 2,
            FakeVideoVAE(),
            FakeNoise(),
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


def _load_sibling_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH.parent / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_node_import_loads_no_sam_or_continuum_implementation_dependency():
    _load_sibling_module("nodes.py", "h3_face_refine_nodes_dependency_test")
    assert "segment_anything" not in sys.modules
    assert not any(name.startswith("ComfyUI_H3_Continuum") for name in sys.modules)


def test_existing_standalone_node_registrations_remain_present():
    face_nodes = _load_sibling_module("nodes.py", "h3_face_refine_nodes_mapping_test")
    for key in (
        "H3FaceTrackCrop",
        "H3FaceStitch",
        "H3InjectVideoLatent",
        "H3PerFrameDenoise",
        "H3FaceMaskSAM",
        "H3FaceTransformInfo",
    ):
        assert key in face_nodes.NODE_CLASS_MAPPINGS


def test_standalone_package_import_cannot_resolve_comfy_core_nodes(monkeypatch):
    core_nodes = types.ModuleType("nodes")
    core_nodes.NODE_CLASS_MAPPINGS = {"ComfyCoreOnly": object()}
    core_nodes.NODE_DISPLAY_NAME_MAPPINGS = {"ComfyCoreOnly": "Comfy Core"}
    monkeypatch.setitem(sys.modules, "nodes", core_nodes)
    package = _load_sibling_module("__init__.py", "h3_face_refine_package_collision_test")
    assert "H3ContinuumFaceRefine" in package.NODE_CLASS_MAPPINGS
    assert "ComfyCoreOnly" not in package.NODE_CLASS_MAPPINGS
