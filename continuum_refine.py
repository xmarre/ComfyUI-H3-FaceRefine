"""Exact H3 Continuum sampler-2 integration for globally tracked face crops.

The module deliberately depends only on Continuum's public dictionary/list
contracts. H3 Continuum is not imported, so standalone FaceRefine workflows
remain loadable when Continuum is absent.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


ASSEMBLY_MAGIC = "H3_CONTINUUM_ASSEMBLY_PLAN"
ASSEMBLY_SCHEMA = 1
DECODE_GROUP_VERSION = 1
REFINE_STATE_API = 1
H3_REFINEMENT_API = 1
H3_REFINEMENT_KEY = "h3_refinement"
H3_VIDEO_CHANNELS = 24
H3_AUDIO_CHANNELS = 32


class _FallbackNested:
    """NestedTensor-like test fallback used only outside ComfyUI."""

    is_nested = True

    def __init__(self, members):
        self._members = tuple(members)

    def unbind(self):
        return self._members

    def to(self, device):
        return _FallbackNested(member.to(device) for member in self._members)


def _nested(members):
    try:
        import comfy.nested_tensor
    except ImportError:
        return _FallbackNested(members)
    return comfy.nested_tensor.NestedTensor(tuple(members))


def _is_nested(value: Any) -> bool:
    return bool(getattr(value, "is_nested", False)) and callable(
        getattr(value, "unbind", None)
    )


def _single(value: Any, name: str) -> Any:
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one value")
        value = value[0]
    return value


def _group_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        return [value]
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not value:
        raise ValueError(f"{name} must not be empty")
    return list(value)


def _value_at(value: Any, index: int) -> Any:
    if not isinstance(value, list):
        return value
    if not value:
        return None
    return value[index if index < len(value) else -1]


def _strict_int(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_physical_groups(plan: Any) -> tuple[list[dict[str, Any]], int]:
    """Validate and return the physical H3 sample/decode groups in timeline order."""

    plan = _single(plan, "assembly_plan")
    if not isinstance(plan, Mapping) or plan.get("magic") != ASSEMBLY_MAGIC:
        raise ValueError("invalid H3 Continuum assembly_plan")
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != ASSEMBLY_SCHEMA:
        raise ValueError(
            f"unsupported H3 Continuum assembly_plan schema {plan.get('schema_version')!r}"
        )
    if int(plan.get("fps", 0)) != 24:
        raise ValueError("H3 Continuum assembly_plan FPS must be 24")

    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("assembly_plan.chunks must be a non-empty list")

    if "decode_groups" in plan:
        if int(plan.get("decode_group_version", -1)) != DECODE_GROUP_VERSION:
            raise ValueError(
                f"unsupported assembly_plan.decode_group_version {plan.get('decode_group_version')!r}"
            )
        raw_groups = plan.get("decode_groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("assembly_plan.decode_groups must be a non-empty list")
        if _strict_int(
            "assembly_plan.physical_decode_group_count",
            plan.get("physical_decode_group_count"),
            1,
        ) != len(raw_groups):
            raise ValueError("physical_decode_group_count does not match decode_groups")
        if _strict_int(
            "assembly_plan.logical_chunk_count", plan.get("logical_chunk_count"), 1
        ) != len(chunks):
            raise ValueError("logical_chunk_count does not match chunks")
    else:
        raw_groups = chunks

    groups: list[dict[str, Any]] = []
    cursor = 0
    covered: list[int] = []
    for position, raw in enumerate(raw_groups, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"assembly_plan physical group {position} must be a mapping")
        group = dict(raw)
        total = _strict_int(f"physical group {position}.total_frames", group.get("total_frames"), 5)
        trim = _strict_int(f"physical group {position}.trim_frames", group.get("trim_frames"), 0)
        net = _strict_int(f"physical group {position}.net_frames", group.get("net_frames"), 1)
        context = _strict_int(
            f"physical group {position}.context_frames", group.get("context_frames"), 0
        )
        expected_video_t = _strict_int(
            f"physical group {position}.expected_video_latent_t",
            group.get("expected_video_latent_t"),
            1,
        )
        expected_audio_t = _strict_int(
            f"physical group {position}.expected_audio_latent_t",
            group.get("expected_audio_latent_t"),
            1,
        )
        if (total - 5) % 17:
            raise ValueError(
                f"physical group {position}.total_frames={total} is not on H3's 17k+5 grid"
            )
        if total - trim != net:
            raise ValueError(f"physical group {position}: total_frames - trim_frames != net_frames")
        if trim and context != trim:
            raise ValueError(f"physical group {position}: context_frames must equal trim_frames")

        frame_start = int(group.get("frame_start", cursor))
        frame_stop = int(group.get("frame_stop", frame_start + net))
        if frame_start != cursor or frame_stop != cursor + net:
            raise ValueError(f"physical group {position}: natural frame boundary is discontinuous")
        if frame_start < trim:
            raise ValueError(
                f"physical group {position}: retained timeline has only {frame_start} preceding "
                f"frames for its {trim}-frame context"
            )

        logical = group.get("logical_chunk_indices")
        if logical is None:
            logical = [group.get("chunk_index")]
        if not isinstance(logical, list) or not logical:
            raise ValueError(f"physical group {position}.logical_chunk_indices is invalid")
        logical = [
            _strict_int(f"physical group {position}.logical_chunk_indices", item, 1)
            for item in logical
        ]
        covered.extend(logical)
        group.update(
            {
                "total_frames": total,
                "trim_frames": trim,
                "net_frames": net,
                "context_frames": context,
                "expected_video_latent_t": expected_video_t,
                "expected_audio_latent_t": expected_audio_t,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "logical_chunk_indices": logical,
                "terminal_merged": bool(group.get("terminal_merged", False)),
            }
        )
        groups.append(group)
        cursor = frame_stop

    expected_logical = [int(chunk.get("chunk_index", i)) for i, chunk in enumerate(chunks, 1)]
    if covered != expected_logical:
        raise ValueError("physical decode groups do not cover logical chunks exactly once")
    if "natural_frames" in plan and int(plan["natural_frames"]) != cursor:
        raise ValueError("assembly_plan.natural_frames does not match physical groups")
    return groups, cursor


def partition_global_timeline(
    values: torch.Tensor, groups: list[Mapping[str, Any]]
) -> list[torch.Tensor]:
    """Reconstruct every full physical sample from one globally tracked net timeline."""

    if not torch.is_tensor(values) or values.ndim < 1:
        raise TypeError("global crop timeline must be a torch.Tensor with a frame axis")
    result = []
    for position, group in enumerate(groups, start=1):
        start = int(group["frame_start"])
        stop = int(group["frame_stop"])
        trim = int(group["trim_frames"])
        physical = values[start - trim : stop]
        if int(physical.shape[0]) != int(group["total_frames"]):
            raise ValueError(
                f"physical group {position} reconstructed {int(physical.shape[0])} crop frames; "
                f"expected {int(group['total_frames'])}"
            )
        result.append(physical)
    return result


def _smooth_1d(values: torch.Tensor, window: int) -> torch.Tensor:
    window = min(int(window), int(values.numel()))
    if window < 3 or values.numel() < 3:
        return values
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    pad = window // 2
    x = torch.arange(window, dtype=torch.float64) - pad
    sigma = max(window / 6.0, 0.5)
    kernel = torch.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    mode = "reflect" if values.numel() > pad else "replicate"
    work = F.pad(values.view(1, 1, -1), (pad, pad), mode=mode)
    return F.conv1d(work, kernel.view(1, 1, -1)).view(-1)


def face_strength_timeline(
    transform: Mapping[str, Any],
    *,
    strength_small_face: float,
    strength_large_face: float,
    scale_mode: str,
    face_px_small: float,
    face_px_large: float,
    gamma: float,
    smooth_frames: int,
) -> torch.Tensor:
    """Build the globally smoothed FaceRefine denoise multiplier timeline."""

    if not isinstance(transform, Mapping):
        raise TypeError("transform must be an H3FACEXFORM mapping")
    boxes = transform.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("transform has no crop boxes")
    crop_factor = float(transform.get("crop_factor", 3.0))
    if not math.isfinite(crop_factor) or crop_factor <= 0:
        raise ValueError("transform.crop_factor must be finite and positive")
    try:
        face = torch.tensor([float(box[3]) / crop_factor for box in boxes], dtype=torch.float64)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("transform crop boxes are malformed") from exc
    if not bool(torch.isfinite(face).all()):
        raise ValueError("transform face sizes must be finite")

    if str(scale_mode) == "relative_to_clip":
        lo, hi = float(face.min()), float(face.max())
    elif str(scale_mode) == "absolute_px":
        lo, hi = float(face_px_small), float(face_px_large)
    else:
        raise ValueError(f"unknown FaceRefine scale_mode {scale_mode!r}")
    if hi < lo:
        raise ValueError("face_px_large must be >= face_px_small")
    t = torch.zeros_like(face) if hi - lo < 1e-12 else ((face - lo) / (hi - lo)).clamp(0, 1)
    t = t.pow(float(gamma))
    strength = float(strength_small_face) + (
        float(strength_large_face) - float(strength_small_face)
    ) * t
    return _smooth_1d(strength, int(smooth_frames)).clamp(0, 1).float()


def _split_samples(latent: Any, *, label: str, channels: int, rank: int) -> torch.Tensor:
    if not isinstance(latent, Mapping) or "samples" not in latent:
        raise ValueError(f"{label} must be a LATENT dictionary containing 'samples'")
    samples = latent["samples"]
    if _is_nested(samples):
        members = list(samples.unbind())
        if len(members) != 1:
            raise ValueError(f"{label} split LATENT must contain one tensor")
        samples = members[0]
    if not torch.is_tensor(samples) or samples.ndim != rank or int(samples.shape[1]) != channels:
        raise ValueError(
            f"{label} samples must be rank {rank} with {channels} channels; "
            f"got {getattr(samples, 'shape', None)}"
        )
    if int(samples.shape[0]) != 1 or not samples.is_floating_point():
        raise ValueError(f"{label} samples must be floating-point with batch size 1")
    return samples


def _mask_member(latent: Mapping[str, Any]) -> torch.Tensor | None:
    mask = latent.get("noise_mask")
    if mask is None:
        return None
    if _is_nested(mask):
        members = list(mask.unbind())
        if len(members) != 1:
            raise ValueError("split Continuum video noise_mask must contain one tensor")
        mask = members[0]
    if not torch.is_tensor(mask):
        raise TypeError("Continuum video noise_mask must be a torch.Tensor")
    return mask


def inherited_temporal_mask(mask: torch.Tensor | None, source_t: int, target_t: int) -> torch.Tensor:
    """Extract an exact spatially uniform temporal mask and map it to the crop latent T."""

    if mask is None:
        return torch.ones(target_t, dtype=torch.float32)
    if mask.ndim != 5 or int(mask.shape[0]) != 1:
        raise ValueError("Continuum video noise_mask must be [1,C,T,H,W]")
    if int(mask.shape[2]) not in (1, int(source_t)):
        raise ValueError("Continuum video noise_mask temporal size does not match video_latent")
    work = mask.detach().float().movedim(2, 0).reshape(int(mask.shape[2]), -1)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("Continuum video noise_mask contains NaN or Inf")
    low, high = work.amin(dim=1), work.amax(dim=1)
    if not torch.equal(low, high):
        raise ValueError(
            "Continuum video noise_mask is spatially non-uniform; exact crop-space remapping "
            "requires a public spatial transform contract"
        )
    if bool(((low < 0) | (low > 1)).any()):
        raise ValueError("Continuum video noise_mask values must be in [0, 1]")
    values = low.view(1, 1, -1)
    if int(values.shape[-1]) != int(target_t):
        values = F.interpolate(values, size=int(target_t), mode="nearest")
    return values.view(-1)


def compose_video_mask(
    inherited: torch.Tensor,
    face_strength_frames: torch.Tensor,
    video: torch.Tensor,
) -> torch.Tensor:
    """Multiply ComfyUI denoise masks: inherited zero remains strictly protected."""

    face = F.interpolate(
        face_strength_frames.float().view(1, 1, -1),
        size=int(video.shape[2]),
        mode="linear",
        align_corners=True,
    ).view(-1)
    if int(inherited.numel()) != int(video.shape[2]):
        raise ValueError("inherited video mask does not match encoded crop latent T")
    combined = (inherited.to(face.device) * face).clamp(0, 1)
    return combined.view(1, 1, -1, 1, 1).expand(
        int(video.shape[0]), 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
    ).contiguous()


def _zero_prefix(values: torch.Tensor) -> int:
    count = 0
    for item in values.detach().cpu().tolist():
        if float(item) != 0.0:
            break
        count += 1
    return count


def carry_previous_refined_prefix(
    current_video: torch.Tensor,
    previous_video: torch.Tensor | None,
    inherited_mask: torch.Tensor,
) -> int:
    """Carry sampler-2's real tail into the next fully protected Native Masked prefix."""

    steps = _zero_prefix(inherited_mask)
    if not steps or previous_video is None:
        return 0
    if steps > int(current_video.shape[2]) or steps > int(previous_video.shape[2]):
        raise ValueError("protected Continuum crop prefix exceeds available latent time")
    if (
        tuple(current_video.shape[:2]) != tuple(previous_video.shape[:2])
        or tuple(current_video.shape[-2:]) != tuple(previous_video.shape[-2:])
    ):
        raise ValueError("protected Continuum crop groups changed latent geometry")
    current_video[:, :, :steps].copy_(previous_video[:, :, -steps:])
    return steps


def h3_context_latent_steps(frame_count: int) -> int | None:
    frames = int(frame_count)
    if frames <= 0:
        return 0
    if frames < 5 or (frames - 5) % 17:
        return None
    return 2 + 5 * ((frames - 5) // 17)


def h3_vae_frame_overlap(vae: Any) -> int:
    first_stage = getattr(vae, "first_stage_model", None)
    value = getattr(first_stage, "frame_overlap", 5)
    try:
        overlap = int(value)
    except (TypeError, ValueError):
        overlap = 5
    return max(0, overlap)


def repair_previous_decode_tail(
    previous_retained: torch.Tensor,
    next_context: torch.Tensor | None,
    *,
    trim_frames: int,
    carried_latent_prefix: int,
    vae: Any,
) -> int:
    """Legacy experimental seam helper retained for compatibility/tests.

    The active Continuum quality path does not call this helper. Runtime validation
    showed the decoder-tail substitution did not resolve the reported visual defect.
    """

    trim = int(trim_frames)
    if trim <= 0 or next_context is None or int(next_context.shape[0]) < trim:
        return 0
    expected_prefix = h3_context_latent_steps(trim)
    if expected_prefix is None or int(carried_latent_prefix) < expected_prefix:
        return 0
    overlap = min(
        h3_vae_frame_overlap(vae),
        trim,
        int(previous_retained.shape[0]),
        int(next_context.shape[0]),
    )
    if overlap <= 0:
        return 0
    previous_retained[-overlap:].copy_(
        next_context[-overlap:].to(previous_retained.device, previous_retained.dtype)
    )
    return overlap


def _resize_spatial(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tensor.ndim == 4:
        return F.interpolate(tensor.float(), size=(height, width), mode="nearest").to(tensor.dtype)
    if tensor.ndim == 5:
        b, c, t, h, w = tensor.shape
        work = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        out = F.interpolate(work, size=(height, width), mode="nearest")
        return out.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4).to(tensor.dtype)
    raise ValueError("MiniMax H3 keyframe latent must have rank 4 or 5")


def resize_target_conditioning(conditioning: list, target_h: int, target_w: int) -> list:
    result = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            result.append(entry)
            continue
        metadata = entry[1]
        updated = metadata.copy()
        keyframes = metadata.get("minimax_keyframes")
        if keyframes is not None:
            resized = []
            for raw in keyframes:
                if not isinstance(raw, Mapping):
                    raise TypeError("MiniMax H3 keyframe must be a mapping")
                block = dict(raw)
                latent = block.get("latent")
                if latent is not None:
                    if not torch.is_tensor(latent) or int(latent.shape[1]) != H3_VIDEO_CHANNELS:
                        raise ValueError("MiniMax H3 keyframe latent must use 24 video channels")
                    latent = _resize_spatial(latent, target_h, target_w)
                    block["latent"] = latent
                    if "latent_h" in block:
                        block["latent_h"] = int(target_h)
                    if "latent_w" in block:
                        block["latent_w"] = int(target_w)
                resized.append(block)
            updated["minimax_keyframes"] = resized
        rebuilt = [entry[0], updated, *entry[2:]]
        result.append(tuple(rebuilt) if isinstance(entry, tuple) else rebuilt)
    return result


def adapt_conditioning_to_crop(
    conditioning: list,
    vae: Any,
    physical_crops: torch.Tensor,
    target_h: int,
    target_w: int,
) -> list:
    output = []
    frame_count = int(physical_crops.shape[0])
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            output.append(entry)
            continue
        metadata = entry[1]
        updated = metadata.copy()
        keyframes = metadata.get("minimax_keyframes")
        if keyframes is not None:
            rebound = []
            for raw in keyframes:
                if not isinstance(raw, Mapping):
                    raise TypeError("MiniMax H3 keyframe must be a mapping")
                block = dict(raw)
                if block.get("latent") is not None:
                    frame_index = int(block.get("resolved_frame_index", 0))
                    if frame_index < 0 or frame_index >= frame_count:
                        raise ValueError(
                            f"MiniMax H3 visual keyframe index {frame_index} is outside the "
                            f"{frame_count}-frame physical crop group"
                        )
                    latent = vae.encode(physical_crops[frame_index : frame_index + 1, ..., :3])
                    if not torch.is_tensor(latent):
                        raise ValueError("Video VAE must encode a crop keyframe as [1,24,T,H,W]")
                    if latent.ndim == 4:
                        latent = latent.unsqueeze(0).movedim(1, 2)
                    if latent.ndim != 5 or tuple(latent.shape[:2]) != (1, H3_VIDEO_CHANNELS):
                        raise ValueError("Video VAE must encode a crop keyframe as [1,24,T,H,W]")
                    latent = _resize_spatial(latent, target_h, target_w)
                    block["latent"] = latent
                    if "latent_h" in block:
                        block["latent_h"] = int(target_h)
                    if "latent_w" in block:
                        block["latent_w"] = int(target_w)
                    if "latent_t" in block:
                        block["latent_t"] = int(latent.shape[2])
                rebound.append(block)
            updated["minimax_keyframes"] = rebound
        rebuilt = [entry[0], updated, *entry[2:]]
        output.append(tuple(rebuilt) if isinstance(entry, tuple) else rebuilt)
    return output


def validate_refine_state(value: Any) -> tuple[Any, list]:
    if not isinstance(value, Mapping):
        raise TypeError("H3 Continuum refine_state must be a dictionary")
    if type(value.get("api")) is not int or value["api"] != REFINE_STATE_API:
        raise ValueError(
            f"Unsupported H3 Continuum refine_state API {value.get('api')!r}; expected {REFINE_STATE_API}"
        )
    model, positive = value.get("model"), value.get("positive")
    if model is None or not isinstance(positive, list):
        raise ValueError("H3 Continuum refine_state is missing MODEL or positive CONDITIONING")
    return model, positive


def model_for_refinement(model: Any) -> Any:
    clone = getattr(model, "clone", None)
    get_object = getattr(model, "get_model_object", None)
    if not callable(clone) or not callable(get_object):
        raise TypeError("H3 Continuum refine_state MODEL must be a ComfyUI ModelPatcher")
    result = clone()
    if result is model:
        raise RuntimeError("MODEL.clone() returned the captured source MODEL")
    try:
        sigma_max = result.get_model_object("model_sampling").sigma_max
        if torch.is_tensor(sigma_max):
            if sigma_max.numel() != 1:
                raise ValueError("H3 refinement MODEL sigma_max must be scalar")
            sigma_reference = float(sigma_max.detach().cpu().reshape(-1)[0].item())
        else:
            sigma_reference = float(sigma_max)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("H3 refinement MODEL does not expose scalar model_sampling.sigma_max") from exc
    if not math.isfinite(sigma_reference) or sigma_reference <= 0:
        raise ValueError("H3 refinement sigma reference must be finite and positive")

    options = getattr(result, "model_options", None)
    if not isinstance(options, dict):
        raise TypeError("H3 refinement MODEL.model_options must be a dictionary")
    options = dict(options)
    transformer = options.get("transformer_options")
    if transformer is None:
        transformer = {}
    elif not isinstance(transformer, dict):
        raise TypeError("H3 refinement transformer_options must be a dictionary")
    else:
        transformer = dict(transformer)
    transformer[H3_REFINEMENT_KEY] = {
        "api": H3_REFINEMENT_API,
        "active": True,
        "min_actual_prefix_steps": 0,
        "sigma_reference": sigma_reference,
    }
    options["transformer_options"] = transformer
    result.model_options = options
    return result


def encode_video_strict(vae: Any, images: torch.Tensor, expected_t: int) -> torch.Tensor:
    encoded = vae.encode(images[..., :3])
    if not torch.is_tensor(encoded):
        raise ValueError("Video VAE must encode a physical crop group as [1,24,T,H,W]")
    if encoded.ndim == 4:
        encoded = encoded.unsqueeze(0).movedim(1, 2)
    if encoded.ndim != 5 or tuple(encoded.shape[:2]) != (1, H3_VIDEO_CHANNELS):
        raise ValueError("Video VAE must encode a physical crop group as [1,24,T,H,W]")
    if int(encoded.shape[2]) != int(expected_t):
        raise ValueError(
            f"Exact Continuum crop topology failed: Video VAE encoded T={int(encoded.shape[2])}, "
            f"physical group expects T={int(expected_t)}. No temporal trim/pad was applied."
        )
    return encoded


def _basic_guider(model: Any, positive: list):
    import comfy.samplers

    class _BasicGuider(comfy.samplers.CFGGuider):
        def set_positive(self, value):
            self.inner_set_conds({"positive": value})

    guider = _BasicGuider(model)
    guider.set_positive(positive)
    return guider


def validate_partial_sigmas(sigmas: Any, *, label: str = "FaceRefine sigmas") -> torch.Tensor:
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError(f"{label} must contain at least start and end values")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError(f"{label} must be finite")
    sigma_start = float(sigmas[0].detach().cpu())
    if sigma_start < 0 or sigma_start >= 1:
        raise ValueError(
            f"Continuum FaceRefine requires a partial-denoise schedule with "
            f"0 <= sigmas[0] < 1; got {sigma_start!r}. Connect a separate BasicScheduler "
            f"for sampler 2. Current validated Continuum baseline: simple, 12 steps, "
            f"denoise 0.45. Do not reuse Continuum sampler-1 SIGMAS."
        )
    return sigmas


def sample_locked_audio(
    clean: dict[str, Any],
    *,
    model: Any,
    positive: list,
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import comfy.model_management
    import comfy.sample
    import comfy.utils
    import latent_preview

    sigmas = validate_partial_sigmas(sigmas)
    latent = dict(clean)
    samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"], None, None)
    latent["samples"] = samples
    members = list(samples.unbind())
    if len(members) != 2:
        raise ValueError("MiniMax H3 refinement samples must contain [video, audio]")
    clean_video, clean_audio = members
    generated = noise.generate_noise(latent)
    noise_members = list(generated.unbind())
    if len(noise_members) != 2 or tuple(noise_members[0].shape) != tuple(clean_video.shape) or tuple(noise_members[1].shape) != tuple(clean_audio.shape):
        raise ValueError("generated FaceRefine noise does not match the exact H3 AV latent")
    generated = _nested((noise_members[0], torch.zeros_like(noise_members[1])))

    guider = _basic_guider(model, positive)
    callback = latent_preview.prepare_callback(
        getattr(guider, "model_patcher", model), int(sigmas.numel()) - 1, {}
    )
    sampled = guider.sample(
        generated,
        samples,
        sampler,
        sigmas,
        denoise_mask=latent.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(getattr(noise, "seed", 0)),
    )
    sampled = sampled.to(comfy.model_management.intermediate_device())
    sampled_members = list(sampled.unbind())
    if len(sampled_members) != 2:
        raise ValueError("MiniMax H3 sampler returned an invalid AV NestedTensor")
    sampled_video = sampled_members[0]
    restored_audio = clean_audio.to(sampled_members[1].device)
    return sampled_video, restored_audio


def decode_video_strict(
    vae: Any,
    video: torch.Tensor,
    *,
    total_frames: int,
    crop_height: int,
    crop_width: int,
) -> torch.Tensor:
    decoded = vae.decode(video)
    if not torch.is_tensor(decoded):
        raise ValueError("Video VAE decode must return a torch.Tensor")
    if decoded.ndim == 5:
        if int(decoded.shape[0]) != 1:
            raise ValueError(
                "Video VAE decode returned multiple batches; each Continuum physical group "
                "must decode as [1,frames,H,W,C]"
            )
        decoded = decoded.reshape(
            -1, int(decoded.shape[-3]), int(decoded.shape[-2]), int(decoded.shape[-1])
        )
    elif decoded.ndim != 4:
        raise ValueError(
            "Video VAE decode must return [frames,H,W,C] or [1,frames,H,W,C]; "
            f"got shape {tuple(decoded.shape)}"
        )
    if int(decoded.shape[0]) != int(total_frames):
        raise ValueError(
            f"Exact Continuum crop topology failed: Video VAE decoded {int(decoded.shape[0])} "
            f"frames, physical group expects {int(total_frames)}. No temporal trim/pad was applied."
        )
    if tuple(decoded.shape[1:3]) != (int(crop_height), int(crop_width)):
        raise ValueError("Video VAE decoded crop geometry differs from the tracked crop canvas")
    return decoded[..., :3]


def refine_physical_group(
    *,
    position: int,
    group: Mapping[str, Any],
    crops: torch.Tensor,
    strengths: torch.Tensor,
    video_latent: Mapping[str, Any],
    audio_latent: Mapping[str, Any],
    refine_state: Mapping[str, Any],
    vae: Any,
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    previous_refined_video: torch.Tensor | None,
):
    source_video = _split_samples(
        video_latent,
        label=f"video_latents[{position}]",
        channels=H3_VIDEO_CHANNELS,
        rank=5,
    )
    source_audio = _split_samples(
        audio_latent,
        label=f"audio_latents[{position}]",
        channels=H3_AUDIO_CHANNELS,
        rank=4,
    )
    if int(source_video.shape[2]) != int(group["expected_video_latent_t"]):
        raise ValueError(f"video_latents[{position}] temporal size does not match assembly_plan")
    if int(source_audio.shape[-1]) != int(group["expected_audio_latent_t"]):
        raise ValueError(f"audio_latents[{position}] temporal size does not match assembly_plan")

    encoded = encode_video_strict(vae, crops, int(group["expected_video_latent_t"]))
    inherited = inherited_temporal_mask(
        _mask_member(video_latent), int(source_video.shape[2]), int(encoded.shape[2])
    ).to(encoded.device)
    carried = carry_previous_refined_prefix(encoded, previous_refined_video, inherited)
    video_mask = compose_video_mask(inherited, strengths, encoded)

    audio = source_audio.to(encoded.device)
    audio_mask = torch.zeros(
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3])),
        dtype=torch.float32,
        device=audio.device,
    )
    clean = {
        "samples": _nested((encoded, audio)),
        "noise_mask": _nested((video_mask, audio_mask)),
    }
    captured_model, captured_positive = validate_refine_state(refine_state)
    positive = adapt_conditioning_to_crop(
        captured_positive,
        vae,
        crops,
        int(encoded.shape[-2]),
        int(encoded.shape[-1]),
    )
    refined_video, restored_audio = sample_locked_audio(
        clean,
        model=model_for_refinement(captured_model),
        positive=positive,
        noise=noise,
        sampler=sampler,
        sigmas=sigmas,
    )
    if not torch.equal(restored_audio.detach().cpu(), source_audio.detach().cpu()):
        raise RuntimeError("FaceRefine audio lock failed to restore pass-1 audio exactly")

    decoded = decode_video_strict(
        vae,
        refined_video,
        total_frames=int(group["total_frames"]),
        crop_height=int(crops.shape[1]),
        crop_width=int(crops.shape[2]),
    )
    trim = int(group["trim_frames"])
    context = decoded[:trim] if trim else None
    segment = decoded[trim:]
    if int(segment.shape[0]) != int(group["net_frames"]):
        raise RuntimeError("refined physical group retained the wrong number of frames")
    report = (
        f"group {position + 1}: logical={group['logical_chunk_indices']}, "
        f"physical={group['total_frames']}, trim={trim}, retained={group['net_frames']}, "
        f"terminal_merged={group['terminal_merged']}, carried_latent_prefix={carried}"
    )
    context_out = None if context is None else context.to(crops.device, crops.dtype)
    return segment.to(crops.device, crops.dtype), refined_video, context_out, carried, report


class H3ContinuumFaceRefine:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
                "video_latents": ("LATENT",),
                "audio_latents": ("LATENT",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "refine_state": ("H3_CONTINUUM_REFINE_STATE",),
                "vae": ("VAE",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "Connect a separate sampler-2 BasicScheduler. Current validated "
                        "Continuum baseline: simple, 12 steps, denoise 0.45. Do not reuse "
                        "Continuum sampler-1 SIGMAS."
                    },
                ),
                "strength_small_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "strength_large_face": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scale_mode": (["absolute_px", "relative_to_clip"], {"default": "absolute_px"}),
                "face_px_small": ("FLOAT", {"default": 30.0, "min": 4.0, "max": 400.0, "step": 1.0}),
                "face_px_large": ("FLOAT", {"default": 120.0, "min": 8.0, "max": 800.0, "step": 1.0}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("refined_crops", "report")
    FUNCTION = "refine"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = (
        "Refine one globally tracked H3 Continuum crop timeline using exact physical-group "
        "MODEL/CONDITIONING state, inherited Native Masked protection, and locked pass-1 audio."
    )

    def refine(
        self,
        crops,
        transform,
        video_latents,
        audio_latents,
        assembly_plan,
        refine_state,
        vae,
        noise,
        sampler,
        sigmas,
        strength_small_face,
        strength_large_face,
        scale_mode,
        face_px_small,
        face_px_large,
        gamma,
        smooth_frames,
    ):
        crops = _single(crops, "crops")
        transform = _single(transform, "transform")
        vae = _single(vae, "vae")
        groups, natural_frames = validate_physical_groups(assembly_plan)
        if not torch.is_tensor(crops) or crops.ndim != 4:
            raise ValueError("crops must be one IMAGE batch [frames,H,W,C]")
        if int(crops.shape[0]) != natural_frames:
            plan = _single(assembly_plan, "assembly_plan")
            target = int(plan.get("target_frames", -1))
            hint = (
                " Select Timeline Output = Natural retained timeline (Refinement), then use "
                "H3 Continuum Finalize Duration V3.4 after H3 Face Stitch Back."
                if int(crops.shape[0]) == target and target != natural_frames
                else ""
            )
            raise ValueError(
                f"globally tracked crops contain {int(crops.shape[0])} frames, while the "
                f"assembly_plan physical groups retain {natural_frames}.{hint}"
            )
        if int(transform.get("frames", -1)) != natural_frames or len(transform.get("boxes", ())) != natural_frames:
            raise ValueError("H3FACEXFORM is not aligned with the natural assembled crop timeline")

        videos = _group_list(video_latents, "video_latents")
        audios = _group_list(audio_latents, "audio_latents")
        states = _group_list(refine_state, "refine_state")
        expected = len(groups)
        if len(videos) != expected or len(audios) != expected or len(states) != expected:
            raise ValueError(
                "Continuum physical-group inputs are misaligned: "
                f"plan={expected}, video_latents={len(videos)}, audio_latents={len(audios)}, "
                f"refine_state={len(states)}. Regenerate from Chunk 1 when Refine State is required."
            )

        sigma_values = [_value_at(sigmas, index) for index in range(expected)]
        for index, value in enumerate(sigma_values, start=1):
            validate_partial_sigmas(value, label=f"FaceRefine sigmas for physical group {index}")

        strengths = face_strength_timeline(
            transform,
            strength_small_face=float(_single(strength_small_face, "strength_small_face")),
            strength_large_face=float(_single(strength_large_face, "strength_large_face")),
            scale_mode=str(_single(scale_mode, "scale_mode")),
            face_px_small=float(_single(face_px_small, "face_px_small")),
            face_px_large=float(_single(face_px_large, "face_px_large")),
            gamma=float(_single(gamma, "gamma")),
            smooth_frames=int(_single(smooth_frames, "smooth_frames")),
        )
        crop_groups = partition_global_timeline(crops, groups)
        strength_groups = partition_global_timeline(strengths, groups)

        retained: list[torch.Tensor] = []
        previous_refined_video = None
        report_lines = [
            f"Continuum FaceRefine: {expected} physical group(s), {natural_frames} retained frames"
        ]
        for index, (group, group_crops, group_strength) in enumerate(
            zip(groups, crop_groups, strength_groups)
        ):
            segment, previous_refined_video, context, carried, group_report = refine_physical_group(
                position=index,
                group=group,
                crops=group_crops,
                strengths=group_strength,
                video_latent=videos[index],
                audio_latent=audios[index],
                refine_state=states[index],
                vae=vae,
                noise=_value_at(noise, index),
                sampler=_value_at(sampler, index),
                sigmas=sigma_values[index],
                previous_refined_video=previous_refined_video,
            )
            if index > 0 and retained:
                seam_frames = repair_previous_decode_tail(
                    retained[-1],
                    context,
                    trim_frames=int(group["trim_frames"]),
                    carried_latent_prefix=carried,
                    vae=vae,
                )
                if seam_frames:
                    report_lines.append(
                        f"boundary {index}->{index + 1}: decoder seam repaired with "
                        f"{seam_frames} future-context frame(s)"
                    )
            retained.append(segment)
            report_lines.append(group_report)

        output = torch.cat(retained, dim=0)
        if int(output.shape[0]) != natural_frames:
            raise RuntimeError("refined crop timeline length/order does not match assembly_plan")
        report_lines.append("audio: locked during sampler 2 and restored bit-exact")
        report = "\n".join(report_lines)
        print("[H3FaceRefine] " + report.replace("\n", "\n[H3FaceRefine] "))
        return output, report
