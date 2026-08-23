"""Runtime-quality fixes for H3 FaceRefine's Continuum integration.

The public Continuum path is deliberately stricter than stock FaceRefine:

* interpolated/no-face frames remain temporal context but get a zero sampler-2
  denoise mask;
* sampler-2 RGB output is converted to a VAE-roundtrip-cancelled correction:
  ``source_crop + (decoded_refined - decoded_clean)``.  This removes the VideoVAE
  encode/decode reconstruction component before Stitch Back sees the crop;
* stitch opacity around detector dropouts is causal: valid frames are never
  pre-faded before a loss of detection.

The module is installed as a small compatibility layer so the standalone nodes
and upstream-facing implementation stay importable without Continuum.
"""

from __future__ import annotations

import copy
import math
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F


VALIDATED_SAMPLER2_STEPS = 12
VALIDATED_SAMPLER2_DENOISE = 0.45


def require_detection_gate(transform: Mapping[str, Any], frames: int) -> torch.Tensor:
    detected = transform.get("detected") if isinstance(transform, Mapping) else None
    if not isinstance(detected, (list, tuple)) or len(detected) != int(frames):
        raise ValueError(
            "H3FACEXFORM is missing an aligned per-frame 'detected' mask. "
            "Rerun H3 Face Track + Crop with the current FaceRefine version before "
            "using H3 Continuum Face Refine."
        )
    return torch.tensor([1.0 if bool(v) else 0.0 for v in detected], dtype=torch.float32)


def dropout_runs(detected: torch.Tensor) -> list[tuple[int, int]]:
    values = detected.detach().cpu().bool().tolist()
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(values):
        if not value and start is None:
            start = index
        elif value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def safe_stitch_weights(detected: list[bool], mode: str, fade_frames: int = 4) -> list[float] | None:
    """Return causal stitch opacity; never pre-fade a still-valid frame."""

    if mode == "composite_anyway":
        return None
    valid = [bool(v) for v in detected]
    weights = [1.0 if value else 0.0 for value in valid]
    if mode == "skip" or fade_frames <= 1:
        return weights
    if mode != "fade_out":
        raise ValueError(f"unknown undetected_frames mode {mode!r}")

    fade = max(1, int(fade_frames))
    for index in range(1, len(valid)):
        if valid[index] and not valid[index - 1]:
            for offset in range(fade):
                pos = index + offset
                if pos >= len(valid) or not valid[pos]:
                    break
                weights[pos] = min(weights[pos], float(offset + 1) / float(fade))
    return weights


def _gaussian_blur(mask: torch.Tensor, feather: int) -> torch.Tensor:
    if int(feather) <= 0:
        return mask
    k = 2 * int(feather) + 1
    shortest = min(int(mask.shape[-2]), int(mask.shape[-1]))
    if shortest <= k:
        k = max(3, int(shortest / 2) | 1)
    sigma = max(k / 6.0, 0.5)
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(x**2) / (2.0 * sigma**2))
    g = (g / g.sum()).to(mask.dtype)
    pad = k // 2
    out = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, k, 1))
    return out


def fractional_rect_mask(
    height: int,
    width: int,
    rect: tuple[float, float, float, float],
    dilation: float,
    *,
    device,
) -> torch.Tensor:
    """Rasterise a float rectangle with fractional-pixel edge coverage."""

    fx, fy, fw, fh = (float(v) for v in rect)
    x0, y0 = fx - dilation, fy - dilation
    x1, y1 = fx + fw + dilation, fy + fh + dilation
    xx = torch.arange(int(width), device=device, dtype=torch.float32)
    yy = torch.arange(int(height), device=device, dtype=torch.float32)
    sx0 = torch.tensor(x0, device=device, dtype=torch.float32)
    sx1 = torch.tensor(x1, device=device, dtype=torch.float32)
    sy0 = torch.tensor(y0, device=device, dtype=torch.float32)
    sy1 = torch.tensor(y1, device=device, dtype=torch.float32)
    cov_x = (torch.minimum(xx + 1.0, sx1) - torch.maximum(xx, sx0)).clamp(0.0, 1.0)
    cov_y = (torch.minimum(yy + 1.0, sy1) - torch.maximum(yy, sy0)).clamp(0.0, 1.0)
    return (cov_y[:, None] * cov_x[None, :]).view(1, 1, int(height), int(width))


def fractional_ellipse_mask(
    height: int,
    width: int,
    rect: tuple[float, float, float, float],
    dilation: float,
    *,
    device,
) -> torch.Tensor:
    fx, fy, fw, fh = (float(v) for v in rect)
    fx -= dilation
    fy -= dilation
    fw += 2.0 * dilation
    fh += 2.0 * dilation
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    rx, ry = max(fw / 2.0, 1.0), max(fh / 2.0, 1.0)
    xx = torch.arange(int(width), device=device, dtype=torch.float32) + 0.5
    yy = torch.arange(int(height), device=device, dtype=torch.float32) + 0.5
    radius = torch.sqrt(((xx[None, :] - cx) / rx) ** 2 + ((yy[:, None] - cy) / ry) ** 2)
    signed_px = (1.0 - radius) * min(rx, ry)
    return (signed_px + 0.5).clamp(0.0, 1.0).view(1, 1, int(height), int(width))


def _canvas_mask_batch(
    *,
    start: int,
    stop: int,
    height: int,
    width: int,
    paste_region: str,
    face_rects,
    dilation: int,
    masks,
    device,
) -> torch.Tensor:
    n = stop - start
    if masks is not None:
        if int(masks.shape[0]) < stop:
            raise ValueError("H3 Face Stitch Back MASK batch is shorter than the image timeline")
        block = masks[start:stop].to(device).float()
        if block.ndim != 3:
            raise ValueError("H3 Face Stitch Back masks must be [frames,H,W]")
        if tuple(block.shape[-2:]) != (int(height), int(width)):
            block = F.interpolate(
                block.unsqueeze(1), size=(int(height), int(width)), mode="bilinear", align_corners=False
            ).squeeze(1)
        block = block.unsqueeze(1)
        if int(dilation) > 0:
            k = 2 * int(dilation) + 1
            block = F.max_pool2d(block, k, stride=1, padding=k // 2)
        return block.clamp(0, 1)

    if paste_region == "full_crop":
        return torch.ones((n, 1, int(height), int(width)), device=device, dtype=torch.float32)

    result = []
    for index in range(start, stop):
        rect = (
            face_rects[index]
            if face_rects is not None and index < len(face_rects)
            else (width * 0.25, height * 0.25, width * 0.5, height * 0.5)
        )
        if paste_region == "face_ellipse":
            result.append(fractional_ellipse_mask(height, width, rect, float(dilation), device=device))
        else:
            result.append(fractional_rect_mask(height, width, rect, float(dilation), device=device))
    return torch.cat(result, dim=0)


def _warp_grid(boxes, start: int, stop: int, *, width: int, height: int, device) -> torch.Tensor:
    n = stop - start
    theta = torch.empty((n, 2, 3), dtype=torch.float32, device=device)
    for row, index in enumerate(range(start, stop)):
        x, y, bw, bh = (float(v) for v in boxes[index])
        if not all(math.isfinite(v) for v in (x, y, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f"H3 Face Stitch Back has invalid crop box at frame {index}: {boxes[index]!r}")
        theta[row, 0, 0] = float(width) / bw
        theta[row, 0, 1] = 0.0
        theta[row, 0, 2] = (float(width) - 2.0 * x) / bw - 1.0
        theta[row, 1, 0] = 0.0
        theta[row, 1, 1] = float(height) / bh
        theta[row, 1, 2] = (float(height) - 2.0 * y) / bh - 1.0
    return F.affine_grid(theta, (n, 3, int(height), int(width)), align_corners=False)


def validate_partial_sigmas_current(sigmas: Any, *, label: str = "FaceRefine sigmas") -> torch.Tensor:
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError(f"{label} must contain at least start and end values")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError(f"{label} must be finite")
    sigma_start = float(sigmas[0].detach().cpu())
    if sigma_start < 0 or sigma_start >= 1:
        raise ValueError(
            f"Continuum FaceRefine requires a partial-denoise schedule with "
            f"0 <= sigmas[0] < 1; got {sigma_start!r}. Connect a separate BasicScheduler "
            f"for sampler 2; the currently validated runtime baseline is "
            f"{VALIDATED_SAMPLER2_STEPS} steps at denoise {VALIDATED_SAMPLER2_DENOISE:.2f}. "
            "Do not reuse Continuum sampler-1 SIGMAS."
        )
    return sigmas


def _decode_pair_strict(
    vae: Any,
    clean_video: torch.Tensor,
    refined_video: torch.Tensor,
    *,
    total_frames: int,
    crop_height: int,
    crop_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode clean/refined latents in one VAE call and validate exact topology."""

    if tuple(clean_video.shape) != tuple(refined_video.shape):
        raise ValueError("clean/refined crop latents changed shape during sampler 2")
    pair = torch.cat(
        [clean_video, refined_video.to(clean_video.device, clean_video.dtype)], dim=0
    )
    decoded = vae.decode(pair)
    if not torch.is_tensor(decoded):
        raise ValueError("Video VAE paired decode must return a torch.Tensor")
    frames = int(total_frames)
    h, w = int(crop_height), int(crop_width)
    if decoded.ndim == 5:
        if int(decoded.shape[0]) != 2:
            raise ValueError(
                "Video VAE paired decode must preserve the clean/refined batch dimension"
            )
        if int(decoded.shape[1]) != frames:
            raise ValueError(
                f"Exact Continuum crop topology failed: paired Video VAE decoded "
                f"{int(decoded.shape[1])} frames, physical group expects {frames}."
            )
        if tuple(decoded.shape[2:4]) != (h, w):
            raise ValueError("Video VAE paired decode crop geometry differs from tracked canvas")
        return decoded[0, ..., :3], decoded[1, ..., :3]
    if decoded.ndim == 4:
        if int(decoded.shape[0]) != frames * 2:
            raise ValueError(
                "Video VAE flattened paired decode returned an unexpected frame count"
            )
        if tuple(decoded.shape[1:3]) != (h, w):
            raise ValueError("Video VAE paired decode crop geometry differs from tracked canvas")
        decoded = decoded.reshape(2, frames, h, w, int(decoded.shape[-1]))
        return decoded[0, ..., :3], decoded[1, ..., :3]
    raise ValueError(
        "Video VAE paired decode must return [2,frames,H,W,C] or flattened [2*frames,H,W,C]"
    )


def build_roundtrip_cancelled_physical_group(module):
    """Build the production physical-group sampler with VAE baseline cancellation."""

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
        source_video = module._split_samples(
            video_latent,
            label=f"video_latents[{position}]",
            channels=module.H3_VIDEO_CHANNELS,
            rank=5,
        )
        source_audio = module._split_samples(
            audio_latent,
            label=f"audio_latents[{position}]",
            channels=module.H3_AUDIO_CHANNELS,
            rank=4,
        )
        if int(source_video.shape[2]) != int(group["expected_video_latent_t"]):
            raise ValueError(f"video_latents[{position}] temporal size does not match assembly_plan")
        if int(source_audio.shape[-1]) != int(group["expected_audio_latent_t"]):
            raise ValueError(f"audio_latents[{position}] temporal size does not match assembly_plan")

        encoded = module.encode_video_strict(vae, crops, int(group["expected_video_latent_t"]))
        inherited = module.inherited_temporal_mask(
            module._mask_member(video_latent), int(source_video.shape[2]), int(encoded.shape[2])
        ).to(encoded.device)
        carried = module.carry_previous_refined_prefix(encoded, previous_refined_video, inherited)
        video_mask = module.compose_video_mask(inherited, strengths, encoded)

        audio = source_audio.to(encoded.device)
        audio_mask = torch.zeros(
            (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3])),
            dtype=torch.float32,
            device=audio.device,
        )
        clean = {
            "samples": module._nested((encoded, audio)),
            "noise_mask": module._nested((video_mask, audio_mask)),
        }
        captured_model, captured_positive = module.validate_refine_state(refine_state)
        positive = module.adapt_conditioning_to_crop(
            captured_positive,
            vae,
            crops,
            int(encoded.shape[-2]),
            int(encoded.shape[-1]),
        )
        refined_video, restored_audio = module.sample_locked_audio(
            clean,
            model=module.model_for_refinement(captured_model),
            positive=positive,
            noise=noise,
            sampler=sampler,
            sigmas=sigmas,
        )
        if not torch.equal(restored_audio.detach().cpu(), source_audio.detach().cpu()):
            raise RuntimeError("FaceRefine audio lock failed to restore pass-1 audio exactly")

        clean_decoded, refined_decoded = _decode_pair_strict(
            vae,
            encoded,
            refined_video,
            total_frames=int(group["total_frames"]),
            crop_height=int(crops.shape[1]),
            crop_width=int(crops.shape[2]),
        )
        # Cancel the entire VAE encode/decode baseline. The public IMAGE remains a
        # source-looking crop so existing wiring is unchanged, but the only difference
        # from the tracked source crop is sampler 2's learned latent-space correction.
        delta = refined_decoded.float() - clean_decoded.float()
        corrected = crops[..., :3].to(delta.device, torch.float32) + delta

        trim = int(group["trim_frames"])
        segment = corrected[trim:]
        if int(segment.shape[0]) != int(group["net_frames"]):
            raise RuntimeError("refined physical group retained the wrong number of frames")
        report = (
            f"group {position + 1}: logical={group['logical_chunk_indices']}, "
            f"physical={group['total_frames']}, trim={trim}, retained={group['net_frames']}, "
            f"terminal_merged={group['terminal_merged']}, carried_latent_prefix={carried}, "
            "vae_roundtrip_cancelled=True"
        )
        return (
            segment.to(crops.device, crops.dtype),
            refined_video,
            None,
            carried,
            report,
        )

    refine_physical_group._h3_roundtrip_cancelled = True
    return refine_physical_group


def build_fixed_continuum_refine(base_class):
    module = sys.modules.get(base_class.__module__)
    if module is None:
        raise ImportError(
            f"could not locate loaded Continuum FaceRefine module {base_class.__module__!r}"
        )

    class H3ContinuumFaceRefineFixed(base_class):
        @classmethod
        def INPUT_TYPES(cls):
            spec = copy.deepcopy(base_class.INPUT_TYPES())
            required = spec.get("required", {})
            if "sigmas" in required:
                required["sigmas"] = (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Connect a separate sampler-2 BasicScheduler. Current validated "
                            f"Continuum baseline: {VALIDATED_SAMPLER2_STEPS} steps, denoise "
                            f"{VALIDATED_SAMPLER2_DENOISE:.2f}. Four-step refinement was "
                            "visually rejected because it produced artifacts. Do not reuse "
                            "Continuum sampler-1 SIGMAS."
                        )
                    },
                )
            return spec

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
            crops = module._single(crops, "crops")
            transform = module._single(transform, "transform")
            vae = module._single(vae, "vae")
            groups, natural_frames = module.validate_physical_groups(assembly_plan)
            if not torch.is_tensor(crops) or crops.ndim != 4:
                raise ValueError("crops must be one IMAGE batch [frames,H,W,C]")
            if int(crops.shape[0]) != natural_frames:
                plan = module._single(assembly_plan, "assembly_plan")
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
            if (
                int(transform.get("frames", -1)) != natural_frames
                or len(transform.get("boxes", ())) != natural_frames
            ):
                raise ValueError(
                    "H3FACEXFORM is not aligned with the natural assembled crop timeline"
                )

            videos = module._group_list(video_latents, "video_latents")
            audios = module._group_list(audio_latents, "audio_latents")
            states = module._group_list(refine_state, "refine_state")
            expected = len(groups)
            if len(videos) != expected or len(audios) != expected or len(states) != expected:
                raise ValueError(
                    "Continuum physical-group inputs are misaligned: "
                    f"plan={expected}, video_latents={len(videos)}, audio_latents={len(audios)}, "
                    f"refine_state={len(states)}. Regenerate from Chunk 1 when Refine State is required."
                )

            sigma_values = [module._value_at(sigmas, index) for index in range(expected)]
            for index, value in enumerate(sigma_values, start=1):
                validate_partial_sigmas_current(
                    value, label=f"FaceRefine sigmas for physical group {index}"
                )

            strengths = module.face_strength_timeline(
                transform,
                strength_small_face=float(module._single(strength_small_face, "strength_small_face")),
                strength_large_face=float(module._single(strength_large_face, "strength_large_face")),
                scale_mode=str(module._single(scale_mode, "scale_mode")),
                face_px_small=float(module._single(face_px_small, "face_px_small")),
                face_px_large=float(module._single(face_px_large, "face_px_large")),
                gamma=float(module._single(gamma, "gamma")),
                smooth_frames=int(module._single(smooth_frames, "smooth_frames")),
            )
            gate = require_detection_gate(transform, natural_frames)
            strengths = strengths * gate.to(strengths.device, strengths.dtype)

            crop_groups = module.partition_global_timeline(crops, groups)
            strength_groups = module.partition_global_timeline(strengths, groups)
            runs = dropout_runs(gate)
            missing = natural_frames - int(gate.sum().item())

            retained: list[torch.Tensor] = []
            previous_refined_video = None
            report_lines = [
                f"Continuum FaceRefine: {expected} physical group(s), {natural_frames} retained frames",
                f"detection gate: {natural_frames - missing}/{natural_frames} real-face frames; "
                f"{missing} interpolated/no-face frames forced to denoise mask 0",
                "VAE baseline: clean/refined latents decoded as one pair; RGB round-trip error cancelled",
            ]
            if runs:
                report_lines.append(
                    "protected dropout runs: "
                    + ", ".join(f"{start}-{stop - 1} ({stop - start}f)" for start, stop in runs)
                )

            for index, (group, group_crops, group_strength) in enumerate(
                zip(groups, crop_groups, strength_groups)
            ):
                result = module.refine_physical_group(
                    position=index,
                    group=group,
                    crops=group_crops,
                    strengths=group_strength,
                    video_latent=videos[index],
                    audio_latent=audios[index],
                    refine_state=states[index],
                    vae=vae,
                    noise=module._value_at(noise, index),
                    sampler=module._value_at(sampler, index),
                    sigmas=sigma_values[index],
                    previous_refined_video=previous_refined_video,
                )
                if not isinstance(result, tuple) or len(result) < 3:
                    raise RuntimeError("H3 Continuum FaceRefine physical-group result is invalid")
                segment = result[0]
                previous_refined_video = result[1]
                group_report = result[-1]
                retained.append(segment)
                report_lines.append(group_report)

            output = torch.cat(retained, dim=0)
            if int(output.shape[0]) != natural_frames:
                raise RuntimeError("refined crop timeline length/order does not match assembly_plan")
            report_lines.append("audio: locked during sampler 2 and restored bit-exact")
            report = "\n".join(report_lines)
            print("[H3FaceRefine] " + report.replace("\n", "\n[H3FaceRefine] "))
            return output, report

    H3ContinuumFaceRefineFixed.__name__ = base_class.__name__
    H3ContinuumFaceRefineFixed.__qualname__ = base_class.__qualname__
    return H3ContinuumFaceRefineFixed


def install_quality_fixes(node_class_mappings: dict[str, type]) -> None:
    continuum = node_class_mappings.get("H3ContinuumFaceRefine")
    if continuum is None:
        raise ImportError("FaceRefine node mappings are incomplete; cannot install quality fixes")
    module = sys.modules.get(continuum.__module__)
    if module is None:
        raise ImportError(f"could not locate Continuum FaceRefine module {continuum.__module__!r}")
    current_group = getattr(module, "refine_physical_group", None)
    if not callable(current_group):
        raise ImportError("Continuum FaceRefine module has no physical-group refinement function")
    if not bool(getattr(current_group, "_h3_roundtrip_cancelled", False)):
        module.refine_physical_group = build_roundtrip_cancelled_physical_group(module)
    module.validate_partial_sigmas = validate_partial_sigmas_current
    node_class_mappings["H3ContinuumFaceRefine"] = build_fixed_continuum_refine(continuum)
