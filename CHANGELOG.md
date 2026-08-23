# Changelog

## Unreleased

## 1.1.0

- Added exact H3 Continuum V3.4 sampler-2 integration using Continuum's captured per-physical-sample MODEL, positive CONDITIONING, latent topology, inherited Native Masked protection, and assembly plan instead of rebuilding a second ReferenceToVideo path.
- Added natural retained timeline support for downstream global face tracking/refinement together with the Continuum post-stitch duration finalizer contract.
- Preserved sampler-1 audio bit-exactly during face refinement by applying zero sampler-2 audio noise/mask and restoring the original pass-1 audio member after sampling.
- Added real-detection gating so interpolated/no-face tracker frames remain temporal context but receive zero sampler-2 denoise strength.
- Added VAE round-trip cancellation: clean and refined crop latents are decoded together and only `decode(refined) - decode(clean)` is transferred onto the tracked source crop.
- Reworked Continuum stitch-back to transfer only the learned residual onto the untouched full-resolution source, eliminating absolute resized-crop replacement and moving black crop-boundary artifacts.
- Added causal detector-dropout compositing, fractional/subpixel mask geometry, source-space feathering, and residual colour-bias handling.
- Removed the obsolete crop-magnification hard gate. The validated Continuum baseline is 512x512; larger canvases are optional quality/performance A/B choices rather than required fixes for sub-1x crops.
- Set the current validated Continuum sampler-2 guidance to `simple`, 12 steps, `denoise=0.45`; the earlier 4-step suggestion was removed after GPU testing showed visible artifacts.
- Documented CUDA InsightFace/ONNX Runtime setup and the `onnxruntime` versus `onnxruntime-gpu` package-metadata caveat.
- Added comprehensive CPU contract coverage for physical-group reconstruction, mask composition, conditioning preservation, audio locking, VAE topology, residual stitch invariants, moving-edge regressions, canvas guidance, and standalone import compatibility.
- Added tested-main GitHub Release automation, expanded Python CI, release archive validation, and pinned Comfy Registry publishing for the xmarre fork.

## 1.0.0

- Upstream baseline from Carasibana/ComfyUI-H3-FaceRefine: per-frame face detection/tracking, normalized crop generation, H3 latent injection/per-frame denoise helpers, optional SAM masks, and stitch-back workflows.
