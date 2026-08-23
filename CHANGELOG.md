# Changelog

## Unreleased

- Fixed face tracking switching or drifting to the wrong person during fast motion, detector dropouts, and crowd transitions. Tracking now uses motion prediction with plausibility gating, rejects unsafe lone-face reacquisition, and enables identity handling when multiple faces appear anywhere in the clip rather than only on frame 0.
- Kept the configured YOLO detector as the sole tracking-geometry source. InsightFace now identifies which YOLO box belongs to the subject without substituting its differently shaped bbox, eliminating detector-to-detector box jumps.
- Fixed body fallback selecting the largest person instead of the tracked subject; fallback bodies are now associated to the expected tracked head position.
- Split stabilized crop motion from responsive face-region geometry. A median-3 detector trajectory rejects isolated spikes; the crop centre is smoothed but bounded to within 12% of face height from real motion, while the refine/SAM region follows the responsive detector geometry inside the crop instead of inheriting crop lag.
- Fixed refine-mask placement when crops are clamped to frame boundaries. Face rectangles are now mapped through the actual source-to-canvas crop transform, preserving both the face's motion offset inside a stabilized crop and edge-clamping offsets instead of forcing the region to canvas centre.
- Separated face-mask geometry from `max_of_clip` crop sizing so constant crop size no longer inflates the actual face region on smaller-face frames.
- Added regression coverage for velocity prediction, late crowd activation, lone-bystander rejection, YOLO/InsightFace geometry separation, body association, smoothing lag/spike handling, responsive face motion inside a stabilized crop, and clamped-crop face placement.

## 1.1.1

- Fixed no-face clips aborting the entire workflow after expensive upstream generation. `H3 Face Track + Crop` now emits an explicit no-face passthrough transform instead of throwing when no human face is detected in any frame.
- `H3 Continuum Face Refine` recognises that passthrough contract before any physical-group validation, VAE work, or sampler-2 refinement and returns the original timeline unchanged.
- `H3 Face Stitch Back` returns the untouched base video for no-face timelines; SAM masks, per-frame denoise, and transform diagnostics also become cheap no-ops instead of inventing fallback face regions or failing.
- The passthrough only converts the deliberate all-frames-missed condition. Detector/configuration/topology errors remain hard failures so genuine problems are not hidden.
- Added regression coverage for the full no-face chain, auxiliary no-op stages, unrelated error propagation, and installer idempotence.

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
