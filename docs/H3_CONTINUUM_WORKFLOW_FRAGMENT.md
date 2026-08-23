# H3 Continuum V3.4 → FaceRefine workflow fragment

This path is for H3 video produced by **H3 Continuum Sampler V3.4**. It reuses Continuum's exact per-physical-sample MODEL, positive CONDITIONING, latent masks, and assembly state for sampler 2. It does **not** build a second `MiniMaxH3ReferenceToVideo` path.

## Required order

1. `H3 Continuum Sampler V3.4`
2. decode the physical video/audio outputs
3. `H3 Continuum Assemble + Seam V3.4`
   - **Timeline Output = Natural retained timeline (Refinement)**
4. `H3 Face Track + Crop`
5. `H3 Continuum Face Refine`
6. `H3 Face Stitch Back`
7. `H3 Continuum Finalize Duration V3.4`
8. save/combine video with the assembler/finalizer audio path

Do not shorten the video to exact requested duration before tracking/refinement. The natural retained timeline must stay intact until the finalizer.

## Port map

| From | Output | To | Input | Why |
|---|---|---|---|---|
| H3 Continuum Sampler V3.4 | `video_latents` | H3 Continuum Face Refine | `video_latents` | Pass-1 video samples and exact Native Masked protection mask |
| H3 Continuum Sampler V3.4 | `audio_latents` | H3 Continuum Face Refine | `audio_latents` | Pass-1 audio, locked bit-exact through sampler 2 |
| H3 Continuum Sampler V3.4 | `assembly_plan` | H3 Continuum Face Refine | `assembly_plan` | Physical-group topology, context trim and retained frame order |
| H3 Continuum Sampler V3.4 | `refine_state` | H3 Continuum Face Refine | `refine_state` | Exact chunk-local MODEL and positive CONDITIONING |
| H3 Continuum Assemble + Seam V3.4 | `images` | H3 Face Track + Crop | `images` | Natural retained full-frame timeline |
| H3 Face Track + Crop | `crops` | H3 Continuum Face Refine | `crops` | Global tracked crop timeline |
| H3 Face Track + Crop | `transform` | H3 Continuum Face Refine | `transform` | Boxes, detection mask, face sizes and stitch coordinates |
| MiniMax H3 video VAE | `VAE` | H3 Continuum Face Refine | `vae` | Physical-group encode and paired clean/refined decode |
| RandomNoise | `NOISE` | H3 Continuum Face Refine | `noise` | Sampler-2 video noise; audio noise is forced to zero |
| KSamplerSelect | `SAMPLER` | H3 Continuum Face Refine | `sampler` | Sampler 2 implementation |
| Separate BasicScheduler | `SIGMAS` | H3 Continuum Face Refine | `sigmas` | Sampler-2-only partial schedule; current baseline `simple`, 12 steps, 0.45 denoise |
| H3 Continuum Face Refine | `refined_crops` | H3 Face Stitch Back | `refined_crops` | Source-looking crops carrying only the learned residual correction |
| H3 Face Track + Crop | `transform` | H3 Face Stitch Back | `transform` | Exact float paste geometry and detection state |
| H3 Continuum Assemble + Seam V3.4 | `images` | H3 Face Stitch Back | `base_images` | Untouched natural-timeline source frames |
| H3 Face Stitch Back | `images` | H3 Continuum Finalize Duration V3.4 | `images` | Natural refined/stitched full frames |
| H3 Continuum Assemble + Seam V3.4 | `audio` | H3 Continuum Finalize Duration V3.4 | `audio` | Untouched assembled/generated or Driving Audio |
| H3 Continuum Sampler V3.4 | `assembly_plan` | H3 Continuum Finalize Duration V3.4 | `assembly_plan` | Exact target-frame/final-anchor/audio-duration policy |

## Sampler-2 schedule

Use a **separate BasicScheduler**. Do not reuse sampler-1 SIGMAS.

Current runtime-validated baseline for the Continuum path:

- scheduler: `simple`
- steps: **12**
- denoise: **0.45**

The earlier 4-step suggestion has been removed. GPU visual validation showed that four sampler-2 steps produced visible artifacts in this Continuum/Spectrum workflow. Treat 12 steps as the current quality baseline and only reduce it after direct visual A/B testing.

## Canvas size: use 512×512 first

For the **Continuum residual-transfer path**, **512×512 is the validated and recommended baseline**. Do not raise the canvas merely because the tracker reports crop magnification below 1.0x.

The tracker is shared with the original standalone FaceRefine workflow. In that older absolute-paste path, a sub-1x crop can mean that a lower-resolution reconstructed crop is pasted over higher-resolution source detail, so its generic warning about downscaling is meaningful there. Continuum now has a different stitch contract: it keeps the untouched full-resolution source frame and transfers only the VAE-cancelled learned residual. A sub-1x crop therefore does **not** imply that the higher-resolution source baseline will be replaced.

A current validated 512×512 GPU run had **125/294** crop frames below 1.0x and still worked correctly with residual stitch. Raising the same workflow to 704×704 removed that downscale statistic but made sampler 2 much more expensive without being required for the visual result. In those consecutive runs, `H3ContinuumFaceRefine` measured about **105.77 s at 512** versus **189.41 s at 704**. Treat those wall times as workflow observations rather than a controlled benchmark; the scaling direction is expected from the spatial latent area.

Approximate sampler-2 spatial token area relative to 512:

| Canvas | H3 spatial latent | Relative area |
|---:|---:|---:|
| 512 | 32×32 | 1.00× |
| 576 | 36×36 | 1.27× |
| 608 | 38×38 | 1.41× |
| 640 | 40×40 | 1.56× |
| 704 | 44×44 | 1.89× |

Use a larger canvas only when a direct visual A/B test shows a worthwhile refinement benefit on the actual clip. Do not use `auto_no_downscale` or raise the manual canvas simply to force the reported magnification to `>= 1.0x` on the Continuum path.

`H3 Continuum Face Refine` prints Continuum-specific canvas guidance before sampler 2. At 512 it explicitly states that sub-1x crops are allowed by the residual-stitch contract; at larger sizes it reports the approximate spatial-area cost relative to 512.

## Runtime-quality contract

### Missing-face frames remain context but are not refinement targets

Interpolated crop boxes are useful for keeping the timeline defined, but they are not evidence that a face exists. `H3 Continuum Face Refine` multiplies sampler-2 denoise strength by the tracker's exact per-frame real-face detection mask. Missing-face frames stay in temporal context with denoise mask zero.

Stitch-time `fade_out` is causal: still-valid frames before a dropout remain fully refined. Missing frames preserve the source exactly; reacquisition may fade in after the face returns.

### MiniMax H3 VideoVAE round-trip error is cancelled

For each physical H3 group, FaceRefine has the clean encoded crop latent before sampler 2. After sampling, clean and refined latents are decoded together as one paired VAE batch:

```text
clean_rgb   = decode(clean_latent)
refined_rgb = decode(refined_latent)
corrected_crop = source_crop + (refined_rgb - clean_rgb)
```

This removes crop VideoVAE reconstruction error from the public correction. The crop handed to Stitch Back differs from the tracked source crop only by sampler 2's learned latent-space change.

The paired decode is intentionally one VAE invocation rather than two separate calls. It does add the clean-reference decode work required to cancel the round-trip error, so FaceRefine node time can increase versus an absolute single-decode implementation; that cost is the explicit quality tradeoff and is kept isolated from the transformer sampling pass.

The runtime report prints `vae_roundtrip_cancelled=True` for every physical group.

### Stitch Back transfers only the residual

Stitch Back reconstructs the exact source crop with the same float affine geometry used by tracking and computes:

```text
residual = corrected_crop - exact_source_crop
output   = untouched_source + warp(residual) * mask * opacity
```

It never pastes an absolute 512/768px crop over the full-resolution source.

There is **no magnification hard gate**. A crop that was downscaled into the FaceRefine canvas can still contribute the learned residual without replacing the higher-resolution source baseline. This avoids spending a full sampler-2 pass only to discard every frame on close-up clips.

The residual inverse warp uses zero outside support because zero there means **no correction**, not black image pixels. Face rectangles keep fractional-pixel coverage, and normal feathering is applied in source-pixel space after the inverse warp.

`colour_match` in this path removes only per-channel residual DC bias. It does not perform absolute patch mean/std replacement.

The stitch log prints:

```text
stitch mode: VAE-roundtrip-cancelled residual transfer; no magnification hard gate
```

## Physical groups and continuation

`assembly_plan.decode_groups` is authoritative. Public `video_latents`, `audio_latents`, and `refine_state` are parallel to physical decode groups, not necessarily logical chunks.

- ordinary groups are sampled once each;
- terminal FL2VA merged tails remain one physical sampler-2 call;
- Native Masked zeros remain protected because FaceRefine strength multiplies the inherited mask;
- sampler 2's refined latent tail is carried into the next fully protected prefix before the next physical group is sampled.

No temporal trim/pad fallback is used to hide mismatched topology.

## Audio

Sampler-2 audio is locked by all three invariants:

- audio noise = zero;
- audio denoise mask = zero;
- pass-1 audio restored after sampling and checked for exact equality.

Keep Continuum's assembled/driving audio on the finalizer/save path.

## Conditioning and model patches

The captured MODEL clone and positive CONDITIONING remain authoritative. The path preserves:

- text/Qwen state;
- all MiniMax H3 references;
- Guide / Motion Context;
- First/Last visual conditioning, with target-grid visual keyframes rebound to matching crop frames;
- Spectrum, DiffAid, Untwist, and other captured model patches.

Only the public `transformer_options.h3_refinement` API-v1 marker is added to the sampler-2 clone.

## Run Storage

`refine_state` contains live MODEL/CONDITIONING objects and is only available for physical groups sampled in the current execution. Use Run Storage **Off**, or regenerate from **Chunk 1** when refinement state capture is required.

## GPU validation matrix

Validate at least:

| Case | Required observation |
|---|---|
| ordinary 2×6 s | two physical sampler-2 groups, natural-frame output order preserved |
| terminal 2×5 s | one terminal-merged physical sampler-2 call |
| 3+ chunk 5 s tail | ordinary leading group(s) plus one merged terminal tail |
| Native Masked | inherited zero mask remains zero and refined prefix is carried |
| Guide / Motion Context | captured guide/reference state preserved |
| multiple refs + First/Last | all refs preserved; target visual keyframes rebound to crop frames |
| generated / Driving Audio | pass-1 audio remains bit-exact through sampler 2 |
| Spectrum + DiffAid + Untwist | captured patch stack retained; refinement API marker active only for sampler 2 |
| detector dropout | missing-face frames show zero sampler-2 strength and no pre-fade before loss |
| close-up/downscaled crop | 512 remains the baseline; refinement is retained as residual with no absolute low-resolution patch replacement |
| larger canvas A/B | any visual gain must justify the spatial-token/runtime increase relative to 512 |
| shot boundary | no moving rectangular line and no pre-cut texture degradation from VAE/crop reconstruction |

The CPU suite validates the contracts and invariants above; final visual quality still requires the GPU cases.
