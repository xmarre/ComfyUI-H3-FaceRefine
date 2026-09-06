# ComfyUI-H3-FaceRefine-Plus

> **Plus fork:** This is the `xmarre` maintained Plus fork of [Carasibana/ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine). It preserves the upstream project's foundation while carrying additional features, integrations, fixes, and behavior that may intentionally diverge from upstream.

**A ComfyUI custom node set to refine and improve the quality of small faces in MiniMax H3 video.**

MiniMax H3 renders faces poorly when the head occupies a small fraction of the frame. This is a property
of head-size-in-frame, not of output resolution, so it persists at 720p and above. These nodes
detect the face on every frame, crop to it so it fills a canvas, let H3 re-generate it, and
composite the result back into the original video.

Modelled on [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s **FaceDetailer**,
adapted from stills to video.

---

## Results

Source on the left, refined on the right:

![source vs refined, side by side](screenshots/COMPARISON.gif)

Single frame at full resolution:

| Source | Refined |
|---|---|
| ![source frame](screenshots/INPUT_00001.png) | ![refined frame](screenshots/REFINED_00001.png) |

The reason it works is what H3 is handed. Instead of a distant head a few dozen pixels tall, it
gets the face tracked and normalized to fill the canvas:

| Crop in: what H3 sees | Crop out: what H3 returns |
|---|---|
| ![input crop](screenshots/CROPS_INPUT_00001.gif) | ![refined crop](screenshots/CROPS_00001.gif) |
| [full-res still](screenshots/CROPS_INPUT_00001.png) | [full-res still](screenshots/CROPS_00001.png) |

These two are what to watch for temporal behaviour: the box has to sit still on a moving subject,
or the refined face boils.

---

## Installation

Clone into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/xmarre/ComfyUI-H3-FaceRefine-Plus.git ComfyUI-H3-FaceRefine
```

Restart ComfyUI. The nodes appear under **MiniMax H3/Face Refine**.

### Requirements

**For the nodes themselves:**

| | |
|---|---|
| ComfyUI with MiniMax H3 support | H3's own nodes are **core** (`comfy_extras/nodes_minimax_h3.py`), not an add-on, you just need a build recent enough to have them |
| a face detector | e.g. `face_yolov8m.pt` in `models/ultralytics/bbox/`. The one thing you must supply yourself |

Python packages (`ultralytics`, `scipy`, `insightface`) install automatically from
`requirements.txt` / `pyproject.toml`.

> **A note on onnxruntime.** `insightface` and some other ComfyUI extensions declare the CPU-named
> `onnxruntime` distribution even on CUDA systems. Pip treats `onnxruntime` and
> `onnxruntime-gpu` as separate distributions even though both install the same Python module tree.
> FaceRefine therefore does **not** pin an ONNX Runtime release or require CPU package metadata to be
> removed. On install/update and again at ComfyUI prestartup it checks the providers dynamically; if
> a CUDA host has lost `CUDAExecutionProvider`, it reinstalls the already installed
> `onnxruntime-gpu` version last (or installs the current unpinned GPU package if none is installed)
> and verifies the provider in a fresh interpreter. See
> [`docs/ONNXRUNTIME_GPU.md`](docs/ONNXRUNTIME_GPU.md) for the repair policy and manual check.

**Additionally, to run the example workflows:**

| | |
|---|---|
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | video load and save |
| [ComfyUI-H3-NativeAudioLock](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow) | supplies `MiniMaxH3NativeAudioLock`, which drives lipsync. It ships inside that repository under `custom_nodes/ComfyUI-H3-NativeAudioLock`, so copy that folder into your own `ComfyUI/custom_nodes/` |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | only if loading H3 as GGUF |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | only for the SAM workflow (supplies `SAMLoader`) |
| a SAM model in `models/sams/` | as above |

Model lookups go through ComfyUI's `folder_paths`, so anything registered in
`extra_model_paths.yaml` is found automatically.

### Models

| Model | Goes in | Source |
|---|---|---|
| `face_yolov8m.pt` | `models/ultralytics/bbox/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/face_yolov8m.pt) |
| `person_yolov8m-seg.pt` *(optional, for `fallback_detector`)* | `models/ultralytics/segm/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/person_yolov8m-seg.pt) |
| MiniMax H3 diffusion model | `models/diffusion_models/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Qwen3-VL text encoder | `models/text_encoders/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors` *(optional, big speed-up)* | `models/loras/` | [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras), distilled by [lightx2v](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| `sam_vit_b_01ec64.pth` *(SAM workflow only)* | `models/sams/` | [Meta segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| InsightFace `buffalo_l` *(crowd tracking)* | `models/insightface/` | downloaded automatically on first use |

H3's original weights are [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3); the
Comfy-Org repository above is the repackaged form ComfyUI expects. The example workflows load the
model and text encoder as **GGUF** through [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF),
which is what lets H3 fit in 12 GB. Swap those two loaders for the stock `UNETLoader` and
`CLIPLoader` if you have VRAM for full-precision weights.

Only the face detector is genuinely required by the nodes themselves. Everything else is needed
because the pipeline runs H3, so if you are already generating H3 video you will have it already.

---

## Quick start

Two example workflows are included, both annotated in-graph:

| Template | Paste mask | Extra dependency |
|---|---|---|
| **H3 Face Refine** | dilated + blurred rect | none |
| **H3 Face Refine SAM** | true face-shaped (SAM) | Impact Pack |

Both are installed as ComfyUI **workflow templates**. Open them from
**Workflow → Browse Templates** once the pack is installed, no file hunting required.
They also live in [`example_workflows/`](example_workflows) if you would rather load the JSON
directly.

**Start with the rect one.** Load it, then set:

1. **Source clip**: point the loader at your video.
2. **Ref image(s)**: the same character references you generated the clip with.
3. **Vocals**: an isolated vocal track, for lipsync.
4. **Prompt**: the clip's own prompt.
5. **`length`** on the H3 node: your clip's exact frame count.

Everything else can stay at the shipped values for a first run.

```
source clip ───┬──────────────────────────────────────────────────────┐
               ▼                                                      │
      H3 Face Track + Crop ──┬── crops ─────────────────┐             │
               │             │                          ▼             │
               │             │     MiniMaxH3ReferenceToVideo          │
               │             │        (refs; W/H <- canvas_w/h)       │
               │             │                          │             │
               │             └──────────► H3 Inject Video Latent      │
               │                                        │             │
               │                 MiniMaxH3NativeAudioLock <- vocals   │
               │                                        │             │
               │                        H3 Per-Frame Denoise          │
               │                                        │             │
               │              SamplerCustomAdvanced ─► VAEDecode      │
               │                                        │             │
               └── transform ─────────► H3 Face Stitch Back ◄─────────┘
                                                │
                                  original audio ──► save
```

The `transform` output is the spine of the whole thing: it records where every crop came from, so
the stitch can put each refined face back exactly where it belongs.

### H3 Continuum sampler-2 integration

Use **H3 Continuum Face Refine** when the source was generated by H3 Continuum V3.4. It consumes
Continuum's exact per-physical-sample MODEL, CONDITIONING, latent, mask, and assembly state and
runs the second H3 sampling pass internally. It does not create another
`MiniMaxH3ReferenceToVideo`, CLIP path, or reference-image path.

The source video must be assembled on Continuum's **natural retained timeline** before it is sent
to **H3 Face Track + Crop**. Select
`Timeline Output = Natural retained timeline (Refinement)` on the V3.4 assembler. After stitch-back,
connect the images, untouched assembler audio, and the same assembly plan to
**H3 Continuum Finalize Duration V3.4**. This keeps the globally smoothed face track and every
ordinary or terminal-merged physical H3 group on the same frame topology, then applies Continuum's
validated target-frame, final-frame, and audio-duration policy once.

```text
H3 Continuum Sampler V3.4
  video_latents ───────────────────────────────┐
  audio_latents ───────────────────────────────┤
  assembly_plan ───────────────────────────────┤
  refine_state ────────────────────────────────┤
                                                ▼
natural assembled images → H3 Face Track + Crop → H3 Continuum Face Refine
                              │ crops + transform       ▲  ▲  ▲
                              │                         │  │  └─ low-sigma SIGMAS
                              │                         │  └──── SAMPLER
                              │                         └─────── NOISE + video VAE
                              └───────────────────────────────► H3 Face Stitch Back
                                                                    │
assembler audio + assembly_plan ─────────────────────► H3 Continuum Finalize Duration V3.4
```

Important operating rules:

- Connect `refine_state`; its link makes Continuum capture the exact sampler-1 state. With Run
  Storage, use **Off**, or **Save + Auto Resume** with **Regenerate From = Chunk 1** for the refine
  run. Reused chunks intentionally have no serializable runtime MODEL/CONDITIONING state.
- Add a **second BasicScheduler dedicated to sampler 2**; do not reuse Continuum's pass-1 `SIGMAS`.
  The current runtime-validated Continuum baseline is `scheduler=simple`, **12 steps**,
  `denoise=0.45`. The former 4-step suggestion is intentionally removed: GPU visual testing showed
  obvious artifacts at four sampler-2 steps in the Continuum/Spectrum path.
- `video_latents` is required because its `noise_mask` protects Native Masked context. Face strength
  multiplies that mask, so an inherited zero remains exactly zero. The actually refined crop-latent
  tail is carried into the next protected crop prefix.
- Frames where the tracker has no real face detection remain temporal context, but sampler-2
  denoise strength is forced to zero for those frames. An interpolated crop path is not treated as
  evidence that a face exists.
- For every physical group, clean and refined crop latents are decoded together. FaceRefine cancels
  the MiniMax H3 VideoVAE round-trip baseline before stitch-back:
  `source_crop + (decode(refined_latent) - decode(clean_latent))`. The clean/reference decode adds
  real VAE work, but clean and refined are decoded in one paired invocation to avoid two separate
  VAE load/launch paths.
- On the Continuum path **H3 Face Stitch Back transfers only the residual correction** onto the
  untouched full-resolution source. It does not paste an absolute 512/768px crop. There is no
  magnification hard gate, so a downscaled crop can still contribute the learned correction without
  replacing higher-resolution source pixels.
- Audio receives zero sampler-2 noise, a zero denoise mask, and is restored from pass 1 after the
  sampler. **H3 Continuum Face Refine** outputs images only; keep Continuum's assembled/driving
  audio on the save path.
- Guide / Motion Context, multiple references, hybrid First/Last conditioning, Qwen state, and
  model patches such as Spectrum, DiffAid, and Untwist remain attached to the captured state.
  Target-grid visual keyframes are re-encoded from the corresponding tracked crop frame.
- The node uses rectangular transform masks and needs no SAM model. Stock FaceRefine nodes and
  standalone workflows are unchanged and still load without H3 Continuum installed.

Expected Continuum runtime diagnostics include:

```text
VAE baseline: clean/refined latents decoded as one pair; RGB round-trip error cancelled
...
vae_roundtrip_cancelled=True
...
stitch mode: VAE-roundtrip-cancelled residual transfer; no magnification hard gate
```

See [`docs/H3_CONTINUUM_WORKFLOW_FRAGMENT.md`](docs/H3_CONTINUUM_WORKFLOW_FRAGMENT.md) for the
complete port map, terminal-group behavior, residual-stitch contract, and validation checklist.

---

# Node reference

## H3 Continuum Face Refine

Runs crop-space sampler 2 once for each **physical** H3 sample in Continuum's assembly plan, then
returns one globally ordered crop batch for **H3 Face Stitch Back**. Ordinary chunks remain one
physical group each. Continuum's final 2×5-second or 3+-chunk terminal tail remains the single
physical sample recorded by the plan rather than being sampled again as separate logical chunks.

**Inputs**

| Input | What to connect |
|---|---|
| `crops`, `transform` | The two matching outputs of **H3 Face Track + Crop**, run once over the natural assembled video |
| `video_latents`, `audio_latents`, `assembly_plan`, `refine_state` | Same-named outputs from **H3 Continuum Sampler V3.4** |
| `vae` | MiniMax H3 video VAE |
| `noise` | `RandomNoise` or another ComfyUI `NOISE` provider |
| `sampler` | `KSamplerSelect` or another ComfyUI `SAMPLER` provider |
| `sigmas` | A separate sampler-2 `BasicScheduler`; current Continuum baseline is `simple`, **12 steps**, `0.45` denoise; never Continuum's pass-1 `SIGMAS` |
| face-strength controls | Same meaning as **H3 Per-Frame Denoise**, evaluated and smoothed across the full assembled timeline |

**Outputs:** `refined_crops` to **H3 Face Stitch Back**, plus a report containing physical-group,
trim, terminal-merge, carried-prefix, detection-gate, VAE-roundtrip-cancellation, and audio-lock
diagnostics.

The node fails closed on stale/misaligned refine state, incomplete physical-group coverage, unknown
contract versions, non-uniform inherited spatial masks, frame/latent topology changes, and
exact-duration input that was shortened before tracking. It never temporally trims or pads a crop
group to make mismatched data fit. Final duration is applied by Continuum's dedicated finalizer.

## H3 Face Track + Crop

<img src="screenshots/H3%20Face%20Track%20%2B%20Crop.png" alt="H3 Face Track + Crop" width="380">

Detects the face on every frame, fills gaps where detection fails, smooths the trajectory, and
emits a constant-size batch of crops plus the `transform` needed to paste results back.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `images` | - | The clip to refine. Connect your video loader. |
| `detector` | first found | Face detection model from `models/ultralytics/`. `face_yolov8m.pt` is a good default. |
| `confidence` | `0.35` | Detection threshold. Lower catches more profiles and partial faces at the cost of false positives. |
| `crop_factor` | `2.5` | Crop side as a multiple of face **height**. 2.5 puts the face at ~40% of the crop. Bigger gives more context so the seam lands in hair and background, but less magnification. **2.0-3.0 is the useful range.** |
| `canvas_width` / `canvas_height` | `512` | Resolution H3 generates at. Ignored unless `canvas_mode` is `manual`. 768 is H3's native short edge and gives the best faces, at 2.25× the cost of 512. |
| `canvas_mode` | `manual` | `manual` uses the two values above. `auto_no_downscale` sizes from the largest crop so no frame is ever downscaled. `auto_capped_768` does the same but clamps to 768, a sane VRAM ceiling and the best default. |
| `smooth_window` | `21` | Frames of smoothing on the crop **centre**. 21 at 24 fps is ~0.9 s. Raise if the box shivers, lower if it lags fast head movement. |
| `size_smooth_window` | `51` | Frames of smoothing on the crop **size**. Deliberately larger than the centre window, because size jitter makes the crop breathe, which changes the resample factor every frame and reads as shimmer. |
| `smooth_method` | `gaussian` | `gaussian` rejects jitter best. `savgol` preserves the shape of a push-in at large windows. `moving_average` is a plain boxcar and leaves residual jitter. |
| `size_mode` | `per_frame` | `per_frame` holds the face at a constant fraction of every crop, which is correct for push-ins. `max_of_clip` uses one size throughout, only useful when the shot is static. |
| `identity_reference` *(opt)* | - | A clear face image of the person to track. Picks the subject by identity rather than size. |
| `identity_track` *(opt)* | `True` | Hold one subject through a crowd. Continuity decides most frames; the identity embedding is consulted only when two candidates are similarly plausible or their boxes overlap. |
| `identity_threshold` *(opt)* | `0.28` | Minimum cosine similarity to accept a face as the reference person. Below it, the frame falls back to continuity, which is what carries tracking through profiles and occlusion. |
| `select` *(opt)* | `largest` | Used only when no `identity_reference` is connected, and as the first-frame tie-break. `largest` or `most_central`. |
| `fallback_detector` *(opt)* | `none` | Used only on frames where the face detector finds nothing. A person/body model gives a real head position from the top of the body box, which beats interpolating blindly. |
| `fallback_head_frac` *(opt)* | `0.5` | Head centre as a multiple of face height below the top of the person box. 0.5 suits a head seen from behind. |

**Outputs**

| Output | Goes to |
|---|---|
| `crops` | `H3 Inject Video Latent` → `images`, and `H3 Face Mask (SAM)` → `crops` |
| `transform` | `H3 Face Stitch Back`, `H3 Per-Frame Denoise`, `H3 Face Mask (SAM)` |
| `preview` | Optional: a debug view of the tracked boxes |
| `report` | Text summary: detections, gaps, magnification warnings |
| `canvas_w` / `canvas_h` | **Must** be wired to the H3 node's `width` / `height` |

> **Wire the canvas, don't type it.** In the `auto_*` modes this node chooses the size. If the H3
> node's `width`/`height` disagree, the latent shapes differ and injection refuses.

> Watch `report` for `magnification < 1.0x`. That means the crop is being *downscaled* into the
> canvas and real crop detail is discarded. Raising the canvas remains preferable when practical.
> On the Continuum path, however, downscaled frames are **not disabled**: the round-trip-cancelled
> residual stitch preserves the full-resolution source baseline and transfers only the learned
> correction.

**Multiple people:** run the pipeline once per subject, each with that person's `identity_reference`
and their own refs on the H3 node, then chain them, feeding run 1's stitched output in as run 2's
`base_images`. The composites accumulate.

---

## H3 Inject Video Latent (img2img)

<img src="screenshots/H3%20Inject%20Video%20Latent%20(img2img).png" alt="H3 Inject Video Latent (img2img)" width="380">

Encodes real frames into the **video** stream of H3's joint audio-video latent, leaving the audio
stream intact. This is the missing img2img path: H3's stock nodes always build a zeros latent, because
references are conditioning re-injected each step, never a starting point. Without this there
is no video-to-video.

**Inputs**

| Input | What to connect |
|---|---|
| `av_latent` | The `LATENT` output of `MiniMaxH3ReferenceToVideo` |
| `images` | `crops` from **H3 Face Track + Crop** |
| `vae` | The **video** VAE |

**Outputs:** `av_latent` (onwards to `MiniMaxH3NativeAudioLock`) and `report`.

It has no widgets. Strength is set downstream by `BasicScheduler`'s `denoise`, **not** by
`SplitSigmas`. See [Denoise](#denoise) below.

---

## H3 Per-Frame Denoise

<img src="screenshots/H3%20Per-Frame%20Denoise.png" alt="H3 Per-Frame Denoise" width="380">

Varies denoise strength along the temporal axis via the latent's noise mask, so one sampling pass
covers a shot that goes from distant to close.

A single denoise cannot serve a whole clip: a tiny face has no detail to preserve and wants a
strong pass so the model *synthesizes*, while a large face has real detail and wants a gentle pass
so it is not rewritten. This node scales the base denoise per frame by measured face size.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `av_latent` | - | From `MiniMaxH3NativeAudioLock` |
| `transform` | - | From **H3 Face Track + Crop**. This is where face sizes come from |
| `strength_small_face` | `1.0` | Multiplier where the face is smallest. `1.0` = the full denoise set on `BasicScheduler`. |
| `strength_large_face` | `0.35` | Multiplier where the face is largest. Lower preserves the detail those frames already have. |
| `scale_mode` | `absolute_px` | `absolute_px` keys off real face size in source pixels, which is safe across a batch, since a clip that never has a small face just sits at the baseline. `relative_to_clip` normalizes to that clip's own min/max, so its smallest face always gets the full boost. Use the latter when tuning one clip to its extremes. |
| `face_px_small` | `30.0` | Face height (source px) at or below which full `strength_small_face` applies. |
| `face_px_large` | `120.0` | Face height (source px) at or above which `strength_large_face` applies. |
| `gamma` | `1.0` | Curve on the interpolation. `>1` keeps strength high until the face is genuinely large; `<1` drops it off early. |
| `smooth_frames` | `9` | Smooths the strength curve over time. An abrupt denoise change between neighbouring frames shows up as a texture pop, so be generous. |

**Outputs:** `av_latent` (to `SamplerCustomAdvanced` → `latent_image`) and `report`.

Base denoise and these multipliers are tuned **together**. The example standalone workflows ship a
base of `0.45` which this node scales down on large-face frames. If you bypass this node, drop the
base a long way or every large face gets rewritten.

---

### Denoise

Denoise values do **not** transfer from SDXL-family models. H3 is flow matching with a large sigma
shift:

```
sigma = shift * t / (1 + (shift - 1) * t)
```

At the default shift of 12, `0.25`, an ordinary FaceDetailer value, lands at an effective sigma
of **0.800** and rewrites the frame.

| denoise | effective sigma (shift 12) |
|---|---|
| 0.02 | 0.197 |
| 0.05 | 0.387 |
| 0.15 | ~0.66 |
| 0.25 | 0.800 |

`steps` and `denoise` are **independent**: `BasicScheduler` builds a `steps/denoise`-long
full-range schedule and keeps the lowest `steps+1` sigmas. Do **not** use `SplitSigmas`.

> The standalone Turbo example and the Continuum sampler-2 path are different regimes. The
> Continuum/Spectrum integration is currently validated at **12 steps / 0.45 denoise**; four-step
> sampler-2 refinement produced visible artifacts in GPU testing and is not recommended there.

Push the ceiling too high and the head drifts relative to the body, a content problem no mask can
hide.

---

## H3 Face Mask (SAM)

<img src="screenshots/H3%20Face%20Mask%20(SAM).png" alt="H3 Face Mask (SAM)" width="380">

Produces true face-shaped paste masks instead of a rectangle, temporally smoothed. Optional, and
requires Impact Pack for `SAMLoader`.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `crops` | - | **The tracker's `crops`, not the decoded result.** See the warning below. |
| `sam_model` | - | From Impact Pack's `SAMLoader` |
| `transform` | - | From **H3 Face Track + Crop** |
| `threshold` | `0.93` | SAM confidence threshold for accepting mask pixels. |
| `dilation` | `0` | Grow the mask. SAM masks are accurate, so they rarely need it. |
| `temporal_smooth` | `5` | Frames of averaging across the mask stack. `1` disables it, and the mask edge will shimmer. |

**Outputs:** `masks` (to `H3 Face Stitch Back` → `masks`) and `report`.

> **Mask the input, never the output.** Wire the tracker's crops here. This matches FaceDetailer,
> which computes its mask from the source image; generation never feeds back into the mask. Mask
> the *generated* result instead and, if the model nudges the face inward, the mask traces the new
> smaller silhouette while the original face pokes out past it, most visibly the nose on profile
> shots. It is also cheaper: no dependency on the sampler, so SAM need not be resident alongside
> the video model.

**Is it worth it?** Often not. A rect mask frequently beats SAM here. SAM traces the face tightly,
so any drift in the refined face lands right on the silhouette, whereas a slightly looser rect puts
the seam in hair and background where it reads far less. Try the rect workflow first.

> If you use SAM masks, drop `feather` on the stitch node to **4-8**. A rect needs far more.

---

## H3 Face Stitch Back

<img src="screenshots/H3%20Face%20Stitch%20Back.png" alt="H3 Face Stitch Back" width="380">

Warps each refined crop back onto the exact float box it came from and composites through the
selected face mask.

> **Continuum path:** the installed Continuum-aware implementation is residual-only. It reconstructs
> the exact source crop, subtracts that from the VAE-roundtrip-cancelled FaceRefine crop, and warps
> only the learned correction onto the untouched source frame. It does not paste an absolute crop,
> and it has no magnification hard gate. The standalone path retains the original FaceDetailer-style
> absolute crop composite described below.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `base_images` | - | The **original** frames, the same clip fed to the tracker |
| `refined_crops` | - | FaceRefine crop output |
| `transform` | - | From **H3 Face Track + Crop** |
| `paste_region` | `face_only` | What actually composites. `face_only` / `face_ellipse` paste just the detected face box; `full_crop` covers hair, shoulders and background too. |
| `mask_dilation` | `16` | Grow the face box before blurring, in canvas px. |
| `feather` | `6` | Gaussian blur radius on the paste mask, in **source pixels**. Use ~24 with a rect mask, 4-8 with SAM. |
| `colour_match` | `1.0` | Standalone: patch colour match. Continuum residual path: removes only per-channel residual DC bias rather than remapping an absolute patch. |
| `blend` | `1.0` | Global opacity of the refinement correction. |
| `undetected_frames` | `fade_out` | Controls compositing where no face was found. On the Continuum path those frames are already protected from sampler-2 denoise; fade-in after reacquisition is causal and valid frames are never pre-faded before a loss. |
| `feather_scales_with_crop` *(opt)* | `False` | Legacy canvas-relative feather behavior. Leave off. |
| `masks` *(opt)* | - | Per-frame masks from **H3 Face Mask (SAM)**. Overrides `paste_region`. |

**Outputs:** `images`, the finished frames. Send to your video save/finalizer node.

> **Only the face region composites.** The wide crop exists to give the sampler *context*. It is
> not what gets pasted. Pasting the whole crop covers roughly 88% of the canvas versus 16% for the
> face box, and any change the model made to hair or background returns as a rectangle.

---

## H3 Face Transform Info

<img src="screenshots/H3%20Face%20Transform%20Info.png" alt="H3 Face Transform Info" width="380">

A debug node. Prints the per-frame boxes so you can sanity-check tracking before spending time on a
sampling pass.

| Input | Default | What it does |
|---|---|---|
| `transform` | - | From **H3 Face Track + Crop** |
| `max_rows` | `12` | How many frames to print. |

**Outputs:** `info` (a string). It is an output node, so it displays in the graph directly.

Use it when the stitch looks misaligned, or to confirm gap-filling behaved on a clip where the
subject turns away.

---

## Lipsync

H3 is a **joint** audio-video model. `MiniMaxH3NativeAudioLock` encodes real audio into the audio
stream of the AV latent, sets `noise_mask` to ones for video and zeros for audio so only video
denoises, and the video branch cross-attends to that fixed audio. That is what shapes the mouth.

Feed it an isolated **vocals** track for a cleaner signal. The **original** audio goes separately
into the save node. Two distinct audio paths, easy to confuse.

---

## Gotchas

- **Frame count must sit on H3's 17k+5 grid** (5, 22, 39 … 175, 226, 362). Clips generated by H3
  already do.
- **Cost is `canvas² × frames`.** Auto canvas sizing considers face size only, so on a long clip it
  can pick a canvas that exceeds VRAM and falls back to streaming weights from system RAM, an
  order-of-magnitude slowdown rather than a clean error.
- **`SAMLoader`'s `AUTO` device mode leaves the model on CPU** until `prepare_device()` is called.
  This pack calls it; if you write your own SAM path, missing it makes mask passes 10-50× slower
  with the GPU idle at half power.

---

## Credits

The compositing approach is taken directly from
**[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)** by
**[ltdrdata](https://github.com/ltdrdata)**, specifically `FaceDetailer` and its detailer paste.
The dilate-then-blur face-region mask, the crop-for-context-but-paste-only-the-face principle, and
the bbox+SAM masking path are all its design; this pack adapts them to a per-frame video pipeline.
If you find this useful, that project is why.

The lipsync path in the example workflows depends on `MiniMaxH3NativeAudioLock`, from
**[MiniMax-H3-NativeAudio-MusicVideo-Workflow](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow)**
by **[Shrek3OnVH5](https://github.com/Shrek3OnVH5)**. It is not redistributed here, so install it
from that repository.

Also builds on:

- **[MiniMax H3](https://github.com/MiniMax-AI)**, the joint audio-video model being refined
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous
- **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by Kosinkadink
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** by city96
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** for detection, and
  **[InsightFace](https://github.com/deepinsight/insightface)** for identity embeddings

---

## Made with these nodes

[![More Than Words Can Ever Steal](https://img.youtube.com/vi/18iiffk-QWE/maxresdefault.jpg)](https://www.youtube.com/watch?v=18iiffk-QWE)

**[More Than Words Can Ever Steal](https://www.youtube.com/watch?v=18iiffk-QWE)**, a music video by
[Carasibana](https://www.youtube.com/@Carasibana-Music), generated with MiniMax H3. Every shot
where the subject sits at any distance from camera was face-refined with this pack.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

<sub>Built with [Claude Code](https://claude.com/claude-code).</sub>