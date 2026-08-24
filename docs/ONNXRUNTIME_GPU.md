# InsightFace ONNX Runtime on CUDA hosts

`insightface` declares the CPU-named `onnxruntime` distribution as a hard dependency. Other ComfyUI extensions can do the same; notably, ComfyUI-WD14-Tagger includes plain `onnxruntime` in its requirements. Pip treats `onnxruntime` and `onnxruntime-gpu` as separate distributions even though both install the same `onnxruntime` Python module tree. A later dependency reconciliation can therefore overwrite a previously working GPU module with CPU files and remove `CUDAExecutionProvider`.

For a mixed ComfyUI installation, the durable invariant is **not** that the CPU-named distribution metadata must be absent. Removing it can simply cause another node or Manager dependency check to install it again on the next launch. FaceRefine instead requires that the GPU distribution's module payload wins and that a fresh interpreter reports `CUDAExecutionProvider`.

Two lifecycle hooks enforce that invariant:

- `install.py` runs when ComfyUI-Manager actually installs or updates FaceRefine. If a CUDA host lacks `CUDAExecutionProvider`, it force-reinstalls the existing `onnxruntime-gpu` version (or installs it if absent) with `--no-deps`, then verifies providers in a fresh interpreter.
- `prestartup_script.py` runs on every ComfyUI launch **after Manager's own prestartup/dependency work and before normal custom-node imports**. Healthy GPU environments take a cheap provider-probe fast path and do not run pip. If another dependency has overwritten the module payload, the GPU wheel is reinstalled last before InsightFace or WD14 can create an ONNX session.

The runtime tracker has a final safety invariant. When identity tracking is actually required, a CUDA ComfyUI host with only `CPUExecutionProvider` is a hard configuration failure. It is checked before `tracking_fixes` enters its recoverable optional-InsightFace exception block, so a broken GPU backend cannot silently degrade the run to motion-only subject tracking. Native CPU hosts remain supported. Deliberate CPU identity execution on a CUDA host can be enabled with `H3FACEREFINE_ALLOW_CPU_IDENTITY=1`.

Manual repair, when needed, mirrors the self-healing policy. If `onnxruntime-gpu` is already installed, reinstall that same version last rather than deleting dependency metadata required by other nodes:

```bash
python -m pip install --force-reinstall --no-deps onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

The final provider list must contain `CUDAExecutionProvider` before identity-aware FaceRefine tracking is used.
