# InsightFace ONNX Runtime on CUDA hosts

`insightface` declares the CPU-named `onnxruntime` distribution as a hard dependency. Other ComfyUI extensions can do the same; notably, ComfyUI-WD14-Tagger includes plain `onnxruntime` in its requirements. Pip treats `onnxruntime` and `onnxruntime-gpu` as separate distributions even though both install the same `onnxruntime` Python module tree. A later dependency reconciliation can therefore overwrite a previously working GPU module with CPU files and remove `CUDAExecutionProvider`.

For a mixed ComfyUI installation, the durable invariant is the **live provider set**, not pip distribution ownership. If a fresh interpreter already reports `CUDAExecutionProvider`, FaceRefine leaves that runtime untouched even when no `onnxruntime-gpu` pip metadata exists (for example, a working conda-managed installation). Plain `onnxruntime` metadata may also remain when another node requires it.

Two lifecycle hooks enforce that invariant:

- `install.py` runs when ComfyUI-Manager actually installs or updates FaceRefine. If a CUDA host lacks `CUDAExecutionProvider`, it force-reinstalls the detected installed `onnxruntime-gpu` version with `--no-deps`; if no pip GPU distribution is registered, it installs `onnxruntime-gpu` without a version pin. It then verifies providers in a fresh interpreter.
- `prestartup_script.py` runs on every ComfyUI launch **after Manager's own prestartup/dependency work and before normal custom-node imports**. Healthy GPU environments take a provider-probe fast path and do not run pip. If another dependency has overwritten the module payload, the GPU wheel is installed last before InsightFace or WD14 can create an ONNX session.

No ONNX Runtime release number is hardcoded by FaceRefine. Existing managed GPU installs retain their detected version during repair; unmanaged working CUDA runtimes are left alone; missing managed GPU installs use the package index's current unpinned `onnxruntime-gpu` candidate.

The runtime tracker has a final safety invariant. When identity tracking is actually required, a CUDA ComfyUI host with only `CPUExecutionProvider` is a hard configuration failure. It is checked before `tracking_fixes` enters its recoverable optional-InsightFace exception block, so a broken GPU backend cannot silently degrade the run to motion-only subject tracking. Native CPU hosts remain supported. Deliberate CPU identity execution on a CUDA host can be enabled with `H3FACEREFINE_ALLOW_CPU_IDENTITY=1`.

Manual repair, when needed, mirrors the self-healing policy without a release pin:

```bash
python -m pip install --force-reinstall --no-deps onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

The final provider list must contain `CUDAExecutionProvider` before identity-aware FaceRefine tracking is used.
