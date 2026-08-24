# InsightFace ONNX Runtime on CUDA hosts

`insightface` declares the CPU-named `onnxruntime` distribution as a hard dependency. The CPU and GPU ONNX Runtime distributions install the same `onnxruntime` Python module tree, and pip does not treat `onnxruntime-gpu` as satisfying InsightFace's `onnxruntime` metadata requirement. A dependency refresh can therefore install the CPU distribution after a working GPU runtime and silently remove `CUDAExecutionProvider` from the imported module.

ComfyUI-Manager installs `requirements.txt` before executing this repository's `install.py`. On a CUDA host, `install.py` therefore checks the runtime in a fresh Python process after dependency installation. If the plain CPU distribution is installed, both ORT distributions are removed and a single `onnxruntime-gpu` distribution is installed with `--no-deps`, then `CUDAExecutionProvider` is verified. The `--no-deps` flag avoids unrelated NumPy/package churn in an established ComfyUI environment.

The runtime tracker also checks the provider immediately before InsightFace FaceAnalysis is constructed. A CUDA ComfyUI host with only `CPUExecutionProvider` is treated as a broken environment and identity tracking stops with a direct repair message instead of silently running the expensive CPU path. Native CPU hosts remain supported. Deliberate CPU identity execution on a CUDA host can be enabled with `H3FACEREFINE_ALLOW_CPU_IDENTITY=1`.

Manual repair uses the same ownership rule:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install --no-deps onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

The final provider list must contain `CUDAExecutionProvider` before ComfyUI is restarted.
