from __future__ import annotations

"""ComfyUI-Manager post-install repair for InsightFace ONNX Runtime.

InsightFace declares the CPU-only ``onnxruntime`` distribution as a hard dependency.
On a CUDA ComfyUI host that package can overwrite the files provided by
``onnxruntime-gpu`` even when the GPU distribution was already installed. Manager
installs ``requirements.txt`` before executing this script, so this is the safe point
to restore an unambiguous GPU runtime after InsightFace's dependencies settle.
"""

from importlib import metadata
import json
import subprocess
import sys

CPU_DIST = "onnxruntime"
GPU_DIST = "onnxruntime-gpu"
CUDA_PROVIDER = "CUDAExecutionProvider"


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cuda_host() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and getattr(torch.version, "cuda", None))
    except Exception:
        return False


def _fresh_providers() -> tuple[list[str], str | None]:
    code = (
        "import json, onnxruntime as ort; "
        "print(json.dumps({'version': ort.__version__, "
        "'providers': ort.get_available_providers()}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        return [], detail or f"provider probe exited {proc.returncode}"
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return [str(p) for p in payload.get("providers", [])], None
    except Exception as exc:
        return [], f"could not parse ONNX Runtime provider probe: {exc}"


def _needs_gpu_repair(
    *,
    cuda_host: bool,
    providers: list[str],
    cpu_version: str | None,
    gpu_version: str | None,
) -> bool:
    if not cuda_host:
        return False
    if cpu_version is not None:
        return True
    return gpu_version is None or CUDA_PROVIDER not in providers


def _run_pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def main() -> int:
    if not _cuda_host():
        print("[H3FaceRefine install] non-CUDA host: leaving ONNX Runtime unchanged")
        return 0

    providers, probe_error = _fresh_providers()
    cpu_version = _dist_version(CPU_DIST)
    gpu_version = _dist_version(GPU_DIST)
    if not _needs_gpu_repair(
        cuda_host=True,
        providers=providers,
        cpu_version=cpu_version,
        gpu_version=gpu_version,
    ):
        print(
            "[H3FaceRefine install] InsightFace ONNX Runtime already GPU-clean: "
            f"providers={providers} onnxruntime-gpu={gpu_version}"
        )
        return 0

    print(
        "[H3FaceRefine install] repairing InsightFace ONNX Runtime: "
        f"onnxruntime={cpu_version or 'absent'} "
        f"onnxruntime-gpu={gpu_version or 'absent'} "
        f"providers={providers or 'unavailable'}"
    )
    if probe_error:
        print(f"[H3FaceRefine install] provider probe detail: {probe_error}")

    # Both distributions own the same ``onnxruntime`` module tree. Remove both first,
    # then install one owner. ``--no-deps`` avoids churning NumPy and other established
    # ComfyUI packages merely to replace the runtime backend.
    _run_pip("uninstall", "-y", CPU_DIST, GPU_DIST)
    _run_pip("install", "--no-deps", GPU_DIST)

    repaired, repaired_error = _fresh_providers()
    if CUDA_PROVIDER not in repaired:
        detail = f"; probe error={repaired_error}" if repaired_error else ""
        raise RuntimeError(
            "H3 FaceRefine installed onnxruntime-gpu, but CUDAExecutionProvider is still "
            f"unavailable (providers={repaired}){detail}. Refusing to leave a CUDA ComfyUI "
            "host on silent CPU InsightFace fallback."
        )

    print(
        "[H3FaceRefine install] GPU ONNX Runtime verified: "
        f"providers={repaired} onnxruntime-gpu={_dist_version(GPU_DIST)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
