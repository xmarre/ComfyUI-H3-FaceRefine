from __future__ import annotations

"""Repair InsightFace's ONNX Runtime backend on CUDA ComfyUI hosts.

Several ComfyUI nodes (including InsightFace consumers and WD14 Tagger) legitimately
declare the distribution named ``onnxruntime``.  Pip treats that CPU distribution and
``onnxruntime-gpu`` as separate projects even though both install the same import
package.  Dependency reconciliation can therefore overwrite a working GPU module with
CPU files.

The durable invariant is not "CPU metadata must be absent".  Other installed nodes may
need that metadata to remain satisfied.  Instead, when CUDA is available and the live
module has lost CUDAExecutionProvider, reinstall the GPU distribution *last* and verify
its provider set in a fresh interpreter.  ``prestartup_script.py`` runs this same check
after Manager dependency reconciliation on every ComfyUI launch.
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
    """Probe PyTorch CUDA in a child interpreter to avoid prestartup CUDA side effects."""
    code = (
        "import torch; "
        "print('1' if torch.cuda.is_available() and torch.version.cuda else '0')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    if proc.returncode != 0:
        return False
    lines = proc.stdout.strip().splitlines()
    return bool(lines and lines[-1].strip() == "1")


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
    gpu_version: str | None,
) -> bool:
    if not cuda_host:
        return False
    return gpu_version is None or CUDA_PROVIDER not in providers


def _run_pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def _gpu_install_args(gpu_version: str | None) -> tuple[str, ...]:
    requirement = GPU_DIST if gpu_version is None else f"{GPU_DIST}=={gpu_version}"
    # Reinstall the GPU payload last.  Keeping plain onnxruntime's dist-info when some
    # other node requires it prevents Manager from reintroducing CPU files next launch.
    return ("install", "--force-reinstall", "--no-deps", requirement)


def main() -> int:
    # Fast healthy path: probing ORT is much cheaper than probing PyTorch CUDA during
    # every startup. A verified GPU distribution/provider needs no further work.
    providers, probe_error = _fresh_providers()
    gpu_version = _dist_version(GPU_DIST)
    if gpu_version is not None and CUDA_PROVIDER in providers:
        print(
            "[H3FaceRefine install] InsightFace ONNX Runtime GPU verified: "
            f"providers={providers} onnxruntime-gpu={gpu_version}"
        )
        return 0

    if not _cuda_host():
        print("[H3FaceRefine install] non-CUDA host: leaving ONNX Runtime unchanged")
        return 0

    cpu_version = _dist_version(CPU_DIST)
    if not _needs_gpu_repair(
        cuda_host=True,
        providers=providers,
        gpu_version=gpu_version,
    ):
        return 0

    print(
        "[H3FaceRefine install] repairing InsightFace ONNX Runtime GPU payload: "
        f"onnxruntime={cpu_version or 'absent'} "
        f"onnxruntime-gpu={gpu_version or 'absent'} "
        f"providers={providers or 'unavailable'}"
    )
    if probe_error:
        print(f"[H3FaceRefine install] provider probe detail: {probe_error}")

    _run_pip(*_gpu_install_args(gpu_version))

    repaired, repaired_error = _fresh_providers()
    if CUDA_PROVIDER not in repaired:
        detail = f"; probe error={repaired_error}" if repaired_error else ""
        raise RuntimeError(
            "H3 FaceRefine reinstalled onnxruntime-gpu, but CUDAExecutionProvider is "
            f"still unavailable (providers={repaired}){detail}. Refusing silent CPU "
            "InsightFace fallback."
        )

    print(
        "[H3FaceRefine install] GPU ONNX Runtime repaired and verified: "
        f"providers={repaired} onnxruntime-gpu={_dist_version(GPU_DIST)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
