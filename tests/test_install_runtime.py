from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3_face_refine_install_runtime_test", ROOT / "install.py"
)
assert SPEC is not None and SPEC.loader is not None
INSTALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL)


def test_cuda_provider_is_healthy_even_if_cpu_distribution_metadata_exists():
    assert INSTALL._needs_gpu_repair(
        cuda_host=True,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        gpu_version="1.28.0",
    ) is False


def test_missing_cuda_provider_requires_repair_on_cuda_host():
    assert INSTALL._needs_gpu_repair(
        cuda_host=True,
        providers=["AzureExecutionProvider", "CPUExecutionProvider"],
        gpu_version="1.28.0",
    ) is True


def test_cpu_host_is_left_unchanged():
    assert INSTALL._needs_gpu_repair(
        cuda_host=False,
        providers=["CPUExecutionProvider"],
        gpu_version=None,
    ) is False


def test_gpu_repair_reinstalls_existing_gpu_version_last_without_uninstalling_cpu(monkeypatch):
    monkeypatch.setattr(INSTALL, "_cuda_host", lambda: True)
    probes = iter(
        [
            (["AzureExecutionProvider", "CPUExecutionProvider"], None),
            (["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"], None),
        ]
    )
    monkeypatch.setattr(INSTALL, "_fresh_providers", lambda: next(probes))

    versions = {
        INSTALL.CPU_DIST: "1.28.0",
        INSTALL.GPU_DIST: "1.28.0",
    }
    monkeypatch.setattr(INSTALL, "_dist_version", lambda name: versions.get(name))

    commands = []
    monkeypatch.setattr(INSTALL, "_run_pip", lambda *args: commands.append(args))

    assert INSTALL.main() == 0
    assert commands == [
        (
            "install",
            "--force-reinstall",
            "--no-deps",
            "onnxruntime-gpu==1.28.0",
        )
    ]


def test_healthy_gpu_fast_path_does_not_import_cuda_host_probe(monkeypatch):
    monkeypatch.setattr(
        INSTALL,
        "_fresh_providers",
        lambda: (["CUDAExecutionProvider", "CPUExecutionProvider"], None),
    )
    monkeypatch.setattr(
        INSTALL,
        "_dist_version",
        lambda name: "1.28.0" if name == INSTALL.GPU_DIST else "1.28.0",
    )

    def should_not_run():
        raise AssertionError("healthy startup should not import torch just to detect CUDA")

    monkeypatch.setattr(INSTALL, "_cuda_host", should_not_run)
    monkeypatch.setattr(
        INSTALL,
        "_run_pip",
        lambda *args: (_ for _ in ()).throw(AssertionError("pip must not run")),
    )

    assert INSTALL.main() == 0


def test_failed_gpu_provider_verification_is_hard_error(monkeypatch):
    monkeypatch.setattr(INSTALL, "_cuda_host", lambda: True)
    probes = iter(
        [
            (["CPUExecutionProvider"], None),
            (["CPUExecutionProvider"], None),
        ]
    )
    monkeypatch.setattr(INSTALL, "_fresh_providers", lambda: next(probes))
    monkeypatch.setattr(
        INSTALL,
        "_dist_version",
        lambda name: "1.28.0" if name in {INSTALL.CPU_DIST, INSTALL.GPU_DIST} else None,
    )
    monkeypatch.setattr(INSTALL, "_run_pip", lambda *args: None)

    try:
        INSTALL.main()
    except RuntimeError as exc:
        assert "CUDAExecutionProvider is still unavailable" in str(exc)
    else:
        raise AssertionError("GPU repair must fail when CUDAExecutionProvider remains absent")
