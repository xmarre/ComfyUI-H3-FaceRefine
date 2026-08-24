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


def test_cuda_host_with_plain_cpu_distribution_requires_repair_even_if_cuda_visible():
    assert INSTALL._needs_gpu_repair(
        cuda_host=True,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        cpu_version="1.28.0",
        gpu_version="1.28.0",
    ) is True


def test_clean_gpu_runtime_needs_no_repair():
    assert INSTALL._needs_gpu_repair(
        cuda_host=True,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        cpu_version=None,
        gpu_version="1.28.0",
    ) is False


def test_cpu_host_is_left_unchanged():
    assert INSTALL._needs_gpu_repair(
        cuda_host=False,
        providers=["CPUExecutionProvider"],
        cpu_version="1.28.0",
        gpu_version=None,
    ) is False


def test_manager_post_install_removes_both_owners_then_installs_only_gpu(monkeypatch):
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
        ("uninstall", "-y", "onnxruntime", "onnxruntime-gpu"),
        ("install", "--no-deps", "onnxruntime-gpu"),
    ]


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
        lambda name: "1.28.0" if name == INSTALL.CPU_DIST else None,
    )
    monkeypatch.setattr(INSTALL, "_run_pip", lambda *args: None)

    try:
        INSTALL.main()
    except RuntimeError as exc:
        assert "CUDAExecutionProvider is still unavailable" in str(exc)
    else:
        raise AssertionError("GPU repair must fail when CUDAExecutionProvider remains absent")
