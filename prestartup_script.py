from __future__ import annotations

"""Reassert FaceRefine's GPU ONNX Runtime invariant after Manager dependency setup.

ComfyUI runs custom-node ``prestartup_script.py`` files after the Manager prestartup
phase and before normal custom-node imports.  That is exactly the boundary needed here:
other nodes may have just reconciled a plain ``onnxruntime`` dependency, while no
InsightFace/WD14 ONNX session should have been created yet.
"""

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_INSTALL = _ROOT / "install.py"


def _load_installer():
    spec = importlib.util.spec_from_file_location(
        "h3_face_refine_ort_install", _INSTALL
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FaceRefine ORT installer from {_INSTALL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_prestartup_repair() -> None:
    installer = _load_installer()
    try:
        installer.main()
    except Exception as exc:
        print(f"[H3FaceRefine prestartup] GPU ONNX Runtime repair failed: {exc}")
        raise


_run_prestartup_repair()
