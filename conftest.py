"""Minimal ComfyUI module stubs for the CPU-only repository test suite."""

from __future__ import annotations

import sys
import types


if "comfy" not in sys.modules:
    comfy = types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = type("NestedTensor", (), {})
    comfy.nested_tensor = nested
    sys.modules["comfy"] = comfy
    sys.modules["comfy.nested_tensor"] = nested

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = "models"
    folder_paths.get_filename_list = lambda _key: []
    folder_paths.get_full_path = lambda _key, _name: None
    sys.modules["folder_paths"] = folder_paths
