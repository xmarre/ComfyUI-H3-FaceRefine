"""ComfyUI-H3-FaceRefine

Per-frame face tracking, normalised cropping, AV-latent injection and feathered
stitch-back, so MiniMax H3 can re-generate small/distant faces at a useful scale
and be composited back into the source video.

Why this exists
---------------
H3 renders faces poorly when the head occupies a small fraction of the frame.
That is a property of head-size-in-frame, not of output resolution, so no
post-process upscaler fixes it. The workaround is to crop to the face, hand H3
a sequence where the face is large, refine at low denoise so the result stays
frame-aligned, then paste back.

The crop must be PER FRAME. A single fixed crop degenerates to a wide shot on a
push-in, which hands H3 exactly the small face it fails on.
"""

if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .quality_fixes import install_quality_fixes
    from .quality_fixes_v2 import install_quality_fixes_v2
else:  # pytest/importlib loading this custom-node root as a standalone module
    import importlib.util
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent

    def _load_sibling(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, _root / filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load sibling FaceRefine module {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _nodes_module = _load_sibling("h3_face_refine_nodes", "nodes.py")
    NODE_CLASS_MAPPINGS = _nodes_module.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _nodes_module.NODE_DISPLAY_NAME_MAPPINGS
    install_quality_fixes = _load_sibling(
        "h3_face_refine_quality_fixes", "quality_fixes.py"
    ).install_quality_fixes
    install_quality_fixes_v2 = _load_sibling(
        "h3_face_refine_quality_fixes_v2", "quality_fixes_v2.py"
    ).install_quality_fixes_v2

install_quality_fixes(NODE_CLASS_MAPPINGS)
install_quality_fixes_v2(NODE_CLASS_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
