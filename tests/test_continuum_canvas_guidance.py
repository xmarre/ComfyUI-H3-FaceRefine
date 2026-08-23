from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "quality_fixes_v2.py"
SPEC = importlib.util.spec_from_file_location("h3_face_refine_canvas_guidance_tests", MODULE_PATH)
assert SPEC and SPEC.loader
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


def _transform(canvas, heights):
    return {
        "canvas": tuple(canvas),
        "boxes": [(0.0, 0.0, float(h), float(h)) for h in heights],
    }


def test_512_is_validated_continuum_baseline_even_with_sub_1x_crops():
    message = quality.continuum_canvas_guidance(
        _transform((512, 512), [704.0, 400.0])
    )

    assert "512x512 validated baseline" in message
    assert "1/2 crop frame(s) are sub-1x" in message
    assert "residual stitch preserves the untouched full-resolution source" in message
    assert "do not raise the canvas just to eliminate that statistic" in message


def test_larger_canvas_reports_spatial_cost_relative_to_512():
    message = quality.continuum_canvas_guidance(
        _transform((704, 704), [704.0, 400.0])
    )

    assert "704x704 uses ~1.89x the spatial token area of 512x512" in message
    assert "512x512 is the validated baseline" in message
    assert "not merely to make magnification >= 1.0x" in message


def test_canvas_guidance_wrapper_prints_before_delegating(capsys):
    class Base:
        def refine(self, crops, transform, *args, **kwargs):
            print("BASE_REFINE")
            return (crops, "ok")

    wrapped = quality.build_continuum_canvas_guidance(Base)()
    result = wrapped.refine("crops", _transform((512, 512), [704.0]))

    assert result == ("crops", "ok")
    lines = capsys.readouterr().out.splitlines()
    assert "512x512 validated baseline" in lines[0]
    assert lines[1] == "BASE_REFINE"


def test_canvas_guidance_fails_soft_when_geometry_is_missing():
    assert "guidance unavailable" in quality.continuum_canvas_guidance({"boxes": []})
