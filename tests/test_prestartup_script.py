from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _stage_prestartup(tmp_path: Path, installer_source: str) -> Path:
    script = tmp_path / "prestartup_script.py"
    shutil.copyfile(ROOT / "prestartup_script.py", script)
    (tmp_path / "install.py").write_text(installer_source, encoding="utf-8")
    return script


def test_prestartup_executes_sibling_installer_main(tmp_path):
    script = _stage_prestartup(
        tmp_path,
        "from pathlib import Path\n"
        "def main():\n"
        "    Path(__file__).with_name('called.txt').write_text('yes')\n"
        "    return 0\n",
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "called.txt").read_text() == "yes"


def test_prestartup_propagates_repair_failure(tmp_path):
    script = _stage_prestartup(
        tmp_path,
        "def main():\n"
        "    raise RuntimeError('repair failed deliberately')\n",
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "GPU ONNX Runtime repair failed" in combined
    assert "repair failed deliberately" in combined
