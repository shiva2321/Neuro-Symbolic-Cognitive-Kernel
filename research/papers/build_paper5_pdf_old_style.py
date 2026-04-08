"""
Build paper5_transplantation.pdf using the legacy LaTeX style.

This keeps formatting aligned with the old paper PDFs by compiling
paper5_transplantation.tex directly instead of using the custom FPDF builder.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TEX = _HERE / "paper5_transplantation.tex"
_OUT = _HERE / "paper5_transplantation.pdf"


def _pick_engine() -> str | None:
    for name in ("pdflatex", "xelatex", "lualatex", "tectonic"):
        if shutil.which(name):
            return name

    # Windows fallback: MiKTeX can be installed but not yet added to this shell PATH.
    if sys.platform.startswith("win"):
        local_appdata = Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
        program_files = Path("C:/Program Files/MiKTeX/miktex/bin/x64")
        for root in (local_appdata, program_files):
            for exe in ("pdflatex.exe", "xelatex.exe", "lualatex.exe"):
                candidate = root / exe
                if candidate.exists():
                    return str(candidate)

    return None


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _cleanup_aux(tex_path: Path) -> None:
    stem = tex_path.stem
    for ext in (".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"):
        p = tex_path.with_name(stem + ext)
        if p.exists():
            p.unlink()


def build() -> int:
    if not _TEX.exists():
        print(f"[FATAL] Missing TeX source: {_TEX}")
        return 1

    engine = _pick_engine()
    if not engine:
        print("[FATAL] No LaTeX engine found (pdflatex/xelatex/lualatex/tectonic).")
        print("Install TeX Live or MiKTeX, then run this script again.")
        return 2

    print(f"[Build] Using engine: {engine}")
    try:
        if engine == "tectonic":
            _run([engine, _TEX.name], cwd=_HERE)
        else:
            # Two passes keep references/captions stable.
            _run([engine, "-interaction=nonstopmode", _TEX.name], cwd=_HERE)
            _run([engine, "-interaction=nonstopmode", _TEX.name], cwd=_HERE)
    except subprocess.CalledProcessError as exc:
        # MiKTeX on Windows sometimes returns exit code 1 even when PDF is created (update warning).
        if not _OUT.exists():
            # Only fail if PDF wasn't actually produced.
            print(f"[FATAL] LaTeX build failed with exit code {exc.returncode}")
            return exc.returncode or 3
        # PDF exists, treat this as success despite non-zero exit code.

    if not _OUT.exists():
        print(f"[FATAL] Expected output not found: {_OUT}")
        return 4

    _cleanup_aux(_TEX)
    print(f"[OK] PDF written -> {_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(build())




