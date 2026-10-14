"""The computer's LaTeX installation: finding it, and typesetting inline
mathematics to SVG with its baseline.

Used by PyFigures for equation panels and for $…$ maths in labels. Nothing
here is needed without LaTeX: callers fall back to Matplotlib's mathtext.
"""
from __future__ import annotations

import glob
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _tex_bin_dirs():
    """Folders where TeX distributions put latex/dvisvgm when not on the PATH."""
    dirs = []
    if os.environ.get("PYFIGURES_TEX_BIN"):
        dirs.append(os.environ["PYFIGURES_TEX_BIN"])
    if sys.platform.startswith("win"):
        for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if base:
                dirs.append(os.path.join(base, "Programs", "MiKTeX", "miktex", "bin", "x64"))
                dirs.append(os.path.join(base, "MiKTeX", "miktex", "bin", "x64"))
        dirs += sorted(glob.glob(r"C:\texlive\*\bin\win*"), reverse=True)
    else:
        dirs += ["/Library/TeX/texbin", "/usr/local/texlive/bin", "/usr/bin", "/usr/local/bin"]
        dirs += sorted(glob.glob("/usr/local/texlive/*/bin/*"), reverse=True)
    return dirs


def find_tex_tool(name):
    """Full path of a TeX program (latex, pdflatex, dvisvgm), or None."""
    found = shutil.which(name)
    if found:
        return found
    exe = name + (".exe" if sys.platform.startswith("win") else "")
    for d in _tex_bin_dirs():
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    return None


def latex_available() -> bool:
    """True if latex and dvisvgm are installed."""
    return find_tex_tool("latex") is not None and find_tex_tool("dvisvgm") is not None


# ---------------------------------------------------------------------------
# inline maths (labels)
# ---------------------------------------------------------------------------
INLINE_PREAMBLE = r"\usepackage{amsmath,amssymb,bm,siunitx}"
# folders searched (in order) for already-typeset formulas, and written to:
# PyFigures puts the open figure's own folder first, so its formulas travel
# with it; the per-user cache is always last
CACHE_DIRS: list = []
# keys of the formulas drawn since the last bind_figure(): saved with the figure
USED: set = set()


def labels_dir_for(figure_path):
    """Where the formulas of the figure *figure_path* are kept."""
    return os.path.join(os.path.dirname(os.path.abspath(figure_path)), "equations", "labels")


def bind_figure(figure_path, reset=True):
    """Make *figure_path*'s folder the first cache (None: no figure). *reset*
    forgets which formulas were used (a new figure was opened)."""
    CACHE_DIRS[:] = [labels_dir_for(figure_path)] if figure_path else []
    if reset:
        USED.clear()


def save_used(figure_path):
    """Copy every formula drawn for the current figure next to *figure_path*
    (so they travel with it). Returns the number copied."""
    dest = labels_dir_for(figure_path)
    n = 0
    for key in sorted(USED):
        if os.path.isfile(os.path.join(dest, key + ".svg")):
            continue
        for d in list(CACHE_DIRS) + [user_cache_dir()]:
            svg = os.path.join(d, key + ".svg")
            meta = os.path.join(d, key + ".json")
            if os.path.isfile(svg) and os.path.isfile(meta):
                try:
                    os.makedirs(dest, exist_ok=True)
                    shutil.copyfile(svg, os.path.join(dest, key + ".svg"))
                    shutil.copyfile(meta, os.path.join(dest, key + ".json"))
                    n += 1
                except OSError:
                    pass
                break
    return n


def user_cache_dir():
    return os.path.join(os.path.expanduser("~"), ".pyfigures", "latex_labels")


# bumped whenever the document used to typeset a formula changes (new cache keys)
TYPESET_VERSION = 2


def _key(tex, size_pt, sans, color, preamble):
    raw = json.dumps([TYPESET_VERSION, tex, round(float(size_pt), 3), bool(sans), color.lower(),
                      preamble])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _lookup(key):
    for d in list(CACHE_DIRS) + [user_cache_dir()]:
        svg = os.path.join(d, key + ".svg")
        meta = os.path.join(d, key + ".json")
        if os.path.isfile(svg) and os.path.isfile(meta):
            try:
                with open(meta, encoding="utf-8") as f:
                    m = json.load(f)
                return svg, float(m["width"]), float(m["height"]), float(m["depth"])
            except (OSError, ValueError, KeyError):
                continue
    return None


def _store(key, svg_text, metrics):
    """Write a new formula to the user cache (and the figure's folder, if one
    is bound: PyFigures also copies the used ones there on save)."""
    paths = None
    for d in list(CACHE_DIRS[:1]) + [user_cache_dir()]:
        try:
            os.makedirs(d, exist_ok=True)
            svg = os.path.join(d, key + ".svg")
            with open(svg, "w", encoding="utf-8") as f:
                f.write(svg_text)
            with open(os.path.join(d, key + ".json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f)
            if paths is None:
                paths = svg
        except OSError:
            continue
    return paths


def inline_math_svg(tex, size_pt, sans=False, color="#000000", preamble=None, compile_missing=True):
    """The inline formula *tex* (without dollars) typeset by LaTeX at *size_pt*:
    (svg path, width, height, depth) in points — the SVG's baseline is y=0 —
    from the caches (CACHE_DIRS, then the user cache), compiled if missing
    (and *compile_missing*), or None if it cannot be had."""
    preamble = INLINE_PREAMBLE if preamble is None else preamble
    key = _key(tex, size_pt, sans, color, preamble)
    hit = _lookup(key)
    if hit is not None:
        USED.add(key)
        if CACHE_DIRS and not os.path.isfile(os.path.join(CACHE_DIRS[0], key + ".svg")):
            # found in the user cache: also keep it with the bound figure
            try:
                with open(hit[0], encoding="utf-8") as f:
                    _store(key, f.read(), {"width": hit[1], "height": hit[2], "depth": hit[3],
                                           "tex": tex})
            except OSError:
                pass
        return hit
    if not compile_missing:
        return None
    tex_exe = find_tex_tool("latex")
    dvisvgm = find_tex_tool("dvisvgm")
    if tex_exe is None or dvisvgm is None:
        return None
    rgb = color.lstrip("#").upper()
    if len(rgb) != 6 or re.search(r"[^0-9A-F]", rgb):
        rgb = "000000"
    lines = [r"\documentclass{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
             r"\usepackage{xcolor}", preamble]
    if sans:
        lines += [r"\usepackage{sansmath}", r"\renewcommand{\familydefault}{\sfdefault}",
                  # siunitx resets the maths font on purpose: keep the sans one
                  r"\makeatletter\@ifpackageloaded{siunitx}{\sisetup{mode=match,"
                  r"propagate-math-font=true,reset-math-version=false,reset-text-family=false,"
                  r"reset-text-series=false,text-family-to-math=true}}{}\makeatother"]
    lines += [r"\usepackage[active,tightpage,textmath]{preview}", r"\pagestyle{empty}",
              r"\begin{document}",
              f"\\fontsize{{{float(size_pt):g}}}{{{float(size_pt) * 1.2:g}}}\\selectfont"
              f"\\color[HTML]{{{rgb}}}" + (r"\sansmath" if sans else ""),
              f"${tex}$", r"\end{document}", ""]
    work = tempfile.mkdtemp(prefix="pyf_label_math_")
    try:
        with open(os.path.join(work, "m.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run([tex_exe, "-interaction=nonstopmode", "-halt-on-error", "m.tex"],
                           cwd=work, capture_output=True, text=True, timeout=300, creationflags=flags)
        if r.returncode != 0 or not os.path.isfile(os.path.join(work, "m.dvi")):
            return None
        r = subprocess.run([dvisvgm, "--no-fonts", "--bbox=preview", "m.dvi", "-o", "m.svg"],
                           cwd=work, capture_output=True, text=True, timeout=300, creationflags=flags)
        report = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"width=([\d.]+)pt,\s*height=([\d.]+)pt,\s*depth=([\d.]+)pt", report)
        svg_file = os.path.join(work, "m.svg")
        if not m or not os.path.isfile(svg_file):
            return None
        with open(svg_file, encoding="utf-8") as f:
            svg_text = f.read()
        metrics = {"width": float(m.group(1)), "height": float(m.group(2)),
                   "depth": float(m.group(3)), "tex": tex}
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    path = _store(key, svg_text, metrics)
    if path is None:
        return None
    USED.add(key)
    return path, metrics["width"], metrics["height"], metrics["depth"]
