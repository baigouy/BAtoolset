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
    # TinyTeX (installed by PyFigures on demand, batoolset.tex_install, or for R)
    from batoolset.tex_install import bin_dirs as _tinytex_bins
    dirs += _tinytex_bins()
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


def ensure_on_path():
    """Put the LaTeX installation's folder on PATH (Matplotlib's usetex calls
    latex / dvipng by name). Returns True if LaTeX was found."""
    tex = find_tex_tool("latex")
    if tex is None:
        return False
    folder = os.path.dirname(tex)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if folder not in parts:
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    return True


def matplotlib_rc(sans=True):
    """Matplotlib rcParams typesetting every text of a plot with LaTeX
    (text.usetex), sans-serif text and maths by default (to match figure
    labels), with amsmath and siunitx. Empty without a LaTeX installation."""
    if not ensure_on_path():
        return {}
    # µ, ± and ° from maths fonts: Matplotlib's SVG output draws the textcomp
    # (TS1) glyphs LaTeX would use for them far too small
    preamble = (r"\usepackage{amsmath}\usepackage{upgreek}\usepackage{siunitx}"
                r"\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}"
                r"\DeclareUnicodeCharacter{00B5}{\ensuremath{\upmu}}"
                r"\DeclareUnicodeCharacter{00B0}{\ensuremath{^\circ}}"
                r"\DeclareSIPrefix\micro{\ensuremath{\upmu}}{-6}")
    if sans:
        preamble += (r"\usepackage{sansmath}\sansmath\renewcommand{\familydefault}{\sfdefault}"
                     r"\sisetup{mode=match,propagate-math-font=true,reset-math-version=false,"
                     r"reset-text-family=false,reset-text-series=false,text-family-to-math=true}")
    return {"text.usetex": True, "text.latex.preamble": preamble,
            "font.family": "sans-serif" if sans else "serif"}


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
TYPESET_VERSION = 3


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


def font_files_for(family):
    """(regular, bold, italic, bold italic) font files of the installed font
    *family* (Matplotlib's font finder), or None if it is not found."""
    if not family:
        return None
    try:
        from matplotlib import font_manager as fm
        files = []
        for weight, style in (("normal", "normal"), ("bold", "normal"), ("normal", "italic"),
                              ("bold", "italic")):
            files.append(fm.findfont(fm.FontProperties(family=family, weight=weight, style=style),
                                     fallback_to_default=False))
    except Exception:
        return None
    if not files[0] or not os.path.isfile(files[0]):
        return None
    return tuple(f if f and os.path.isfile(f) else files[0] for f in files)


def xelatex_available() -> bool:
    return find_tex_tool("xelatex") is not None and find_tex_tool("dvisvgm") is not None


def font_preamble(font_files):
    """XeLaTeX preamble lines setting the text font and taking the maths
    letters, digits and Greek from it (unicode-math; symbols, fraction bars,
    operators from Latin Modern Math)."""
    def spec(path):
        d, name = os.path.split(os.path.abspath(path))
        return name, d.replace("\\", "/") + "/"
    (reg, rdir), (bold, bdir), (ital, idir), (bital, bidir) = (spec(p) for p in font_files)
    return [
        r"\usepackage{fontspec}",
        f"\\setmainfont{{{reg}}}[Path={rdir}, BoldFont={bold}, ItalicFont={ital}, "
        f"BoldItalicFont={bital}]",
        r"\usepackage{unicode-math}",
        r"\setmathfont{latinmodern-math.otf}",
        f"\\setmathfont{{{reg}}}[Path={rdir}, range={{up/{{Latin,latin,num,Greek,greek}}}}]",
        f"\\setmathfont{{{bold}}}[Path={bdir}, range={{bfup/{{Latin,latin,num,Greek,greek}}}}]",
        f"\\setmathfont{{{ital}}}[Path={idir}, range={{it/{{Latin,latin,Greek,greek}}}}]",
        f"\\setmathfont{{{bital}}}[Path={bidir}, range={{bfit/{{Latin,latin,Greek,greek}}}}]",
        r"\AtBeginDocument{\renewcommand{\bm}[1]{\symbf{#1}}}",
        r"\makeatletter\@ifpackageloaded{siunitx}{\sisetup{mode=match,propagate-math-font=true,"
        r"reset-math-version=false,reset-text-family=false,reset-text-series=false,"
        r"text-family-to-math=true}}{}\makeatother",
    ]


_BP_TO_PT = 72.27 / 72.0


def _typeset(lines, engine):
    """Compile *lines* (a document whose body marks its baseline origin, see
    inline_math_svg) and return (svg text, width, height, depth in pt), or None."""
    tex_exe = find_tex_tool(engine)
    dvisvgm = find_tex_tool("dvisvgm")
    if tex_exe is None or dvisvgm is None:
        return None
    work = tempfile.mkdtemp(prefix="pyf_label_math_")
    try:
        with open(os.path.join(work, "m.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cmd = [tex_exe, "-interaction=nonstopmode", "-halt-on-error"]
        if engine == "xelatex":
            cmd.append("-no-pdf")
        r = subprocess.run(cmd + ["m.tex"], cwd=work, capture_output=True, text=True,
                           timeout=600, creationflags=flags)
        dvi = os.path.join(work, "m.xdv" if engine == "xelatex" else "m.dvi")
        if r.returncode != 0 or not os.path.isfile(dvi):
            return None
        subprocess.run([dvisvgm, "--no-fonts", "--bbox=min", os.path.basename(dvi), "-o", "m.svg"],
                       cwd=work, capture_output=True, text=True, timeout=300, creationflags=flags)
        svg_file = os.path.join(work, "m.svg")
        if not os.path.isfile(svg_file):
            return None
        with open(svg_file, encoding="utf-8") as f:
            svg_text = f.read()
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    # the box (dvisvgm bbox special) is the viewBox; the marker is its baseline origin
    vb = re.search(r"viewBox=['\"]([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)['\"]", svg_text)
    base = re.search(r"pyfbase ([-\d.]+) ([-\d.]+)", svg_text)
    if not vb or not base:
        return None
    vy, vw, vh = float(vb.group(2)), float(vb.group(3)), float(vb.group(4))
    by = float(base.group(2))
    return (svg_text, vw * _BP_TO_PT, (by - vy) * _BP_TO_PT, (vy + vh - by) * _BP_TO_PT)


def inline_math_svg(tex, size_pt, sans=False, color="#000000", preamble=None, compile_missing=True,
                    font_files=None):
    """The inline formula *tex* (without dollars) typeset by LaTeX at *size_pt*:
    (svg path, width, height, depth) in points — the SVG's viewBox spans
    [-height, +depth] around the baseline — from the caches (CACHE_DIRS, then
    the user cache), compiled if missing (and *compile_missing*), or None if it
    cannot be had.

    *font_files* (font_files_for): with XeLaTeX, the maths letters, digits and
    Greek are set in that font (the label's own); otherwise Latin Modern (sans
    when *sans*)."""
    preamble = INLINE_PREAMBLE if preamble is None else preamble
    plain_key = _key(tex, size_pt, sans, color, preamble)
    font_key = (_key(tex, size_pt, sans, color,
                     preamble + "|font:" + "|".join(os.path.basename(f) for f in font_files))
                if font_files else None)
    use_font = bool(font_files) and xelatex_available()
    key = font_key if use_font else plain_key
    # the stored formula is used whatever is installed now: in the font when it
    # was typeset so, else (no XeLaTeX here to make that) in Latin Modern
    hit = None
    for candidate in ([font_key] if font_key else []) + ([] if use_font else [plain_key]):
        hit = _lookup(candidate)
        if hit is not None:
            key = candidate
            break
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
    rgb = color.lstrip("#").upper()
    if len(rgb) != 6 or re.search(r"[^0-9A-F]", rgb):
        rgb = "000000"
    lines = [r"\documentclass{article}"]
    if use_font:
        lines += [r"\usepackage{xcolor}", preamble] + font_preamble(font_files)
        sans_switch = ""
    else:
        lines += [r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}", r"\usepackage{xcolor}",
                  preamble]
        sans_switch = r"\sansmath" if sans else ""
        if sans:
            lines += [r"\usepackage{sansmath}", r"\renewcommand{\familydefault}{\sfdefault}",
                      # siunitx resets the maths font on purpose: keep the sans one
                      r"\makeatletter\@ifpackageloaded{siunitx}{\sisetup{mode=match,"
                      r"propagate-math-font=true,reset-math-version=false,reset-text-family=false,"
                      r"reset-text-series=false,text-family-to-math=true}}{}\makeatother"]
    lines += [r"\pagestyle{empty}", r"\begin{document}",
              f"\\fontsize{{{float(size_pt):g}}}{{{float(size_pt) * 1.2:g}}}\\selectfont"
              f"\\color[HTML]{{{rgb}}}{sans_switch}",
              # the box, then dvisvgm told its size (bbox) and its baseline origin (marker)
              f"\\setbox0=\\hbox{{${tex}$}}%",
              r"\noindent\special{dvisvgm:raw <!-- pyfbase {?x} {?y} -->}"
              r"\special{dvisvgm:bbox \the\wd0\space\the\ht0\space\the\dp0}\box0",
              r"\end{document}", ""]
    done = _typeset(lines, "xelatex" if use_font else "latex")
    if done is None:
        return None
    svg_text, width, height, depth = done
    metrics = {"width": width, "height": height, "depth": depth, "tex": tex}
    path = _store(key, svg_text, metrics)
    if path is None:
        return None
    USED.add(key)
    return path, metrics["width"], metrics["height"], metrics["depth"]
