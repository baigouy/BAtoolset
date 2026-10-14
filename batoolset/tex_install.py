"""Install LaTeX on demand: TinyTeX, a small TeX Live, in the user's folder.

PyFigures does not ship TeX inside its apps (size, licences, macOS signing);
instead this downloads TinyTeX (https://yihui.org/tinytex/, the TeX Live
binaries prebuilt for Windows, macOS and Linux, ~55-75 MB) into its standard
folder — %APPDATA%\\TinyTeX, ~/Library/TinyTeX or ~/.TinyTeX, no admin rights,
nothing added to the PATH — and installs the packages PyFigures uses with
TeX Live's own package manager (tlmgr). batoolset.latex_tools finds it there
(an existing TinyTeX, e.g. one installed for R Markdown, is used as is).

A document needing a package that is not installed can get it with
``install_missing(log)`` (the "File `x.sty' not found" errors of a LaTeX log).
"""
from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

DOWNLOAD_BASE = "https://github.com/rstudio/tinytex-releases/releases/download/daily/"
# what PyFigures uses: equation panels and label maths (latex/xelatex + dvisvgm),
# fonts in the figure's font (fontspec, unicode-math), .tex documents (pdflatex +
# preview), LaTeX data plots (Matplotlib usetex: dvipng, type1cm, cm-super for
# type1ec, underscore). TeX Live package names, not .sty names ("was" has upgreek)
PACKAGES = ["latex-bin", "xetex", "dvisvgm", "dvipng", "preview", "varwidth", "standalone",
            "fontspec", "unicode-math", "lm", "lm-math", "amsmath", "amsfonts", "tools",
            "siunitx", "sansmath", "was", "pgf", "pgfplots", "grfext", "xcolor", "graphics",
            "geometry", "booktabs", "type1cm", "cm-super", "underscore", "psnfss", "l3kernel",
            "l3packages", "etoolbox"]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# appended to "needs LaTeX" errors
INSTALL_HINT = ("PyFigures can install it for you (TinyTeX): Help → Install LaTeX for "
                "PyFigures…, the MCP tool install_latex, or "
                "python -m pyfigures.gui.headless_ops install-latex")


def _is_windows():
    return sys.platform.startswith("win")


def install_dir() -> str:
    """Where TinyTeX lives (its standard folder; PYFIGURES_TINYTEX_DIR overrides)."""
    if os.environ.get("PYFIGURES_TINYTEX_DIR"):
        return os.environ["PYFIGURES_TINYTEX_DIR"]
    if _is_windows():
        appdata = os.environ.get("APPDATA", "")
        # TeX Live wants a path without spaces or non-ASCII characters
        base = appdata if re.fullmatch(r"[!-~]+", appdata or " ") else os.environ.get(
            "ProgramData", r"C:\ProgramData")
        return os.path.join(base, "TinyTeX")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/TinyTeX")
    return os.path.expanduser("~/.TinyTeX")


def candidate_roots():
    """Folders where a TinyTeX may already be (ours first)."""
    roots = [install_dir()]
    if _is_windows():
        for base in (os.environ.get("APPDATA"), os.environ.get("ProgramData")):
            if base:
                roots.append(os.path.join(base, "TinyTeX"))
    else:
        roots += [os.path.expanduser("~/Library/TinyTeX"), os.path.expanduser("~/.TinyTeX")]
    out = []
    for r in roots:
        if r not in out:
            out.append(r)
    return out


def bin_dirs(root=None):
    """The bin/<platform> folders of a TinyTeX installation."""
    roots = [root] if root else candidate_roots()
    dirs = []
    for r in roots:
        dirs += sorted(d for d in glob.glob(os.path.join(r, "bin", "*")) if os.path.isdir(d))
    return dirs


def _tool(name, root=None):
    exts = (".exe", ".bat") if _is_windows() else ("",)
    for d in bin_dirs(root):
        for ext in exts:
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return p
    return None


def tlmgr(root=None):
    return _tool("tlmgr", root)


def is_installed(root=None) -> bool:
    """True when a TinyTeX with everything PyFigures needs is there."""
    return all(_tool(t, root) for t in ("latex", "pdflatex", "xelatex", "dvisvgm"))


def asset_name(system=None, machine=None) -> str:
    """The TinyTeX bundle for this computer (TinyTeX-1: ~90 common packages)."""
    system = system or sys.platform
    machine = (machine or platform.machine() or "").lower()
    if system.startswith("win"):
        return "TinyTeX-1-windows.exe"
    if system == "darwin":
        return "TinyTeX-1-darwin.tar.xz"
    if system.startswith("linux"):
        if machine in ("x86_64", "amd64"):
            return "TinyTeX-1-linux-x86_64.tar.xz"
        if machine in ("aarch64", "arm64"):
            return "TinyTeX-1-linux-arm64.tar.xz"
    raise ValueError(f"no prebuilt TinyTeX for {system} {machine}: install TeX Live, MiKTeX or "
                     "MacTeX instead")


def _report(progress, fraction, message):
    if progress is not None:
        progress(fraction, message)


def _ssl_context():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download(url, dest, progress=None, cancelled=None):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "PyFigures"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("cancelled")
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            frac = 0.6 * done / total if total else 0.0
            _report(progress, frac, f"Downloading TinyTeX… {done / 1e6:.0f}"
                                    + (f" / {total / 1e6:.0f} MB" if total else " MB"))


def _extract(archive, work):
    """Unpack the bundle in *work*; returns the TinyTeX folder it made."""
    if archive.endswith(".exe"):
        # a self-extracting 7-Zip archive (what TinyTeX's own installer runs)
        r = subprocess.run([archive, "-y"], cwd=work, capture_output=True, text=True,
                           timeout=1800, creationflags=_NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError("could not unpack TinyTeX: " + (r.stderr or r.stdout or "").strip()[-500:])
    else:
        with tarfile.open(archive, "r:*") as tf:
            try:
                tf.extractall(work, filter="data")
            except TypeError:                               # Python without extraction filters
                tf.extractall(work)
    for name in ("TinyTeX", ".TinyTeX"):
        if os.path.isdir(os.path.join(work, name)):
            return os.path.join(work, name)
    raise RuntimeError("the TinyTeX bundle has an unexpected layout")


def _run(cmd, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       creationflags=_NO_WINDOW)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def install_packages(packages, root=None, progress=None):
    """tlmgr install *packages* (already installed ones are skipped by tlmgr)."""
    tl = tlmgr(root)
    if tl is None:
        raise RuntimeError("tlmgr not found: no TinyTeX / TeX Live installation")
    _report(progress, 0.8, "Installing the LaTeX packages PyFigures uses…")
    code, out = _run([tl, "install"] + list(packages))
    if code != 0:
        raise RuntimeError("tlmgr could not install the packages:\n" + out.strip()[-1500:])
    # tlmgr only warns about names it does not know: check what is there
    code, listing = _run([tl, "info", "--list", "--only-installed", "--data", "name"], timeout=600)
    if code == 0:
        have = set(listing.split())
        missing = [p for p in packages if p not in have]
        if missing:
            raise RuntimeError("these LaTeX packages could not be installed: " + ", ".join(missing))
    return out


def install(progress=None, cancelled=None, packages=None) -> str:
    """Download and install TinyTeX plus PyFigures' packages (if not there yet).
    *progress(fraction, message)* is called along the way; *cancelled()* can
    stop the download. Returns the installation folder; raises on failure."""
    packages = PACKAGES if packages is None else packages
    root = install_dir()
    if not is_installed(root):
        name = asset_name()
        work = tempfile.mkdtemp(prefix="pyf_tinytex_")
        try:
            archive = os.path.join(work, name)
            _download(DOWNLOAD_BASE + name, archive, progress, cancelled)
            _report(progress, 0.65, "Unpacking TinyTeX…")
            unpacked = _extract(archive, work)
            if os.path.exists(root):
                # an incomplete earlier installation: kept aside, not deleted
                os.replace(root, root + time.strftime("-incomplete-%Y%m%d-%H%M%S"))
            os.makedirs(os.path.dirname(root), exist_ok=True)
            shutil.move(unpacked, root)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        tl = tlmgr(root)
        if tl is not None:
            _run([tl, "postaction", "install", "script", "xetex"], timeout=600)
    if packages:
        install_packages(packages, root, progress)
    _report(progress, 1.0, "LaTeX is installed.")
    return root


def missing_files(log: str):
    """Files a LaTeX run could not find ('x.sty', 'y.cls'), from its log."""
    names = re.findall(r"File [`']([^'`\s]+\.(?:sty|cls|def|tex|fd|cfg|bbx|cbx|lbx))' not found", log)
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def install_missing(log: str):
    """Install the TeX Live packages providing the files *log* says are
    missing (tlmgr search --file). Returns the packages installed (empty when
    nothing could be done, e.g. no tlmgr: MiKTeX installs on its own)."""
    from batoolset.latex_tools import find_tex_tool
    tl = find_tex_tool("tlmgr") or tlmgr()
    files = missing_files(log)
    if tl is None or not files:
        return []
    wanted = []
    for name in files:
        code, out = _run([tl, "search", "--global", "--file", "/" + name], timeout=600)
        if code != 0:
            continue
        for line in out.splitlines():
            m = re.match(r"^([A-Za-z0-9_.-]+):\s*$", line.strip())
            if m and m.group(1) not in wanted:
                wanted.append(m.group(1))
                break
    if not wanted:
        return []
    code, _ = _run([tl, "install"] + wanted)
    return wanted if code == 0 else []


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Install LaTeX (TinyTeX) for PyFigures")
    ap.parse_args(argv)

    def show(frac, msg):
        print(f"[{frac * 100:3.0f}%] {msg}", flush=True)
    try:
        root = install(progress=show)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 1
    print("TinyTeX:", root)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
