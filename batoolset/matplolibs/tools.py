import matplotlib.pyplot as plt

# Ordered preference list for Helvetica-like fonts (used by both Qt and matplotlib lookups).
HELVETICA_CANDIDATES = [
    'Helvetica',
    'Helvetica Neue',
    'Tex Gyre Heros',
    'Nimbus Sans',
    'Nimbus Sans L',
    'Liberation Sans',
    'Arial',
    'FreeSans',
    'DejaVu Sans',   # always bundled with matplotlib — final fallback
]


def best_helvetica_clone():
    """Return the best available Helvetica-like font from matplotlib's font list."""
    import matplotlib.font_manager as fm
    available = {f.name.lower() for f in fm.fontManager.ttflist}
    for candidate in HELVETICA_CANDIDATES:
        if candidate.lower() in available:
            return candidate
    return 'DejaVu Sans'

def figure_pad_inches(fig) -> float:
    """White margin kept around a figure's content when it is drawn as a panel
    (savefig pad_inches). A figure can opt out — image panels (channel merges,
    flipped images) set ``fig._pyfigures_pad_inches = 0`` so the pixels reach
    the panel edge instead of sitting in a white frame."""
    pad = getattr(fig, '_pyfigures_pad_inches', None)
    if isinstance(pad, (int, float)):
        return float(pad)
    import matplotlib as mpl
    pad = mpl.rcParams.get('savefig.pad_inches', 0.1)
    return float(pad) if isinstance(pad, (int, float)) else 0.1


def _fig_signature(fig):
    """Cheap fingerprint of what decides a figure's tight bounding box: size,
    dpi, padding, and per axes its position, limits, scales, visibility,
    labels and title, plus figure texts and legends. Measuring the figure
    itself (get_tightbbox) re-lays out every tick label — ~0.1 s per call — and
    PyFigures' layout asks for a panel's size thousands of times when a figure
    opens, so the size is cached under this fingerprint."""
    sup = getattr(fig, '_suptitle', None)
    sig = [tuple(fig.get_size_inches()), fig.get_dpi(), figure_pad_inches(fig),
           tuple(t.get_text() for t in fig.texts), len(fig.legends),
           sup.get_text() if sup is not None else None]
    for ax in fig.axes:
        sig.append((tuple(ax.get_position().bounds), ax.get_xlim(), ax.get_ylim(),
                    ax.get_xscale(), ax.get_yscale(), ax.axison, ax.get_visible(),
                    ax.get_xlabel(), ax.get_ylabel(), ax.get_title(),
                    ax.get_legend() is not None, len(ax.texts)))
    return tuple(sig)


def invalidate_fig_rect(fig):
    """Forget the cached panel size of *fig* — call after editing a figure in a
    way the fingerprint does not see (e.g. tick label font sizes)."""
    try:
        fig._pyfigures_rect_cache = None
    except Exception:
        pass


def get_fig_rect(fig: plt.Figure):
    """
    Get the size of the Matplotlib figure in pixels and return it as a QRectF object.

    Args:
        fig (plt.Figure): The Matplotlib figure.

    Returns:
        QRectF: A QRectF object representing the figure size in pixels.
    """
    from qtpy.QtCore import QRectF
    try:
        sig = _fig_signature(fig)
    except Exception:
        sig = None
    cached = getattr(fig, '_pyfigures_rect_cache', None)
    if sig is not None and cached is not None and cached[0] == sig:
        return QRectF(cached[1])            # a copy: callers may modify it
    rect = _compute_fig_rect(fig)
    if sig is not None:
        try:
            fig._pyfigures_rect_cache = (sig, QRectF(rect))
        except Exception:
            pass
    return rect


def _compute_fig_rect(fig):
    from qtpy.QtCore import QRectF
    fig_dpi = fig.get_dpi()

    # PyFigures draws the figure from savefig(..., bbox_inches='tight'), i.e.
    # cropped to its content + savefig.pad_inches. The panel rectangle must have
    # that same extent: using the full figure size (e.g. 6.4x4.8 in = 4:3) made
    # every panel whose content is not 4:3 get stretched — an equal-aspect image
    # in a default figure came out horizontally distorted.
    try:
        renderer = fig.canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer).padded(figure_pad_inches(fig))  # inches
        w, h = bbox.width * fig_dpi, bbox.height * fig_dpi
        if w > 0 and h > 0:
            return QRectF(0, 0, w, h)
    except Exception:
        pass

    # fallback: full figure size in pixels
    fig_size_pixels = fig.get_size_inches() * fig_dpi
    return QRectF(0, 0, fig_size_pixels[0], fig_size_pixels[1])


if __name__ == '__main__':
    import numpy as np

    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)', title=None)
    ax.grid()

    fig_rect = get_fig_rect(fig)
    print(fig_rect)