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
    # a hair, not matplotlib's 0.1 inch: that white frame made a plot sit further from
    # its neighbours than the panel spacing (user 2026-11-19: "plots leave space on
    # left and right, not respecting the spacing between panels"); the hair keeps the
    # antialiased edge of the outermost text from being cut
    return PANEL_PAD_INCHES


PANEL_PAD_INCHES = 0.01


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
    try:
        # the key AFTER measuring: measuring draws the figure, and draw-time layouts
        # (brackets) change its limits; keyed before, the next call missed the cache and
        # measured another shape, and a plot was letterboxed in its panel (2026-11-19)
        sig = _fig_signature(fig)
    except Exception:
        sig = None
    if sig is not None:
        try:
            fig._pyfigures_rect_cache = (sig, QRectF(rect))
        except Exception:
            pass
    return rect


def settle_figure(fig, rounds=3):
    """Draw *fig* without rendering until its axes limits stop changing (draw-time
    layouts, e.g. significance brackets sized from their labels, adjust them)."""
    for _ in range(rounds):
        before = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]
        try:
            fig.draw_without_rendering()
        except Exception:
            return
        if [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes] == before:
            return


def _compute_fig_rect(fig):
    from qtpy.QtCore import QRectF
    fig_dpi = fig.get_dpi()

    # PyFigures draws the figure from savefig(..., bbox_inches='tight'), i.e.
    # cropped to its content + savefig.pad_inches. The panel rectangle must have
    # that same extent: using the full figure size (e.g. 6.4x4.8 in = 4:3) made
    # every panel whose content is not 4:3 get stretched — an equal-aspect image
    # in a default figure came out horizontally distorted.
    try:
        # drawn first: artists that settle at draw time (staplot's bracket layout sets
        # the y range from the labels' real size) change the outline; measured without
        # it, the panel was planned narrower-taller than the plot drawn into it, which
        # was then letterboxed: white strips left and right (user, 2026-11-19, twice)
        settle_figure(fig)
        renderer = fig.canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer).padded(figure_pad_inches(fig))  # inches
        # rounded: a 58 px figure (0.58 in at 100 dpi) came out 57.9999… and
        # was truncated to 57 px further on
        w, h = round(bbox.width * fig_dpi, 6), round(bbox.height * fig_dpi, 6)
        if w > 0 and h > 0:
            return QRectF(0, 0, w, h)
    except Exception:
        pass

    # fallback: full figure size in pixels
    fig_size_pixels = fig.get_size_inches() * fig_dpi
    return QRectF(0, 0, round(float(fig_size_pixels[0]), 6), round(float(fig_size_pixels[1]), 6))


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