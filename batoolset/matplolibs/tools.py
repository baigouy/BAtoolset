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

def get_fig_rect(fig: plt.Figure):
    """
    Get the size of the Matplotlib figure in pixels and return it as a QRectF object.

    Args:
        fig (plt.Figure): The Matplotlib figure.

    Returns:
        QRectF: A QRectF object representing the figure size in pixels.
    """
    # Get the figure size in inches
    fig_size_inches = fig.get_size_inches()

    # Get the figure DPI
    fig_dpi = fig.get_dpi()

    # Convert the figure size to pixels
    fig_size_pixels = fig_size_inches * fig_dpi

    # Create a QRectF object using the figure size in pixels
    from qtpy.QtCore import QRectF
    fig_rect = QRectF(0, 0, fig_size_pixels[0], fig_size_pixels[1])

    return fig_rect


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