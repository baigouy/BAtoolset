from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
import matplotlib.pyplot as plt
from qtpy.QtCore import QRectF

def get_fig_rect(fig: plt.Figure) -> QRectF:
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