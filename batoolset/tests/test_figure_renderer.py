"""2026-11-20, PyFigures CI on macOS: a Matplotlib Figure() made without pyplot has the
base canvas, which has no get_renderer(). Measuring it failed: get_fig_rect fell back to
the whole figure size and plots were not fitted to their panel (3 % left empty)."""
import pytest

matplotlib = pytest.importorskip("matplotlib")


def _base_canvas_figure():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(4, 3))
    ax = fig.add_axes([0.3, 0.3, 0.4, 0.4])      # the content is far from the figure's edges
    ax.plot([0, 1], [0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def test_a_figure_without_pyplot_can_be_measured():
    from batoolset.matplolibs.tools import figure_renderer
    fig = _base_canvas_figure()
    renderer = figure_renderer(fig)
    box = fig.get_tightbbox(renderer)
    assert 0 < box.width < 4 and 0 < box.height < 3


def test_its_panel_rect_is_its_content_not_the_whole_figure():
    pytest.importorskip("qtpy")
    from batoolset.matplolibs.tools import get_fig_rect
    fig = _base_canvas_figure()
    rect = get_fig_rect(fig)
    dpi = fig.get_dpi()
    assert rect.width() < 0.6 * 4 * dpi and rect.height() < 0.6 * 3 * dpi


def test_a_listed_font_that_cannot_draw_is_skipped(monkeypatch):
    """macOS lists its system Helvetica (.ttc) but Agg fails on it with "failed to load
    glyph" (PyFigures CI 2026-11-20): the clone must be a font that draws."""
    import batoolset.matplolibs.tools as T
    monkeypatch.setattr(T, "_BEST_HELVETICA", [])
    monkeypatch.setattr(T, "HELVETICA_CANDIDATES", ["Broken Sans", "DejaVu Sans"])
    import matplotlib.font_manager as fm
    real = fm.fontManager.ttflist
    fake = type(real[0])(fname=real[0].fname, name="Broken Sans", style="normal", variant="normal",
                         weight=400, stretch="normal", size="scalable")
    monkeypatch.setattr(fm.fontManager, "ttflist", real + [fake])
    monkeypatch.setattr(T, "font_draws", lambda family: family != "Broken Sans")
    assert T.best_helvetica_clone() == "DejaVu Sans"


def test_dejavu_draws():
    from batoolset.matplolibs.tools import font_draws
    assert font_draws("DejaVu Sans") and not font_draws("No Such Font Anywhere 123")
