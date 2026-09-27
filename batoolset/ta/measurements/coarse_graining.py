"""
Coarse graining of nematics (polarity, cell stretch/elongation) on a square grid.

Port of TissueAnalyzer (Java) 'Coarse grain polarity nematic' (CoarseGrainer). The image is divided into square bins
of bin_size pixels; the nematic (Q1, Q2) of the non border cells whose centroid lies in a bin are averaged and drawn as
a line of length 2 x rescaling_factor x |Q| centered on the bin, oriented along 0.5 x atan2(Q2, Q1).

Nematics available from 'Measure cell properties' (cells_2D):
    - S1/S2: cell stretch (elongation) nematic
    - Q1_polarity_chN/Q2_polarity_chN (and normalized_Q1/Q2_...): polarity nematic of channel N ('Measure polarity')

Outputs, per frame (in the TA folder, visible in the pyTA preview):
    - coarse_grained_<nematic>.tif: the lines drawn over the original image (RGB)
    - coarse_grained_<nematic>.svg: the same as vector graphics (original image embedded), for figures
And for the whole movie, next to the images: coarse_grained_<nematic>.csv (frame_nb, bin center, Q1, Q2, nb of cells,
norm, angle in degrees)
"""
import base64
import io
import math
import os
import numpy as np
from skimage.draw import line_aa
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img
from batoolset.tools.early_stopper_class import early_stop
from batoolset.ta.database.sql import TAsql
from batoolset.tools.logger import TA_logger

logger = TA_logger()

STRETCH = ('S1', 'S2')


def available_nematics(db_path):
    """
    Returns the list of (Q1 column, Q2 column) nematics found in the cells_2D table of a pyTA.db.
    """
    if not os.path.isfile(db_path):
        return []
    db = TAsql(filename_or_connection=db_path)
    try:
        if 'cells_2D' not in db.get_tables():
            return []
        columns = list(db.get_table_column_names_and_types('cells_2D').keys())
    finally:
        db.close()
    out = []
    if 'S1' in columns and 'S2' in columns:
        out.append(STRETCH)
    for col in columns:
        if 'Q1_polarity_ch' in col:
            other = col.replace('Q1_polarity_ch', 'Q2_polarity_ch')
            if other in columns:
                out.append((col, other))
    return out


def nematic_name(nematic):
    if tuple(nematic) == STRETCH:
        return 'stretch'
    return nematic[0].replace('Q1_', '')


def coarse_grain(xs, ys, q1, q2, width, height, bin_size):
    """
    Averages nematics in square bins (bins entirely inside the image, as in TissueAnalyzer; points lying exactly on a
    bin edge are ignored).

    Returns:
        list: dicts with x, y (bin center), Q1, Q2, nb_of_cells for the non empty bins.
    """
    out = []
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(q1) & np.isfinite(q2)
    xs, ys, q1, q2 = xs[ok], ys[ok], q1[ok], q2[ok]
    half = bin_size / 2.
    for i in range(0, width, bin_size):
        cx = i + bin_size // 2
        if cx >= width:
            continue
        in_x = (xs > cx - half) & (xs < cx + half)
        if not in_x.any():
            continue
        for j in range(0, height, bin_size):
            cy = j + bin_size // 2
            if cy >= height:
                continue
            sel = in_x & (ys > cy - half) & (ys < cy + half)
            n = int(sel.sum())
            if n == 0:
                continue
            out.append({'x': cx, 'y': cy, 'Q1': float(q1[sel].mean()), 'Q2': float(q2[sel].mean()), 'nb_of_cells': n})
    return out


def nematic_line(b, rescaling_factor):
    """Returns the two ends (x0, y0, x1, y1) of the line representing a bin nematic, or None if it is null."""
    norm = math.hypot(b['Q1'], b['Q2'])
    if norm == 0 or not math.isfinite(norm):
        return None
    angle = 0.5 * math.atan2(b['Q2'], b['Q1'])
    scale = rescaling_factor * norm
    dx = scale * math.cos(angle)
    dy = scale * math.sin(angle)
    return b['x'] - dx, b['y'] - dy, b['x'] + dx, b['y'] + dy


def _to_display_rgb(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    img = img.astype(float)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 2:
        img = np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]
    lo, hi = np.percentile(img, (0.5, 99.5))
    return (np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def draw_nematics(background_rgb, bins, rescaling_factor, color=(255, 255, 0), stroke=2):
    """Draws the bin nematics (anti-aliased lines) over an RGB uint8 image."""
    out = background_rgb.astype(float)
    h, w = out.shape[:2]
    color = np.asarray(color, float)
    r = max(0, int(round((stroke - 1) / 2.)))
    for b in bins:
        line = nematic_line(b, rescaling_factor)
        if line is None:
            continue
        x0, y0, x1, y1 = line
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                rr, cc, val = line_aa(int(round(y0)) + oy, int(round(x0)) + ox, int(round(y1)) + oy,
                                      int(round(x1)) + ox)
                ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc, val = rr[ok], cc[ok], val[ok][:, None]
                out[rr, cc] = out[rr, cc] * (1 - val) + color * val
    return out.astype(np.uint8)


def nematics_svg(background_rgb, bins, rescaling_factor, color=(255, 255, 0), stroke=2):
    """Returns an SVG string: the background image (embedded PNG) and one line per bin."""
    from PIL import Image
    h, w = background_rgb.shape[:2]
    buffer = io.BytesIO()
    Image.fromarray(background_rgb).save(buffer, format='PNG')
    png = base64.b64encode(buffer.getvalue()).decode('ascii')
    hex_color = '#%02x%02x%02x' % tuple(int(c) for c in color)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             'width="%d" height="%d" viewBox="0 0 %d %d">' % (w, h, w, h),
             '<image width="%d" height="%d" xlink:href="data:image/png;base64,%s"/>' % (w, h, png),
             '<g stroke="%s" stroke-width="%g" stroke-opacity="0.85" stroke-linecap="round">' % (hex_color, stroke)]
    for b in bins:
        line = nematic_line(b, rescaling_factor)
        if line is not None:
            parts.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>' % line)
    parts.append('</g></svg>')
    return '\n'.join(parts)


def _read_cells(db_path, nematic, exclude_border_cells):
    db = TAsql(filename_or_connection=db_path)
    try:
        if 'cells_2D' not in db.get_tables():
            return None
        where = ' WHERE is_border = 0' if exclude_border_cells else ''
        return db.run_SQL_command_and_get_results(
            'SELECT centroid_x, centroid_y, "%s", "%s" FROM cells_2D%s' % (nematic[0], nematic[1], where))
    except Exception:
        return None
    finally:
        db.close()


def coarse_grain_nematics(lst, nematic=STRETCH, bin_size=64, rescaling_factor=100., exclude_border_cells=True,
                          color=(255, 255, 0), stroke=2, save_svg=True, progress_callback=None):
    """
    Coarse grains a nematic for every image of the list and saves the images/svg/csv files (see module doc).

    Args:
        lst (list): the images (measured with 'Measure cell properties').
        nematic (tuple): the (Q1, Q2) columns of cells_2D (see available_nematics).
        bin_size (int): size of the square bins in pixels.
        rescaling_factor (float): half line length in pixels for a nematic of norm 1.
        exclude_border_cells (bool): ignore the cells cut by the image border (as in TissueAnalyzer).
        color (tuple): RGB color of the lines.
        stroke (float): line width in pixels.
        save_svg (bool): also save the svg version.
        progress_callback: optional Qt signal receiving a percentage.

    Returns:
        list: (frame_nb, bins) for every processed frame
    """
    if not lst:
        return []
    name = nematic_name(nematic)
    results = []
    rows = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        if progress_callback is not None:
            progress_callback.emit(int(i / len(lst) * 100))
        TA_path = smart_name_parser(filename, ordered_output='TA')
        db_path = os.path.join(TA_path, 'pyTA.db')
        data = _read_cells(db_path, nematic, exclude_border_cells) if os.path.isfile(db_path) else None
        if data is None:
            logger.error('Missing cells_2D or ' + str(nematic) + ' in ' + str(db_path) +
                         ' please run "Measure cell properties" first (with "Measure polarity" for polarity)')
            continue
        data = [r for r in data if None not in r]
        original = Img(filename)
        height, width = original.shape[:2] if original.ndim >= 2 else (0, 0)
        if data:
            xs, ys, q1, q2 = (np.asarray(c, float) for c in zip(*data))
        else:
            xs = ys = q1 = q2 = np.zeros(0)
        bins = coarse_grain(xs, ys, q1, q2, width, height, bin_size)
        results.append((i, bins))
        background = _to_display_rgb(original)
        Img(draw_nematics(background, bins, rescaling_factor, color, stroke), dimensions='hwc').save(
            os.path.join(TA_path, 'coarse_grained_' + name + '.tif'), mode='raw')
        if save_svg:
            with open(os.path.join(TA_path, 'coarse_grained_' + name + '.svg'), 'w', encoding='utf-8') as f:
                f.write(nematics_svg(background, bins, rescaling_factor, color, stroke))
        for b in bins:
            rows.append((i, os.path.basename(filename), b['x'], b['y'], b['Q1'], b['Q2'], b['nb_of_cells'],
                         math.hypot(b['Q1'], b['Q2']), math.degrees(0.5 * math.atan2(b['Q2'], b['Q1']))))
    parent = smart_name_parser(lst[0], ordered_output='parent')
    with open(os.path.join(parent, 'coarse_grained_' + name + '.csv'), 'w', encoding='utf-8') as f:
        f.write('frame_nb,filename,x,y,Q1,Q2,nb_of_cells,norm,angle_degrees\n')
        for row in rows:
            f.write('%d,%s,%d,%d,%.6g,%.6g,%d,%.6g,%.2f\n' % row)
    return results
