"""
Cell polarity (Polarity tab of pyTA).

The polarity nematic of a cell is measured by 'Measure cell properties' with 'Measure polarity' (TissueAnalyzer
polarity nematic: membrane intensity weighted by cos/sin(2 theta) around the cell centroid), one per channel, in the
cells_2D columns Q1_polarity_chN, Q2_polarity_chN and normalized_Q1/Q2_polarity_chN (divided by the membrane
intensity). This module draws and exports them:
    - draw_cell_polarity: one line per cell, oriented along 0.5 x atan2(Q2, Q1), of half length rescaling_factor x |Q|
      -> polarity_chN.tif (+ .svg) in the TA folder of every image (visible in the pyTA preview)
    - export_polarity: one row per cell and channel -> CSV (with the track ID when the cells are tracked)
"""
import math
import os
import numpy as np
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img
from batoolset.ta.database.sql import TAsql
from batoolset.ta.measurements.coarse_graining import draw_nematics, nematics_svg, _to_display_rgb
from batoolset.tools.early_stopper_class import early_stop
from batoolset.tools.logger import TA_logger

logger = TA_logger()


def polarity_channels(db_path):
    """Returns the channels (int) with a polarity nematic in the cells_2D table of a pyTA.db."""
    if not os.path.isfile(db_path):
        return []
    db = TAsql(filename_or_connection=db_path)
    try:
        if 'cells_2D' not in db.get_tables():
            return []
        columns = list(db.get_table_column_names_and_types('cells_2D').keys())
    finally:
        db.close()
    channels = []
    for col in columns:
        if col.startswith('Q1_polarity_ch'):
            channel = col[len('Q1_polarity_ch'):]
            if channel.isdigit() and 'Q2_polarity_ch' + channel in columns:
                channels.append(int(channel))
    return sorted(channels)


def polarity_columns(channel, normalized=True):
    prefix = 'normalized_' if normalized else ''
    return prefix + 'Q1_polarity_ch' + str(channel), prefix + 'Q2_polarity_ch' + str(channel)


def polarity_norm_and_angle(q1, q2):
    """Norm and angle (degrees, in ]-90, 90], image coordinates: y points down) of a polarity nematic."""
    return math.hypot(q1, q2), math.degrees(0.5 * math.atan2(q2, q1))


def _read_polarity(db_path, channel, normalized, exclude_border_cells):
    q1, q2 = polarity_columns(channel, normalized)
    db = TAsql(filename_or_connection=db_path)
    try:
        where = ' WHERE is_border = 0' if exclude_border_cells else ''
        rows = db.run_SQL_command_and_get_results(
            'SELECT centroid_x, centroid_y, "%s", "%s" FROM cells_2D%s' % (q1, q2, where), warn_on_error=False)
    finally:
        db.close()
    if rows is None:
        return None
    return [r for r in rows if None not in r]


def draw_cell_polarity(lst, channel=0, normalized=True, rescaling_factor=15., exclude_border_cells=False,
                       color=(255, 255, 0), stroke=1, save_svg=True, progress_callback=None):
    """
    Draws the polarity nematic of every cell over the original image, for every image of the list.

    Args:
        lst (list): the images (measured with 'Measure cell properties' + 'Measure polarity').
        channel (int): channel whose polarity is drawn.
        normalized (bool): draw the normalized nematic (norm between 0 and 1) instead of the raw one.
        rescaling_factor (float): half line length in pixels for a nematic of norm 1.
        exclude_border_cells (bool): do not draw the cells cut by the image border.
        color (tuple): RGB color of the lines.
        stroke (int): line width in pixels.
        save_svg (bool): also save polarity_chN.svg (vector lines over the embedded image).
        progress_callback: optional Qt signal receiving a percentage.

    Returns:
        list: the paths of the saved tif images.
    """
    name = 'polarity_ch' + str(channel)
    saved = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        if progress_callback is not None:
            progress_callback.emit(int(i / len(lst) * 100))
        TA_path = smart_name_parser(filename, ordered_output='TA')
        db_path = os.path.join(TA_path, 'pyTA.db')
        rows = _read_polarity(db_path, channel, normalized, exclude_border_cells) if os.path.isfile(db_path) else None
        if rows is None:
            logger.error('No polarity for channel ' + str(channel) + ' in ' + str(db_path) +
                         ', please run "Measure cell properties" with "Measure polarity" first')
            continue
        cells = [{'x': x, 'y': y, 'Q1': float(q1), 'Q2': float(q2)} for x, y, q1, q2 in rows]
        background = _to_display_rgb(Img(filename))
        output = os.path.join(TA_path, name + '.tif')
        Img(draw_nematics(background, cells, rescaling_factor, color, stroke), dimensions='hwc').save(output,
                                                                                                    mode='raw')
        saved.append(output)
        if save_svg:
            with open(os.path.join(TA_path, name + '.svg'), 'w', encoding='utf-8') as f:
                f.write(nematics_svg(background, cells, rescaling_factor, color, stroke))
    return saved


def export_polarity(lst, output_file, exclude_border_cells=False):
    """
    Saves the polarity of every cell (all channels) of every image of the list as a CSV file with the columns:
    frame_nb, filename, local_ID, track_id (empty if not tracked), centroid_x, centroid_y, is_border, channel,
    Q1, Q2, norm, angle_degrees, normalized_Q1, normalized_Q2, normalized_norm

    Returns:
        int: the number of rows written.
    """
    nb_rows = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('frame_nb,filename,local_ID,track_id,centroid_x,centroid_y,is_border,channel,Q1,Q2,norm,'
                'angle_degrees,normalized_Q1,normalized_Q2,normalized_norm\n')
        for frame_nb, filename in enumerate(lst):
            db_path = smart_name_parser(filename, ordered_output='pyTA.db')
            channels = polarity_channels(db_path)
            if not channels:
                logger.warning('No polarity in ' + str(db_path))
                continue
            db = TAsql(filename_or_connection=db_path)
            try:
                tracks = {}
                if 'cell_tracks' in db.get_tables():
                    for local_id, track_id in db.run_SQL_command_and_get_results(
                            'SELECT local_id, track_id FROM cell_tracks'):
                        tracks[local_id] = track_id
                cols = []
                for channel in channels:
                    cols.extend(polarity_columns(channel, False))
                    cols.extend(polarity_columns(channel, True))
                where = ' WHERE is_border = 0' if exclude_border_cells else ''
                rows = db.run_SQL_command_and_get_results(
                    'SELECT local_ID, centroid_x, centroid_y, is_border, ' + ', '.join('"%s"' % c for c in cols) +
                    ' FROM cells_2D' + where)
            finally:
                db.close()
            for row in rows:
                local_id, cx, cy, is_border = row[:4]
                track_id = tracks.get(local_id)
                for k, channel in enumerate(channels):
                    q1, q2, nq1, nq2 = row[4 + 4 * k:8 + 4 * k]
                    if q1 is None or q2 is None:
                        continue
                    norm, angle = polarity_norm_and_angle(float(q1), float(q2))
                    normalized_norm = math.hypot(float(nq1), float(nq2)) if nq1 is not None and nq2 is not None \
                        else ''
                    f.write('%d,%s,%s,%s,%.6g,%.6g,%s,%d,%.6g,%.6g,%.6g,%.3f,%s,%s,%s\n' % (
                        frame_nb, os.path.basename(filename), local_id, '' if track_id is None else track_id, cx, cy,
                        is_border, channel, float(q1), float(q2), norm, angle, _fmt(nq1), _fmt(nq2),
                        _fmt(normalized_norm)))
                    nb_rows += 1
    return nb_rows


def _fmt(value):
    if value is None or value == '':
        return ''
    return '%.6g' % float(value)
