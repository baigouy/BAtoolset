"""
Movie calibration (pixel size and frame interval) and calibrated columns for the exports.

The calibration is stored in the 'properties' table of every image database (one row per image):
    voxel_size_x, voxel_size_y: pixel size (same unit as pixel_unit)
    pixel_unit: e.g. 'µm'
    frame_interval, time_unit: time between two consecutive images of the list
    time: frame number * frame_interval
Exports then get, next to the pixel columns, the same columns in physical units (suffix '_calibrated').
"""
from batoolset.ta.database.sql import set_property, get_property
from batoolset.utils.loadlist import smart_TA_list

# column -> power of the pixel size (1 = length, 2 = area) and axis ('x', 'y' or None for isotropic)
CALIBRATED_COLUMNS = {
    'area': (2, None),
    'cytoplasmic_area': (2, None),
    'area3D': (2, None),
    'area_flat_height': (2, None),
    'perimeter': (1, None),
    'perimeter3D': (1, None),
    'major_axis_length': (1, None),
    'minor_axis_length': (1, None),
    'length': (1, None),
    'length3D': (1, None),
    'centroid_x': (1, 'x'),
    'centroid_y': (1, 'y'),
    'x': (1, 'x'),
    'y': (1, 'y'),
}
CALIBRATED_SUFFIX = '_calibrated'


def set_movie_calibration(lst, pixel_size=None, pixel_unit='µm', frame_interval=None, time_unit='min'):
    """
    Stores the pixel size and/or the frame interval in the 'properties' table of every image of the list.

    Args:
        lst (list): the images of the movie, in time order.
        pixel_size (float or None): size of a pixel in pixel_unit (None = leave the pixel size unchanged).
        pixel_unit (str): unit of the pixel size.
        frame_interval (float or None): time between two consecutive images in time_unit (None = leave the time
            unchanged).
        time_unit (str): unit of the frame interval.
    """
    for frame_nb, db_file in enumerate(smart_TA_list(lst, 'pyTA.db')):
        data = {}
        if pixel_size is not None:
            data['voxel_size_x'] = float(pixel_size)
            data['voxel_size_y'] = float(pixel_size)
            data['pixel_unit'] = str(pixel_unit)
        if frame_interval is not None:
            data['frame_interval'] = float(frame_interval)
            data['time_unit'] = str(time_unit)
            data['time'] = frame_nb * float(frame_interval)
        set_property(db_file, data)


def get_movie_calibration(lst):
    """
    Returns the calibration of the first image of the list.

    Returns:
        dict: pixel_size, pixel_unit, frame_interval, time_unit (None when not set).
    """
    out = {'pixel_size': None, 'pixel_unit': None, 'frame_interval': None, 'time_unit': None}
    if not lst:
        return out
    db_file = smart_TA_list(lst[:1], 'pyTA.db')[0]
    pixel_size = get_property(db_file, 'voxel_size_x')
    frame_interval = get_property(db_file, 'frame_interval')
    out['pixel_size'] = _to_float(pixel_size)
    out['frame_interval'] = _to_float(frame_interval)
    out['pixel_unit'] = get_property(db_file, 'pixel_unit')
    out['time_unit'] = get_property(db_file, 'time_unit')
    return out


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calibrated_query(db, sql_command):
    """
    add_calibrated_columns for a query run on db (a TAsql, e.g. the master database of the exports).
    """
    header, _ = db.run_SQL_command_and_get_results('SELECT * FROM (' + sql_command + ') LIMIT 0', return_header=True,
                                                   warn_on_error=False)
    if not header:
        return sql_command
    return add_calibrated_columns(sql_command, list(header))


def add_calibrated_columns(sql_command, columns):
    """
    Wraps an export query so that it also returns the length/area columns in physical units. Needs the 'properties'
    table in the query (voxel_size_x, voxel_size_y columns); rows without a pixel size get NULL.

    Args:
        sql_command (str): the export query (e.g. 'SELECT * FROM cells_2D NATURAL JOIN properties').
        columns (list): the column names returned by sql_command.

    Returns:
        str: the new query, or sql_command unchanged when there is nothing to calibrate.
    """
    if 'voxel_size_x' not in columns:
        return sql_command
    size_y = 'COALESCE(voxel_size_y, voxel_size_x)'
    expressions = []
    for col in columns:
        if col not in CALIBRATED_COLUMNS or col + CALIBRATED_SUFFIX in columns:
            continue
        power, axis = CALIBRATED_COLUMNS[col]
        if axis == 'y':
            factor = size_y
        elif power == 2:
            factor = 'voxel_size_x * ' + size_y
        else:
            factor = 'voxel_size_x'
        expressions.append('"%s" * %s AS "%s"' % (col, factor, col + CALIBRATED_SUFFIX))
    if not expressions:
        return sql_command
    return 'SELECT *, ' + ', '.join(expressions) + ' FROM (' + sql_command + ')'
