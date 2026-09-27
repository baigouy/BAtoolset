"""
Detection of T1 transitions (neighbour exchanges) from tracked cells.

Port of the TissueAnalyzer (Java) QuickT1Detector ('Detect T1s' + 'Removes all oscillating T1s').

In every frame, two cells a and b in contact that have exactly two common neighbours c and d form a quadruplet
(a, b) -> (c, d). A T1 is detected at frame j when a quadruplet (a, b) -> (c, d) seen at frame i < j is inverted: at j,
c and d are in contact with exactly a and b as common neighbours and a and b are no longer a quadruplet. For each
frame j and losing pair (a, b), the last frame i where (a, b) -> (c, d) was seen before the exchange is kept.

Oscillating T1s (the same four cells exchanging neighbours back and forth, a common artefact of short bonds) are
filtered as in TissueAnalyzer: if the four cells exchange neighbours an odd number of times, only the last exchange is
kept (net T1), if they do it an even number of times, none is kept (no net neighbour change).

The Java version compared every frame to every later frame (frames^2 x pairs, with pixel by pixel flood fills), this
version indexes the inverted quadruplets once (frames x pairs).

Outputs, per frame (in the TA folder, visible in the pyTA preview):
    - T1s.tif / stable_T1s.tif: cells losing contact in red, cells gaining contact in green, both in yellow
    - 'T1s' and 'stable_T1s' tables in pyTA.db (same columns as in TissueAnalyzer TA.db)
And for the whole movie, next to the images: T1s.csv and stable_T1s.csv
"""
import os
from bisect import bisect_right
import numpy as np
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img, int24_to_RGB
from batoolset.tools.early_stopper_class import early_stop
from batoolset.ta.database.sql import TAsql
from batoolset.ta.tracking.divisions_and_deaths import load_tracks
from batoolset.ta.tracking.tracking_error_detector_and_fixer import get_cells_and_their_neighbors_dict_fast
from batoolset.tools.logger import TA_logger

logger = TA_logger()

MEMBRANE = 0xFFFFFF
BACKGROUND = 0
LOSING_COLOR = 0xFF0000
GAINING_COLOR = 0x00FF00
BOTH_COLOR = 0xFFFF00

# same columns as TissueAnalyzer
T1S_COLUMNS = ['losing_pair_cell1_track_pair_still_in_contact', 'losing_pair_cell1_track_id',
               'losing_pair_cell2_track_id', 'gaining_pair_cell1_track_id_in_contact', 'gaining_pair_cell1_track_id',
               'gaining_pair_cell2_track_id']
STABLE_T1S_COLUMNS = ['losing_pair_cell1_track_id', 'losing_pair_cell2_track_id', 'gaining_pair_cell1_track_id',
                      'gaining_pair_cell2_track_id']


def get_quadruplets(tracks):
    """
    Returns a dict (a, b) -> (c, d) for every pair of cells in contact (a < b) that have exactly two common neighbours
    c < d (background and membrane are not cells).
    """
    neighbors = {}
    for cell, neighbs in get_cells_and_their_neighbors_dict_fast(np.asarray(tracks)).items():
        cell = int(cell)
        if cell == BACKGROUND or cell == MEMBRANE:
            continue
        neighbors[cell] = {int(n) for n in neighbs if n != BACKGROUND and n != MEMBRANE and n != cell}
    quadruplets = {}
    for a, neighbs_a in neighbors.items():
        for b in neighbs_a:
            if b <= a or b not in neighbors:
                continue
            common = neighbs_a & neighbors[b]
            if len(common) == 2:
                c, d = sorted(common)
                quadruplets[(a, b)] = (c, d)
    return quadruplets


def find_T1s(quadruplets_per_frame):
    """
    Finds the neighbour exchanges (see module doc).

    Args:
        quadruplets_per_frame (list): for every frame, the dict returned by get_quadruplets (None for a missing frame).

    Returns:
        list: T1s sorted by frame, dicts with frame_nb (j), last_frame_in_contact (i), losing_pair, gaining_pair.
    """
    # frames where each pair (c, d) is in contact with exactly (a, b) as common neighbours while (a, b) is not a
    # quadruplet anymore
    inverted = {}
    for j, quadruplets in enumerate(quadruplets_per_frame):
        if not quadruplets:
            continue
        for gaining, losing in quadruplets.items():
            if losing not in quadruplets:
                inverted.setdefault((gaining, losing), []).append(j)

    # for each quadruplet seen at i, the first exchange after i; per (j, losing pair) the last i wins
    t1s = {}
    for i, quadruplets in enumerate(quadruplets_per_frame):
        if not quadruplets:
            continue
        for losing, gaining in quadruplets.items():
            frames = inverted.get((gaining, losing))
            if frames is None:
                continue
            k = bisect_right(frames, i)
            if k < len(frames):
                t1s[(frames[k], losing)] = (i, gaining)
    out = [{'frame_nb': j, 'last_frame_in_contact': i, 'losing_pair': losing, 'gaining_pair': gaining}
           for (j, losing), (i, gaining) in t1s.items()]
    out.sort(key=lambda t: (t['frame_nb'], t['losing_pair'], t['gaining_pair']))
    return out


def filter_oscillating_T1s(t1s):
    """
    Removes oscillating T1s: for the same four cells, only the last exchange is kept if they exchange neighbours an
    odd number of times, none if it is an even number of times (same result as TissueAnalyzer filterOscillatingT1s).
    """
    by_cells = {}
    for t1 in t1s:
        key = frozenset([t1['losing_pair'], t1['gaining_pair']])
        by_cells.setdefault(key, []).append(t1)
    stable = []
    for events in by_cells.values():
        if len(events) % 2 == 1:
            stable.append(max(events, key=lambda t: t['frame_nb']))
    stable.sort(key=lambda t: (t['frame_nb'], t['losing_pair'], t['gaining_pair']))
    return stable


def _paint_T1s(tracks, t1s_now):
    losing = set()
    gaining = set()
    for t1 in t1s_now:
        losing.update(t1['losing_pair'])
        gaining.update(t1['gaining_pair'])
    out = np.zeros(tracks.shape, np.uint32)
    if losing:
        out[np.isin(tracks, list(losing))] = LOSING_COLOR
    if gaining:
        is_gaining = np.isin(tracks, list(gaining))
        out[is_gaining & (out == LOSING_COLOR)] = BOTH_COLOR
        out[is_gaining & (out != BOTH_COLOR)] = GAINING_COLOR
    out[tracks == MEMBRANE] = MEMBRANE
    return int24_to_RGB(out)


def _t1s_table(t1s_now):
    return {'losing_pair_cell1_track_pair_still_in_contact': [t['last_frame_in_contact'] for t in t1s_now],
            'losing_pair_cell1_track_id': [t['losing_pair'][0] for t in t1s_now],
            'losing_pair_cell2_track_id': [t['losing_pair'][1] for t in t1s_now],
            'gaining_pair_cell1_track_id_in_contact': [t['frame_nb'] for t in t1s_now],
            'gaining_pair_cell1_track_id': [t['gaining_pair'][0] for t in t1s_now],
            'gaining_pair_cell2_track_id': [t['gaining_pair'][1] for t in t1s_now]}


def _stable_t1s_table(t1s_now):
    return {'losing_pair_cell1_track_id': [t['losing_pair'][0] for t in t1s_now],
            'losing_pair_cell2_track_id': [t['losing_pair'][1] for t in t1s_now],
            'gaining_pair_cell1_track_id': [t['gaining_pair'][0] for t in t1s_now],
            'gaining_pair_cell2_track_id': [t['gaining_pair'][1] for t in t1s_now]}


def _save_frame_outputs(filename, tracks, t1s_now, stable_now):
    TA_path = smart_name_parser(filename, ordered_output='TA')
    Img(_paint_T1s(tracks, t1s_now), dimensions='hwc').save(os.path.join(TA_path, 'T1s.tif'), mode='raw')
    Img(_paint_T1s(tracks, stable_now), dimensions='hwc').save(os.path.join(TA_path, 'stable_T1s.tif'), mode='raw')
    db = TAsql(filename_or_connection=os.path.join(TA_path, 'pyTA.db'))
    try:
        for table, rows, make in (('T1s', t1s_now, _t1s_table), ('stable_T1s', stable_now, _stable_t1s_table)):
            db.drop_table(table)
            if rows:
                db.create_and_append_table(table, make(rows))
    finally:
        db.close()


def _write_csv(path, lst, t1s):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('frame_nb,filename,last_frame_in_contact,losing_pair_cell1_track_id,losing_pair_cell2_track_id,'
                'gaining_pair_cell1_track_id,gaining_pair_cell2_track_id\n')
        for t in t1s:
            f.write('%d,%s,%d,%d,%d,%d,%d\n' % (t['frame_nb'], os.path.basename(lst[t['frame_nb']]),
                                                t['last_frame_in_contact'], t['losing_pair'][0], t['losing_pair'][1],
                                                t['gaining_pair'][0], t['gaining_pair'][1]))


def detect_T1s(lst, progress_callback=None):
    """
    Detects T1 transitions in a tracked movie and saves the per frame images/tables and the movie csv files.

    Args:
        lst (list): the ordered list of images of the movie (tracked with pyTA).
        progress_callback: optional Qt signal receiving a percentage.

    Returns:
        tuple: (all T1s, stable T1s), lists of dicts (see find_T1s).
    """
    if not lst:
        return [], []
    quadruplets_per_frame = []
    tracks_per_frame = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            return [], []
        if progress_callback is not None:
            progress_callback.emit(int(i / len(lst) * 50))
        tracks = load_tracks(filename)
        tracks_per_frame.append(tracks is not None)
        quadruplets_per_frame.append(get_quadruplets(tracks) if tracks is not None else None)

    t1s = find_T1s(quadruplets_per_frame)
    stable = filter_oscillating_T1s(t1s)

    for i, filename in enumerate(lst):
        if early_stop.stop:
            return t1s, stable
        if progress_callback is not None:
            progress_callback.emit(50 + int(i / len(lst) * 50))
        if not tracks_per_frame[i]:
            continue
        _save_frame_outputs(filename, load_tracks(filename), [t for t in t1s if t['frame_nb'] == i],
                            [t for t in stable if t['frame_nb'] == i])

    parent = smart_name_parser(lst[0], ordered_output='parent')
    _write_csv(os.path.join(parent, 'T1s.csv'), lst, t1s)
    _write_csv(os.path.join(parent, 'stable_T1s.csv'), lst, stable)
    logger.info(str(len(t1s)) + ' T1s, ' + str(len(stable)) + ' stable T1s')
    return t1s, stable
