"""
Clones (groups of tracked cells) followed through a movie.

Port of the TissueAnalyzer (Java) 'Virtual cloning' tab (CloneTracker, CloneFinder, CloneMathematics).

A clone is saved in every frame as tracked_clone.tif (clone 0) or tracked_clone_XXX.tif (clone XXX), the TissueAnalyzer
format: membranes in white, cells of the clone in blue (0x0000FF), other cells in black. The cells of every clone are
also listed in the 'cell_clones' table of pyTA.db (local_id, clone_XXX = 0/1), so that clones can be used in SQL
queries/exports (NATURAL JOIN with cells_2D).

Tools:
    - track_clone: selected cells (track IDs in one frame) + their progeny (cell_divisions.csv) + their ancestors,
      followed in all frames; or, as in TissueAnalyzer (whole_lineage), the whole family of the selected cells
      (sisters, cousins... included)
    - edit_clone: adds/removes cells (by track ID) to an existing clone in all frames
    - detect_clone: cells whose area is covered above a fraction by pixels brighter than a threshold in a channel
      (e.g. a clone marker), independently in every frame
    - invert_clone, combine_clones (and/or/xor/minus)
    - store_subclone_ids: ID of the connected patch (sub-clone) of every clone cell, per frame, in the
      'local_clone_id_XXX' table (TissueAnalyzer 'store local clone ID in DB' option of the clone roughness)
Clone roughness, boundary roughness and wiggliness are in clone_shape.py.
"""
import os
import numpy as np
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img, int24_to_RGB, RGB_to_int24
from batoolset.tools.early_stopper_class import early_stop
from batoolset.ta.database.sql import TAsql
from batoolset.ta.tracking.bonds_and_vertices import load_divisions, get_bond_pixels
from batoolset.ta.tracking.divisions_and_deaths import load_tracks
from batoolset.tools.logger import TA_logger

logger = TA_logger()

MEMBRANE = 0xFFFFFF
BACKGROUND = 0
CLONE_COLOR = 0x0000FF


def clone_file_name(clone_nb):
    """TissueAnalyzer clone file names."""
    if clone_nb < 0:
        raise ValueError('clone numbers must be >= 0')
    return 'tracked_clone.tif' if clone_nb == 0 else 'tracked_clone_%03d.tif' % clone_nb


def clone_column_name(clone_nb):
    return 'clone_%03d' % clone_nb


def lineage(lst, divisions=None):
    """
    Returns (parent, birth): parent[daughter track] = mother track and birth[daughter track] = frame where the daughter
    appears, from cell_divisions.csv (a daughter keeping the mother ID is not a new track).
    """
    divisions = load_divisions(lst) if divisions is None else divisions
    parent = {}
    birth = {}
    for frame, daughters in divisions.items():
        for daughter, mother in daughters.items():
            parent[daughter] = mother
            birth[daughter] = frame
    return parent, birth


def clone_membership(selected, selection_frame, parent, birth, nb_frames, whole_lineage=False):
    """
    Returns, for every frame, the set of track IDs in the clone: the selected cells (all frames), their progeny born
    after the selection frame and their ancestors (in the frames before the corresponding division).

    Args:
        selected (iterable): track IDs selected at frame selection_frame.
        selection_frame (int): frame of the selection.
        parent (dict): daughter -> mother track (see lineage).
        birth (dict): daughter -> frame of birth.
        nb_frames (int): number of frames.
        whole_lineage (bool): TissueAnalyzer behaviour: the whole family of the selected cells (their first ancestor
            in the movie and all its progeny, i.e. also sisters, cousins... of the selected cells).
    """
    selected = set(int(s) for s in selected)
    if whole_lineage:
        roots = set()
        for cell in selected:
            while cell in parent:
                cell = parent[cell]
            roots.add(cell)
        selected = roots
        selection_frame = -1  # all the progeny of the roots
    per_frame = [set(selected) for _ in range(nb_frames)]
    # progeny (any generation) born after the selection
    children = {}
    for daughter, mother in parent.items():
        children.setdefault(mother, []).append(daughter)
    todo = list(selected)
    seen = set(selected)
    while todo:
        cell = todo.pop()
        for child in children.get(cell, []):
            if birth[child] > selection_frame and child not in seen:
                seen.add(child)
                for t in range(birth[child], nb_frames):
                    per_frame[t].add(child)
                todo.append(child)
    # ancestors, before the division that created the selected cell (or its ancestor)
    for cell in selected:
        while cell in parent:
            mother = parent[cell]
            for t in range(0, min(birth[cell], nb_frames)):
                per_frame[t].add(mother)
            cell = mother
    return per_frame


def _clone_image(tracks, cells_in_clone):
    out = np.zeros(tracks.shape, np.uint32)
    if cells_in_clone:
        out[np.isin(tracks, list(cells_in_clone))] = CLONE_COLOR
    out[tracks == MEMBRANE] = MEMBRANE
    return out


def _cell_tracks_rows(filename):
    """(local_id, track_id) rows of the cell_tracks table written by the tracking (track_id None: untracked cell)."""
    db_path = smart_name_parser(filename, ordered_output='pyTA.db')
    if not os.path.isfile(db_path):
        return []
    db = TAsql(filename_or_connection=db_path)
    try:
        if 'cell_tracks' not in db.get_tables():
            return []
        rows = db.run_SQL_command_and_get_results('SELECT local_id, track_id FROM cell_tracks')
    finally:
        db.close()
    return [(int(l), None if t is None else int(t)) for l, t in rows if l is not None]


def _local_ids(filename, tracks):
    """track ID -> local cell ID (cell_tracks table written by the tracking)."""
    return {t: l for l, t in _cell_tracks_rows(filename) if t is not None}


def save_clone(filename, tracks, cells_in_clone, clone_nb):
    """Saves the clone image and the clone column of the 'cell_clones' table of a frame."""
    TA_path = smart_name_parser(filename, ordered_output='TA')
    Img(int24_to_RGB(_clone_image(tracks, cells_in_clone)), dimensions='hwc').save(
        os.path.join(TA_path, clone_file_name(clone_nb)), mode='raw')
    _save_clone_table(filename, tracks, cells_in_clone, clone_nb)


def _save_clone_table(filename, tracks, cells_in_clone, clone_nb):
    cell_rows = _cell_tracks_rows(filename)
    if not cell_rows:
        return
    track_to_local = {t: l for l, t in cell_rows if t is not None}
    db = TAsql(filename_or_connection=smart_name_parser(filename, ordered_output='pyTA.db'))
    try:
        column = clone_column_name(clone_nb)
        existing = {}
        if 'cell_clones' in db.get_tables():
            header, rows = db.run_SQL_command_and_get_results('SELECT * FROM cell_clones', return_header=True)
            for row in rows:
                existing[row[0]] = dict(zip(header, row))
            columns = [c for c in header if c != 'local_id']
        else:
            columns = []
        if column not in columns:
            columns.append(column)
        table = {'local_id': sorted(l for l, _ in cell_rows)}  # every cell, untracked cells are never in a clone
        in_clone = {track_to_local[t] for t in cells_in_clone if t in track_to_local}
        for col in sorted(columns):
            if col == column:
                table[col] = [int(l in in_clone) for l in table['local_id']]
            else:
                table[col] = [existing.get(l, {}).get(col, 0) or 0 for l in table['local_id']]
        db.drop_table('cell_clones')
        db.create_and_append_table('cell_clones', table)
    finally:
        db.close()


def load_clone(filename, clone_nb, tracks=None):
    """Returns the set of track IDs in the clone for a frame (None if the clone file does not exist)."""
    path = os.path.join(smart_name_parser(filename, ordered_output='TA'), clone_file_name(clone_nb))
    if not os.path.isfile(path):
        return None
    clone = np.asarray(Img(path))
    if clone.ndim == 3:
        clone = RGB_to_int24(clone)
    tracks = load_tracks(filename) if tracks is None else tracks
    if tracks is None:
        return None
    return {int(t) for t in np.unique(tracks[clone == CLONE_COLOR])} - {MEMBRANE, BACKGROUND}


def subclones(tracks, cells_in_clone):
    """
    Splits a clone into sub-clones: groups of clone cells connected through shared bonds (cells touching only by a
    vertex are not connected). Sub-clones are numbered from 1, by decreasing number of cells (ties: top-most first).

    Returns:
        dict: track ID -> sub-clone ID
    """
    cells = sorted(int(c) for c in cells_in_clone)
    root = {c: c for c in cells}

    def find(c):
        while root[c] != c:
            root[c] = root[root[c]]
            c = root[c]
        return c

    for a, b in get_bond_pixels(tracks):
        if a in root and b in root:
            root[find(a)] = find(b)
    groups = {}
    for c in cells:
        groups.setdefault(find(c), []).append(c)
    ys, xs = np.nonzero(np.isin(tracks, cells))
    top = {}
    for t, y in zip(tracks[ys, xs].tolist(), ys.tolist()):
        if t not in top or y < top[t]:
            top[t] = y
    order = sorted(groups.values(), key=lambda g: (-len(g), min(top.get(c, 0) for c in g), g[0]))
    return {c: i + 1 for i, g in enumerate(order) for c in g}


def subclone_table_name(clone_nb):
    """TissueAnalyzer name of the sub-clone table."""
    return 'local_clone_id_%03d' % clone_nb


def save_subclone_ids(filename, clone_nb=0, tracks=None):
    """
    Saves the sub-clone ID of every cell of the clone in the 'local_clone_id_XXX' table of pyTA.db (TissueAnalyzer
    columns: local_id_cells, local_clone_id).

    Returns:
        dict: track ID -> sub-clone ID (None if the clone or the tracks do not exist)
    """
    tracks = load_tracks(filename) if tracks is None else tracks
    if tracks is None:
        return None
    cells_in_clone = load_clone(filename, clone_nb, tracks)
    if cells_in_clone is None:
        logger.error('Clone ' + str(clone_nb) + ' does not exist for ' + str(filename))
        return None
    ids = subclones(tracks, cells_in_clone)
    track_to_local = _local_ids(filename, tracks)
    if not track_to_local:
        logger.error('No cell_tracks table for ' + str(filename) + ', please track cells first')
        return ids
    rows = sorted((track_to_local[t], s) for t, s in ids.items() if t in track_to_local)
    db = TAsql(filename_or_connection=smart_name_parser(filename, ordered_output='pyTA.db'))
    try:
        name = subclone_table_name(clone_nb)
        db.drop_table(name)
        if rows:
            db.create_and_append_table(name, {'local_id_cells': [r[0] for r in rows],
                                              'local_clone_id': [r[1] for r in rows]})
    finally:
        db.close()
    return ids


def store_subclone_ids(lst, clone_nb=0, progress_callback=None):
    """save_subclone_ids for every frame."""
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        save_subclone_ids(filename, clone_nb)


def _progress(progress_callback, value):
    if progress_callback is not None:
        progress_callback.emit(int(value))


def track_clone(lst, selected, selection_frame, clone_nb=0, whole_lineage=False, progress_callback=None):
    """
    Creates clone clone_nb from cells selected in one frame (see clone_membership) and saves it in all frames.

    Returns:
        list: for every frame, the set of track IDs in the clone.
    """
    parent, birth = lineage(lst)
    per_frame = clone_membership(selected, selection_frame, parent, birth, len(lst), whole_lineage=whole_lineage)
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        tracks = load_tracks(filename)
        if tracks is None:
            continue
        save_clone(filename, tracks, per_frame[i], clone_nb)
    return per_frame


def edit_clone(lst, cells_to_add=(), cells_to_remove=(), clone_nb=0, progress_callback=None):
    """
    Adds/removes cells (track IDs) to/from an existing clone in all frames (cells are added in all the frames where
    they exist, as in TissueAnalyzer).
    """
    cells_to_add = set(cells_to_add)
    cells_to_remove = set(cells_to_remove)
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        tracks = load_tracks(filename)
        if tracks is None:
            continue
        cells = load_clone(filename, clone_nb, tracks) or set()
        save_clone(filename, tracks, (cells | cells_to_add) - cells_to_remove, clone_nb)


def cells_drawn_over(tracks, drawing):
    """Track IDs of the cells touched by a drawing (non zero pixels of a 2D mask)."""
    drawing = np.asarray(drawing)
    if drawing.ndim == 3:
        drawing = drawing.any(axis=-1)
    return {int(t) for t in np.unique(tracks[drawing != 0])} - {MEMBRANE, BACKGROUND}


def detect_clone(lst, channel=0, threshold=128, min_fraction=0.5, clone_nb=0, progress_callback=None):
    """
    Clone = cells with more than min_fraction of their pixels >= threshold in a channel of the original image, detected
    independently in every frame (TissueAnalyzer 'Autodetect').
    """
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        tracks = load_tracks(filename)
        if tracks is None:
            continue
        img = np.asarray(Img(filename))
        if img.ndim == 3:
            img = img[..., min(channel, img.shape[-1] - 1)]
        if img.shape != tracks.shape:
            logger.error('Image and tracks sizes differ for ' + str(filename))
            continue
        cells = (tracks != MEMBRANE) & (tracks != BACKGROUND)
        ids, inverse = np.unique(tracks[cells], return_inverse=True)
        area = np.bincount(inverse)
        bright = np.bincount(inverse, weights=(img[cells] >= threshold).astype(float), minlength=ids.size)
        save_clone(filename, tracks, {int(t) for t in ids[bright / area > min_fraction]}, clone_nb)


def invert_clone(lst, clone_nb_input=0, clone_nb_output=1, progress_callback=None):
    """clone_nb_output = the cells that are not in clone_nb_input."""
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        tracks = load_tracks(filename)
        if tracks is None:
            continue
        cells = load_clone(filename, clone_nb_input, tracks)
        if cells is None:
            continue
        all_cells = {int(t) for t in np.unique(tracks)} - {MEMBRANE, BACKGROUND}
        save_clone(filename, tracks, all_cells - cells, clone_nb_output)


OPERATIONS = {'and': lambda a, b: a & b, 'or': lambda a, b: a | b, 'xor': lambda a, b: a ^ b,
              'minus': lambda a, b: a - b}


def combine_clones(lst, operation, clone_nb_1, clone_nb_2, clone_nb_output, progress_callback=None):
    """clone_nb_output = clone_nb_1 <operation> clone_nb_2 (operation: 'and', 'or', 'xor', 'minus'), per frame."""
    op = OPERATIONS[operation]
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        tracks = load_tracks(filename)
        if tracks is None:
            continue
        a = load_clone(filename, clone_nb_1, tracks)
        b = load_clone(filename, clone_nb_2, tracks)
        if a is None and b is None:
            continue
        save_clone(filename, tracks, op(a or set(), b or set()), clone_nb_output)
