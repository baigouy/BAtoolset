"""
Tracking of bonds (cell-cell junctions) and vertices from tracked cells.

Port of TissueAnalyzer (Java) 'Track bonds' (TrackCellsV1.trackbondsv3_light/trackbondsv4, IgnoreCellDivisions,
reprocess_dividing_bonds) and 'Track Vertices' (TrackVertices).

A bond is identified by the pair of cell tracks it separates (a border bond by (cell, 0)). A bond keeps its track ID as
long as the same two cells stay in contact in consecutive frames, a new ID is given when the contact appears (or
reappears). When a cell divides, the bonds of the daughters that replace a bond of the mother are its progeny (parent
bond), the bond between the two sister cells is a 'division' bond. Every bond track gets a fate, as in TissueAnalyzer:
    - stable: present from the first to the last frame (itself or its progeny)
    - appearing: appears during the movie and stays until the last frame
    - disappearing: present in the first frame and disappears
    - transient: appears and disappears during the movie
    - division: created between two sister cells by a division (and its progeny)

A vertex is identified by the sorted tracks of the cells (>= 3) around it; the same set of cells always gets the same
vertex track ID (as in TissueAnalyzer).

Outputs, per frame (in the TA folder, visible in the pyTA preview):
    - tracked_bonds.tif (one color per bond track, vertices in white), bond_fate.tif (fate colors of TissueAnalyzer)
    - tracked_vertices.tif (one color per vertex track)
    - 'bond_tracks' and 'vertex_tracks' tables in pyTA.db; 'local_id' links them to bonds_2D/vertices_2D when the cells
      have been measured (so that they can be NATURAL JOINed)
And for the whole movie, next to the images: bond_tracks.csv and vertex_tracks.csv
"""
import os
import numpy as np
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img, int24_to_RGB
from batoolset.tools.early_stopper_class import early_stop
from batoolset.ta.database.sql import TAsql
from batoolset.ta.tracking.divisions_and_deaths import load_tracks
from batoolset.tools.logger import TA_logger

logger = TA_logger()

MEMBRANE = 0xFFFFFF
BACKGROUND = 0
_NONE = np.uint32(0xFFFFFFFF)  # sorts after every int24 value

# TissueAnalyzer bond fate colors
FATE_COLORS = {'stable': 0xFF0000, 'appearing': 0x00FF33, 'disappearing': 0xFFCC00, 'transient': 0xFF99CC,
               'division': 0x6666FF}


def id_to_color(ids):
    """
    Returns a distinct int24 color for every track ID (bijective for IDs < 2^24, never black; never white for the IDs
    used here).
    """
    ids = np.asarray(ids, dtype=np.uint64)
    colors = (ids * np.uint64(2654435761)) & np.uint64(0xFFFFFF)
    colors[colors == MEMBRANE] = 0xFFFFFE
    colors[(colors == 0) & (ids != 0)] = 1
    return colors.astype(np.uint32)


def membrane_contacts(tracks):
    """
    For every membrane pixel, the sorted distinct cell tracks in its 3x3 neighbourhood.

    Args:
        tracks (ndarray): int24 track image (membrane 0xFFFFFF, background 0).

    Returns:
        tuple: ys, xs, cells (N x 8 uint32, sorted, missing values = 0xFFFFFFFF), nb_cells (N), touches_background (N)
    """
    tracks = np.asarray(tracks, dtype=np.uint32)
    ys, xs = np.nonzero(tracks == MEMBRANE)
    padded = np.pad(tracks, 1, mode='constant', constant_values=BACKGROUND)
    neighbours = np.empty((ys.size, 8), np.uint32)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbours[:, k] = padded[ys + 1 + dy, xs + 1 + dx]
            k += 1
    touches_background = (neighbours == BACKGROUND).any(axis=1)
    neighbours[(neighbours == BACKGROUND) | (neighbours == MEMBRANE)] = _NONE
    neighbours.sort(axis=1)
    duplicate = np.zeros_like(neighbours, dtype=bool)
    duplicate[:, 1:] = neighbours[:, 1:] == neighbours[:, :-1]
    neighbours[duplicate] = _NONE
    neighbours.sort(axis=1)
    nb_cells = (neighbours != _NONE).sum(axis=1)
    return ys, xs, neighbours, nb_cells, touches_background


def get_bond_pixels(tracks, contacts=None):
    """
    Returns a dict (cell1, cell2) -> (ys, xs) of the membrane pixels separating two cells (cell1 < cell2). Border bonds
    are (0, cell): the membrane between a cell and the background and, as in the pyTA measurements (bonds_2D), the
    pixels of a cell lying on the image edge.
    """
    tracks = np.asarray(tracks, dtype=np.uint32)
    ys, xs, cells, nb_cells, touches_background = contacts if contacts is not None else membrane_contacts(tracks)
    two = nb_cells == 2
    border = (nb_cells == 1) & touches_background
    keys = np.zeros((ys.size, 2), np.uint32)
    keys[two] = cells[two, :2]
    keys[border, 1] = cells[border, 0]
    edge = np.zeros(tracks.shape, bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge &= (tracks != MEMBRANE) & (tracks != BACKGROUND)
    eys, exs = np.nonzero(edge)
    ekeys = np.zeros((eys.size, 2), np.uint32)
    ekeys[:, 1] = tracks[eys, exs]
    ys = np.concatenate([ys, eys])
    xs = np.concatenate([xs, exs])
    keys = np.concatenate([keys, ekeys])
    sel = np.nonzero(np.concatenate([two | border, np.ones(eys.size, bool)]))[0]
    if sel.size == 0:
        return {}
    code = keys[sel, 0].astype(np.uint64) << np.uint64(32) | keys[sel, 1].astype(np.uint64)
    order = np.argsort(code, kind='stable')
    code = code[order]
    sel = sel[order]
    starts = np.r_[0, np.nonzero(np.diff(code))[0] + 1]
    ends = np.r_[starts[1:], code.size]
    out = {}
    for s, e in zip(starts, ends):
        idx = sel[s:e]
        out[(int(keys[idx[0], 0]), int(keys[idx[0], 1]))] = (ys[idx], xs[idx])
    # two cells can also be in contact only through two adjacent vertices (bonds_2D bonds of 2 px)
    for pair, pixels in _vertex_pair_bonds(contacts if contacts is not None else membrane_contacts(tracks)).items():
        if pair not in out:
            out[pair] = pixels
    return out


def _vertex_pair_bonds(contacts):
    """(cell1, cell2) -> (ys, xs) for the cells sharing exactly two adjacent vertex pixels."""
    ys, xs, cells, nb_cells, _ = contacts
    vertex_idx = np.nonzero(nb_cells >= 3)[0]
    position = {(int(ys[i]), int(xs[i])): i for i in vertex_idx}
    out = {}
    for i in vertex_idx:
        key_i = set(int(c) for c in cells[i, :nb_cells[i]])
        y, x = int(ys[i]), int(xs[i])
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = position.get((y + dy, x + dx))
            if j is None:
                continue
            common = key_i & set(int(c) for c in cells[j, :nb_cells[j]])
            if len(common) == 2:
                pair = tuple(sorted(common))
                if pair not in out:
                    out[pair] = (np.asarray([y, y + dy]), np.asarray([x, x + dx]))
    return out


def get_vertices(tracks, contacts=None):
    """
    Returns a dict tuple(sorted cells, >= 3) -> (ys, xs) of the vertex pixels.
    """
    ys, xs, cells, nb_cells, _ = contacts if contacts is not None else membrane_contacts(tracks)
    out = {}
    for i in np.nonzero(nb_cells >= 3)[0]:
        key = tuple(int(c) for c in cells[i, :nb_cells[i]])
        py, px = out.setdefault(key, ([], []))
        py.append(ys[i])
        px.append(xs[i])
    return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in out.items()}


def load_divisions(lst):
    """
    Reads cell_divisions.csv (written by 'Detect cell divisions and deaths') next to the images.

    Returns:
        dict: frame_nb -> {daughter_track: mother_track} for the daughters appearing at that frame (empty if the file is
        missing).
    """
    divisions = {}
    if not lst:
        return divisions
    path = os.path.join(smart_name_parser(lst[0], ordered_output='parent'), 'cell_divisions.csv')
    if not os.path.isfile(path):
        logger.warning('No cell_divisions.csv found (run "Detect cell divisions and deaths" first), divisions will be '
                       'ignored: bonds created by divisions will be considered as appearing bonds')
        return divisions
    import csv
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                frame = int(row['frame_nb'])
                mother = int(row['mother_track_id'])
                daughters = divisions.setdefault(frame, {})
                for key in ('daughter1_track_id', 'daughter2_track_id'):
                    daughter = int(row[key])
                    if daughter != mother:
                        daughters[daughter] = mother
            except (KeyError, ValueError):
                continue
    return divisions


def track_bonds(bond_pairs_per_frame, divisions=None):
    """
    Assigns track IDs, parents and fates to bonds (see module doc).

    Args:
        bond_pairs_per_frame (list): for every frame, an iterable of (cell1, cell2) pairs (None for a missing frame).
        divisions (dict): frame_nb -> {daughter: mother}, see load_divisions.

    Returns:
        tuple: (ids_per_frame: list of dicts pair -> track ID, tracks: dict track ID -> info dict with 'pair',
        'first_frame', 'last_frame', 'parent', 'children', 'is_division_bond', 'fate')
    """
    divisions = divisions or {}
    ids_per_frame = []
    tracks = {}
    prev = {}
    next_id = 1
    for t, pairs in enumerate(bond_pairs_per_frame):
        cur = {}
        if pairs is None:
            ids_per_frame.append(cur)
            prev = cur
            continue
        mothers = divisions.get(t, {})
        for pair in sorted(pairs):
            if pair in prev:
                track_id = prev[pair]
                cur[pair] = track_id
                tracks[track_id]['last_frame'] = t
                continue
            track_id = next_id
            next_id += 1
            info = {'pair': pair, 'first_frame': t, 'last_frame': t, 'parent': None, 'children': [],
                    'is_division_bond': False}
            if mothers:
                a, b = (mothers.get(c, c) if c != BACKGROUND else c for c in pair)
                if (a, b) != pair:
                    if a == b:
                        info['is_division_bond'] = True
                    else:
                        parent = prev.get((min(a, b), max(a, b)))
                        if parent is not None:
                            info['parent'] = parent
                            tracks[parent]['children'].append(track_id)
            tracks[track_id] = info
            cur[pair] = track_id
        ids_per_frame.append(cur)
        prev = cur

    last = len(bond_pairs_per_frame) - 1
    first = next((t for t, p in enumerate(bond_pairs_per_frame) if p is not None), 0)
    last = next((t for t in range(last, -1, -1) if bond_pairs_per_frame[t] is not None), last)
    reaches_end = {}
    for track_id in sorted(tracks, reverse=True):  # children always have larger IDs than their parent
        info = tracks[track_id]
        reaches_end[track_id] = info['last_frame'] == last or any(reaches_end[c] for c in info['children'])
    for track_id in sorted(tracks):
        info = tracks[track_id]
        parent = info['parent']
        if parent is not None:
            info['root'] = tracks[parent]['root']
        else:
            info['root'] = track_id
        root = tracks[info['root']]
        appeared = root['first_frame'] != first
        disappears = not reaches_end[track_id]
        if root['is_division_bond']:
            fate = 'division'
        elif appeared and disappears:
            fate = 'transient'
        elif appeared:
            fate = 'appearing'
        elif disappears:
            fate = 'disappearing'
        else:
            fate = 'stable'
        info['fate'] = fate
    return ids_per_frame, tracks


def track_vertices(vertex_keys_per_frame):
    """
    Returns a list (per frame) of dicts vertex key -> vertex track ID; a given set of cells always gets the same ID.
    """
    ids = {}
    out = []
    for keys in vertex_keys_per_frame:
        cur = {}
        if keys is not None:
            for key in sorted(keys):
                if key not in ids:
                    ids[key] = len(ids) + 1
                cur[key] = ids[key]
        out.append(cur)
    return out


def _majority(keys, values):
    """dict key -> most frequent value."""
    if keys.size == 0:
        return {}
    pairs = np.stack([np.asarray(keys, np.int64), np.asarray(values, np.int64)], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((-counts, uniq[:, 0]))
    uniq = uniq[order]
    first = np.r_[True, uniq[1:, 0] != uniq[:-1, 0]]
    return {int(k): int(v) for k, v in uniq[first]}


def _read_table(db_path, table, sql):
    if not os.path.isfile(db_path):
        return None
    db = TAsql(filename_or_connection=db_path)
    try:
        if table not in db.get_tables():
            return None
        return db.run_SQL_command_and_get_results(sql)
    except Exception:
        return None
    finally:
        db.close()


def _bond_local_ids(filename, bond_pixels, vertices, tracks):
    """
    local bond ID (bonds_2D) -> bond pair: majority of the pixels of bonds.tif; for the bonds that are not in bonds.tif
    (bonds made of two adjacent vertices), the two cells around both of their vertices (vx_1, vx_2 of vertices_2D).
    """
    TA_path = smart_name_parser(filename, ordered_output='TA')
    out = {}
    bonds_path = os.path.join(TA_path, 'bonds.tif')
    if os.path.isfile(bonds_path):
        bonds = np.asarray(Img(bonds_path)).astype(np.int64)
        if bonds.shape == tracks.shape:
            pair_codes = {}
            local, codes = [], []
            for n, (pair, (ys, xs)) in enumerate(bond_pixels.items()):
                pair_codes[n] = pair
                ids = bonds[ys, xs]
                ok = ids > 0
                local.append(ids[ok])
                codes.append(np.full(int(ok.sum()), n, np.int64))
            if local:
                for local_id, n in _majority(np.concatenate(local), np.concatenate(codes)).items():
                    out[local_id] = pair_codes[n]
    db_path = os.path.join(TA_path, 'pyTA.db')
    rows = _read_table(db_path, 'bonds_2D', 'SELECT local_ID, vx_1, vx_2 FROM bonds_2D')
    if rows:
        measured = {int(r[0]) for r in rows}
        vertex_positions = None
        for local_id, vx_1, vx_2 in rows:
            if local_id in out or vx_1 is None or vx_2 is None or vx_1 == vx_2:
                continue
            if vertex_positions is None:
                vertex_rows = _read_table(db_path, 'vertices_2D', 'SELECT local_ID, y, x FROM vertices_2D') or []
                vertex_positions = {int(v): (int(y), int(x)) for v, y, x in vertex_rows}
            p1 = vertex_positions.get(int(vx_1))
            p2 = vertex_positions.get(int(vx_2))
            if p1 is None or p2 is None:
                continue
            common = _cells_around(tracks, *p1) & _cells_around(tracks, *p2)
            if len(common) == 2:
                out[int(local_id)] = tuple(sorted(common))
        out = {k: v for k, v in out.items() if k in measured}
    return out


def _cells_around(tracks, y, x):
    patch = tracks[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    return {int(c) for c in np.unique(patch)} - {MEMBRANE, BACKGROUND}


def _vertex_local_ids(filename, vertices, tracks):
    """local vertex ID (vertices_2D / vertices.tif) -> vertex key."""
    TA_path = smart_name_parser(filename, ordered_output='TA')
    path = os.path.join(TA_path, 'vertices.tif')
    if not os.path.isfile(path):
        return {}
    local = np.asarray(Img(path)).astype(np.int64)
    if local.shape != tracks.shape:
        return {}
    out = {}
    for key, (ys, xs) in vertices.items():
        for local_id in np.unique(local[ys, xs]):
            if local_id > 0:
                out[int(local_id)] = key
    return out


def _save_bond_frame(filename, tracks, bond_pixels, vertices, ids, bond_infos):
    TA_path = smart_name_parser(filename, ordered_output='TA')
    tracked = np.zeros(tracks.shape, np.uint32)
    fate = np.zeros(tracks.shape, np.uint32)
    for pair, (ys, xs) in bond_pixels.items():
        track_id = ids[pair]
        tracked[ys, xs] = id_to_color([track_id])[0]
        fate[ys, xs] = FATE_COLORS[bond_infos[track_id]['fate']]
    for ys, xs in vertices.values():
        tracked[ys, xs] = MEMBRANE
        fate[ys, xs] = MEMBRANE
    Img(int24_to_RGB(tracked), dimensions='hwc').save(os.path.join(TA_path, 'tracked_bonds.tif'), mode='raw')
    Img(int24_to_RGB(fate), dimensions='hwc').save(os.path.join(TA_path, 'bond_fate.tif'), mode='raw')

    local_ids = _bond_local_ids(filename, bond_pixels, vertices, tracks)
    table = {'local_id': [], 'track_id': [], 'cell_track_id1': [], 'cell_track_id2': [], 'parent_track_id': [],
             'bond_fate': []}
    linked = set()

    def add(local_id, pair):
        track_id = ids.get(pair)
        info = bond_infos.get(track_id) if track_id is not None else None
        table['local_id'].append(local_id)
        table['track_id'].append(track_id)
        table['cell_track_id1'].append(pair[0] if pair is not None else None)
        table['cell_track_id2'].append(pair[1] if pair is not None else None)
        table['parent_track_id'].append(info['parent'] if info else None)
        table['bond_fate'].append(info['fate'] if info else None)
        if track_id is not None:
            linked.add(pair)

    for local_id in sorted(local_ids):
        add(local_id, local_ids[local_id])
    for pair in sorted(ids):
        if pair not in linked:
            add(None, pair)
    _save_table(os.path.join(TA_path, 'pyTA.db'), 'bond_tracks', table)


def _save_vertex_frame(filename, tracks, vertices, vertex_ids):
    TA_path = smart_name_parser(filename, ordered_output='TA')
    tracked = np.zeros(tracks.shape, np.uint32)
    for key, (ys, xs) in vertices.items():
        tracked[ys, xs] = id_to_color([vertex_ids[key]])[0]
    Img(int24_to_RGB(tracked), dimensions='hwc').save(os.path.join(TA_path, 'tracked_vertices.tif'), mode='raw')

    local_ids = _vertex_local_ids(filename, vertices, tracks)
    key_to_local = {}
    for local_id, key in sorted(local_ids.items()):
        key_to_local.setdefault(key, []).append(local_id)
    # no x/y columns: they would clash with those of vertices_2D in NATURAL JOINs
    table = {'local_id': [], 'track_id': [], 'mean_x': [], 'mean_y': [], 'nb_of_cells': [], 'cell_track_ids': []}
    for key in sorted(vertices, key=lambda k: vertex_ids[k]):
        ys, xs = vertices[key]
        for local_id in key_to_local.get(key, [None]):
            table['local_id'].append(local_id)
            table['track_id'].append(vertex_ids[key])
            table['mean_x'].append(float(xs.mean()))
            table['mean_y'].append(float(ys.mean()))
            table['nb_of_cells'].append(len(key))
            table['cell_track_ids'].append(' '.join(str(c) for c in key))
    _save_table(os.path.join(TA_path, 'pyTA.db'), 'vertex_tracks', table)


def _save_table(db_path, name, table):
    db = TAsql(filename_or_connection=db_path)
    try:
        db.drop_table(name)
        if table and next(iter(table.values())):
            db.create_and_append_table(name, table)
    finally:
        db.close()


def track_bonds_and_vertices(lst, bonds=True, vertices=True, progress_callback=None):
    """
    Tracks the bonds and/or the vertices of a tracked movie and saves the per frame images/tables and the movie csv
    files (see module doc).

    Args:
        lst (list): the ordered list of images of the movie (tracked with pyTA).
        bonds (bool): track bonds.
        vertices (bool): track vertices.
        progress_callback: optional Qt signal receiving a percentage.

    Returns:
        tuple: (bond tracks dict or None, list per frame of vertex key -> vertex track ID or None)
    """
    if not lst:
        return None, None

    def progress(value):
        if progress_callback is not None:
            progress_callback.emit(int(value))

    pairs_per_frame = []
    vertex_keys_per_frame = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            return None, None
        progress(i / len(lst) * 40)
        tracks = load_tracks(filename)
        if tracks is None:
            pairs_per_frame.append(None)
            vertex_keys_per_frame.append(None)
            continue
        contacts = membrane_contacts(tracks)
        pairs_per_frame.append(set(get_bond_pixels(tracks, contacts)) if bonds else None)
        vertex_keys_per_frame.append(set(get_vertices(tracks, contacts)) if vertices else None)

    bond_ids = bond_infos = vertex_ids = None
    if bonds:
        bond_ids, bond_infos = track_bonds(pairs_per_frame, load_divisions(lst))
    if vertices:
        vertex_ids = track_vertices(vertex_keys_per_frame)

    vertex_rows = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            return bond_infos, vertex_ids
        progress(40 + i / len(lst) * 60)
        if (not bonds or pairs_per_frame[i] is None) and (not vertices or vertex_keys_per_frame[i] is None):
            continue
        tracks = load_tracks(filename)
        contacts = membrane_contacts(tracks)
        frame_vertices = get_vertices(tracks, contacts)
        if bonds:
            _save_bond_frame(filename, tracks, get_bond_pixels(tracks, contacts), frame_vertices, bond_ids[i],
                             bond_infos)
        if vertices:
            _save_vertex_frame(filename, tracks, frame_vertices, vertex_ids[i])
            for key, (ys, xs) in frame_vertices.items():
                vertex_rows.append((i, os.path.basename(filename), vertex_ids[i][key], float(xs.mean()),
                                    float(ys.mean()), ' '.join(str(c) for c in key)))

    parent = smart_name_parser(lst[0], ordered_output='parent')
    if bonds:
        with open(os.path.join(parent, 'bond_tracks.csv'), 'w', encoding='utf-8') as f:
            f.write('track_id,cell_track_id1,cell_track_id2,first_frame,last_frame,parent_track_id,is_border,'
                    'bond_fate\n')
            for track_id, info in sorted(bond_infos.items()):
                f.write('%d,%d,%d,%d,%d,%s,%d,%s\n' % (track_id, info['pair'][0], info['pair'][1], info['first_frame'],
                                                      info['last_frame'],
                                                      '' if info['parent'] is None else info['parent'],
                                                      int(info['pair'][0] == BACKGROUND), info['fate']))
        counts = {}
        for info in bond_infos.values():
            counts[info['fate']] = counts.get(info['fate'], 0) + 1
        logger.info(str(len(bond_infos)) + ' bond tracks: ' + ', '.join(
            str(v) + ' ' + k for k, v in sorted(counts.items())))
    if vertices:
        with open(os.path.join(parent, 'vertex_tracks.csv'), 'w', encoding='utf-8') as f:
            f.write('frame_nb,filename,track_id,x,y,cell_track_ids\n')
            for row in vertex_rows:
                f.write('%d,%s,%d,%.2f,%.2f,%s\n' % row)
        logger.info(str(max((max(v.values(), default=0) for v in vertex_ids), default=0)) + ' vertex tracks')
    return bond_infos, vertex_ids
