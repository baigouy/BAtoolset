"""
Detection of cell divisions and cell deaths (delamination/apoptosis) from tracked cells.

Works on the 'tracked_cells_resized.tif' images produced by the pyTA trackers (one RGB colour = one track ID,
membranes are white 0xFFFFFF, 0 is background). Python port and extension of the TissueAnalyzer (Java)
TrackCellsV1.find_cell_divisions and TrackDyingCells.

Divisions: a track appearing at frame t (never seen before) whose neighbour at t existed at t-1 and occupied the
new cell's pixels at t-1 is a daughter of that neighbour (the other daughter keeps the mother's track ID, as the
pyTA trackers do). Pairs are filtered by daughter size ratio, area conservation, persistence of the new track and
optionally border cells.

Deaths: a track whose last frame is t < last frame of the movie, that never reappears, that is small (shrunk
relative to its own maximum area and/or below an absolute area) and that is not a border cell.

Outputs, per frame (in the TA folder, visible in the pyTA preview):
    - dividing_cells.tif: daughter cells of divisions occurring at this frame painted blue (TA convention)
    - dying_cells.tif: cells whose track dies later in the movie (or now) painted blue (TA convention)
    - 'cell_fate' table in pyTA.db (one row per cell): local_id, track_id, is_newborn_daughter,
      sister_track_id, mother_track_id, division_id, is_dying (last frame of a dying track), will_die
And for the whole movie, next to the images: cell_divisions.csv and cell_deaths.csv
"""
import os
import traceback
import numpy as np
from skimage.measure import label, regionprops
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img, RGB_to_int24, int24_to_RGB
from batoolset.tools.early_stopper_class import early_stop
from batoolset.ta.database.sql import TAsql
from batoolset.tools.logger import TA_logger

logger = TA_logger()

MEMBRANE = 0xFFFFFF
BACKGROUND = 0
DIVISION_COLOR = 0x0000FF
DEATH_COLOR = 0x0000FF


def load_tracks(filename):
    """
    Loads the tracked cells of an image as an int24 track ID image.

    Args:
        filename (str): The original image name (the TA folder is inferred from it).

    Returns:
        ndarray: int24 image of track IDs or None if the tracking file is missing.
    """
    tracked_image_path = smart_name_parser(filename, ordered_output='tracked_cells_resized.tif')
    if not os.path.isfile(tracked_image_path):
        logger.error('File not found ' + str(tracked_image_path) + ' please track cells first')
        return None
    tracks = Img(tracked_image_path)
    if tracks.ndim == 3:
        tracks = RGB_to_int24(tracks)
    return np.asarray(tracks).astype(np.uint32)


def get_areas(tracks):
    """
    Returns a dict track_id -> area in pixels (membranes and background excluded).
    """
    ids, counts = np.unique(tracks, return_counts=True)
    keep = (ids != MEMBRANE) & (ids != BACKGROUND)
    return dict(zip(ids[keep].tolist(), counts[keep].tolist()))


def get_neighbors(tracks):
    """
    Finds cells sharing a (one pixel wide) membrane by comparing the labels on both sides of every membrane pixel.

    Args:
        tracks (ndarray): int24 image of track IDs.

    Returns:
        tuple: (dict track_id -> set of neighbouring track_ids, set of track_ids touching the background)
    """
    padded = np.pad(tracks, 1, mode='constant', constant_values=BACKGROUND)
    center = padded[1:-1, 1:-1]
    is_membrane = center == MEMBRANE
    h, w = tracks.shape
    pairs = []
    # opposite pixels across the membrane: horizontal, vertical and both diagonals
    for (dy1, dx1), (dy2, dx2) in [((0, -1), (0, 1)), ((-1, 0), (1, 0)), ((-1, -1), (1, 1)), ((-1, 1), (1, -1))]:
        a = padded[1 + dy1:1 + dy1 + h, 1 + dx1:1 + dx1 + w][is_membrane]
        b = padded[1 + dy2:1 + dy2 + h, 1 + dx2:1 + dx2 + w][is_membrane]
        valid = (a != b) & (a != MEMBRANE) & (b != MEMBRANE)
        pairs.append(np.stack([a[valid], b[valid]], axis=1))
    # also direct contacts (in case some cells touch without membrane in between)
    for a, b in [(center[:, :-1], center[:, 1:]), (center[:-1, :], center[1:, :])]:
        valid = (a != b) & (a != MEMBRANE) & (b != MEMBRANE)
        pairs.append(np.stack([a[valid], b[valid]], axis=1))
    pairs = np.unique(np.concatenate(pairs, axis=0), axis=0)

    neighbors = {}
    touching_background = set()
    for a, b in pairs.tolist():
        if a == BACKGROUND:
            touching_background.add(b)
            continue
        if b == BACKGROUND:
            touching_background.add(a)
            continue
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    return neighbors, touching_background


def get_border_cells(tracks, neighbors=None, touching_background=None, plus_one=False):
    """
    Returns the set of cells touching the image border or the background (and optionally their neighbours).
    """
    edges = np.concatenate([tracks[:2, :].ravel(), tracks[-2:, :].ravel(), tracks[:, :2].ravel(), tracks[:, -2:].ravel()])
    border = set(np.unique(edges).tolist())
    if touching_background is None or (plus_one and neighbors is None):
        neighbors, touching_background = get_neighbors(tracks)
    border |= touching_background
    border -= {MEMBRANE, BACKGROUND}
    if plus_one:
        extra = set()
        for cell in border:
            extra |= neighbors.get(cell, set())
        border |= extra
    return border


def _jaccard(a, b):
    if not a and not b:
        return 0.
    return len(a & b) / len(a | b)


def detect_divisions_between(tracks_prev, tracks_cur, info_prev, info_cur, already_seen_tracks, tracks_next_ids=None,
                             min_daughter_area_ratio=0.25, mother_area_conservation=(0.5, 2.0),
                             exclude_border_cells=True, min_score=0.2):
    """
    Detects the divisions occurring between two consecutive frames.

    Args:
        tracks_prev (ndarray): int24 track IDs at t-1.
        tracks_cur (ndarray): int24 track IDs at t.
        info_prev (dict): frame info at t-1 (see _frame_info).
        info_cur (dict): frame info at t.
        already_seen_tracks (set): all track IDs present in frames < t (a reappearing track is not a new cell).
        tracks_next_ids (set, optional): track IDs at t+1; if provided the new daughter must still exist at t+1.
        min_daughter_area_ratio (float): smallest daughter area / biggest daughter area (TA used 0.25).
        mother_area_conservation (tuple): allowed range for (sum of daughter areas) / (mother area at t-1), None to skip.
        exclude_border_cells (bool): ignore divisions involving a border cell.
        min_score (float): minimal score (overlap fraction + neighbourhood similarity) to accept a pair.

    Returns:
        list: list of dicts with keys mother_track_id, daughter1_track_id (keeps the mother's ID),
              daughter2_track_id (new track), areas and score.
    """
    areas_prev = info_prev['areas']
    areas_cur = info_cur['areas']
    neighbors_prev = info_prev['neighbors']
    neighbors_cur = info_cur['neighbors']
    border_cells = info_cur['border'] if exclude_border_cells else set()

    new_cells = set(areas_cur.keys()) - set(areas_prev.keys()) - already_seen_tracks
    if tracks_next_ids is not None:
        new_cells &= tracks_next_ids

    best_per_mother = {}
    for new_cell in new_cells:
        candidates = [c for c in neighbors_cur.get(new_cell, set()) if c in areas_prev and c not in new_cells]
        if not candidates:
            continue
        # what was below the new cell in the previous frame
        below = tracks_prev[tracks_cur == new_cell]
        below_ids, below_counts = np.unique(below, return_counts=True)
        overlap = dict(zip(below_ids.tolist(), (below_counts / below.size).tolist()))

        best = None
        for candidate in candidates:
            # neighbours of the mother before division vs neighbours of both daughters after
            neighbs_after = (neighbors_cur.get(candidate, set()) | neighbors_cur.get(new_cell, set())) - {candidate, new_cell}
            score = overlap.get(candidate, 0.) + _jaccard(neighbors_prev.get(candidate, set()), neighbs_after)
            if best is None or score > best[1]:
                best = (candidate, score)
        mother, score = best
        if score < min_score:
            continue
        if mother in border_cells or new_cell in border_cells:
            continue
        area_d1 = areas_cur[mother]
        area_d2 = areas_cur[new_cell]
        if min(area_d1, area_d2) / max(area_d1, area_d2) < min_daughter_area_ratio:
            continue
        area_mother = areas_prev[mother]
        if mother_area_conservation is not None:
            conservation = (area_d1 + area_d2) / area_mother
            if not mother_area_conservation[0] <= conservation <= mother_area_conservation[1]:
                continue
        division = {'mother_track_id': mother, 'daughter1_track_id': mother, 'daughter2_track_id': new_cell,
                    'mother_area': area_mother, 'daughter1_area': area_d1, 'daughter2_area': area_d2, 'score': score}
        # a mother can only divide once per frame --> keep the best candidate
        if mother not in best_per_mother or score > best_per_mother[mother]['score']:
            best_per_mother[mother] = division
    return list(best_per_mother.values())


def _frame_info(tracks):
    neighbors, touching_background = get_neighbors(tracks)
    return {'areas': get_areas(tracks), 'neighbors': neighbors,
            'border': get_border_cells(tracks, neighbors, touching_background, plus_one=False),
            'border_plus_one': get_border_cells(tracks, neighbors, touching_background, plus_one=True)}


def detect_deaths(infos, max_area_px=None, max_area_fraction_of_track_max=0.5, exclude_border_cells=True,
                  exclude_border_cells_plus_one=True):
    """
    Detects dying (delaminating/apoptotic) cells: tracks that vanish before the end of the movie and never reappear.

    Args:
        infos (list): per frame info (see _frame_info).
        max_area_px (int, optional): the cell must be smaller than that at its last frame (TA default was 120). None to skip.
        max_area_fraction_of_track_max (float, optional): the cell area at its last frame must be below this fraction of
            the max area of its track (dying cells shrink). None to skip.
        exclude_border_cells (bool): ignore cells touching the border/background at their last frame.
        exclude_border_cells_plus_one (bool): also ignore their neighbours.

    Returns:
        list: list of dicts with keys track_id, first_frame, last_frame, last_area, max_area.
    """
    first_frame = {}
    last_frame = {}
    max_area = {}
    for t, info in enumerate(infos):
        for track, area in info['areas'].items():
            first_frame.setdefault(track, t)
            last_frame[track] = t
            max_area[track] = max(max_area.get(track, 0), area)

    deaths = []
    last_movie_frame = len(infos) - 1
    for track, t in last_frame.items():
        if t >= last_movie_frame:
            continue
        info = infos[t]
        if exclude_border_cells and track in info['border']:
            continue
        if exclude_border_cells_plus_one and track in info['border_plus_one']:
            continue
        area = info['areas'][track]
        if max_area_px is not None and area > max_area_px:
            continue
        if max_area_fraction_of_track_max is not None and area > max_area_fraction_of_track_max * max_area[track]:
            continue
        deaths.append({'track_id': track, 'first_frame': first_frame[track], 'last_frame': t, 'last_area': area,
                       'max_area': max_area[track]})
    return deaths


def _paint(tracks, cells, color):
    out = np.zeros_like(tracks)
    if cells:
        out[np.isin(tracks, list(cells))] = color
    out[tracks == MEMBRANE] = MEMBRANE
    return int24_to_RGB(out)


def _local_ids(filename, tracks):
    """
    Returns a dict track_id -> local_id (local_id as in cells_2D, i.e. label of handCorrection.tif).
    """
    handCorrection1, handCorrection2 = smart_name_parser(filename, ordered_output=['handCorrection.png', 'handCorrection.tif'])
    mask_path = handCorrection2 if os.path.isfile(handCorrection2) else handCorrection1
    if not os.path.isfile(mask_path):
        return {}
    mask = Img(mask_path)
    if mask.ndim >= 3:
        mask = mask[..., 0]
    cell_labels = label(mask, connectivity=1, background=255)
    correspondence = {}
    for region in regionprops(cell_labels):
        y, x = region.coords[0]
        correspondence[int(tracks[y, x])] = region.label
    return correspondence


def _save_frame_outputs(filename, tracks, divisions_now, dying_now, will_die):
    TA_path = smart_name_parser(filename, ordered_output='TA')
    daughters = [d['daughter1_track_id'] for d in divisions_now] + [d['daughter2_track_id'] for d in divisions_now]
    Img(_paint(tracks, daughters, DIVISION_COLOR), dimensions='hwc').save(os.path.join(TA_path, 'dividing_cells.tif'), mode='raw')
    Img(_paint(tracks, will_die, DEATH_COLOR), dimensions='hwc').save(os.path.join(TA_path, 'dying_cells.tif'), mode='raw')

    sister = {}
    mother = {}
    division_id = {}
    for d in divisions_now:
        d1 = d['daughter1_track_id']
        d2 = d['daughter2_track_id']
        sister[d1] = d2
        sister[d2] = d1
        mother[d1] = mother[d2] = d['mother_track_id']
        division_id[d1] = division_id[d2] = d['division_id']

    track_to_local = _local_ids(filename, tracks)
    columns = {'local_id': [], 'track_id': [], 'is_newborn_daughter': [], 'sister_track_id': [], 'mother_track_id': [],
               'division_id': [], 'is_dying': [], 'will_die': []}
    for track in sorted(get_areas(tracks).keys()):
        if track not in track_to_local:
            continue
        columns['local_id'].append(int(track_to_local[track]))
        columns['track_id'].append(int(track))
        columns['is_newborn_daughter'].append(int(track in sister))
        columns['sister_track_id'].append(int(sister.get(track, 0)))
        columns['mother_track_id'].append(int(mother.get(track, 0)))
        columns['division_id'].append(int(division_id.get(track, 0)))
        columns['is_dying'].append(int(track in dying_now))
        columns['will_die'].append(int(track in will_die))

    db = TAsql(filename_or_connection=os.path.join(TA_path, 'pyTA.db'))
    try:
        db.drop_table('cell_fate')
        if columns['local_id']:
            db.create_and_append_table('cell_fate', columns)
    finally:
        db.close()


def _write_csv(path, rows, header):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(row[h]) for h in header) + '\n')


def detect_divisions_and_deaths(lst, detect_divs=True, detect_death=True, min_daughter_area_ratio=0.25,
                                mother_area_conservation=(0.5, 2.0), require_daughter_persistence=True,
                                exclude_border_cells=True, death_max_area_px=None,
                                death_max_area_fraction_of_track_max=0.5, death_exclude_border_cells_plus_one=True,
                                progress_callback=None):
    """
    Detects cell divisions and cell deaths in a tracked movie and saves the results (see module doc).

    Args:
        lst (list): ordered list of the movie frames (images already segmented and tracked).
        detect_divs (bool): detect divisions.
        detect_death (bool): detect deaths.
        min_daughter_area_ratio (float): min ratio between the two daughter areas.
        mother_area_conservation (tuple): allowed range for (daughter1 + daughter2 area) / mother area, None to skip.
        require_daughter_persistence (bool): the new daughter track must still exist in the next frame (removes
            transient over-segmentations).
        exclude_border_cells (bool): ignore border cells (divisions and deaths).
        death_max_area_px (int, optional): absolute max area of a dying cell at its last frame.
        death_max_area_fraction_of_track_max (float, optional): max area of a dying cell at its last frame relative
            to the max area of its track.
        death_exclude_border_cells_plus_one (bool): also ignore the neighbours of border cells for deaths.
        progress_callback: Qt signal emitting an int percentage (optional).

    Returns:
        tuple: (divisions, deaths) lists of dicts.
    """
    if lst is None or not lst:
        return None, None

    def progress(value):
        try:
            if progress_callback is not None:
                progress_callback.emit(int(value))
            else:
                logger.info(str(value) + '%')
        except:
            pass

    # pass 1: analyse every frame (only the frame info is kept, images are reloaded when needed)
    infos = []
    for t, file in enumerate(lst):
        if early_stop.stop:
            return None, None
        progress(t / len(lst) * 40)
        tracks = load_tracks(file)
        if tracks is None:
            return None, None
        infos.append(_frame_info(tracks))

    # pass 2: divisions
    divisions = []
    if detect_divs:
        already_seen = set()
        tracks_prev = load_tracks(lst[0])
        already_seen |= set(infos[0]['areas'].keys())
        for t in range(1, len(lst)):
            if early_stop.stop:
                return None, None
            progress(40 + t / len(lst) * 30)
            tracks_cur = load_tracks(lst[t])
            next_ids = None
            if require_daughter_persistence and t + 1 < len(lst):
                next_ids = set(infos[t + 1]['areas'].keys())
            divs = detect_divisions_between(tracks_prev, tracks_cur, infos[t - 1], infos[t], already_seen,
                                            tracks_next_ids=next_ids, min_daughter_area_ratio=min_daughter_area_ratio,
                                            mother_area_conservation=mother_area_conservation,
                                            exclude_border_cells=exclude_border_cells)
            for d in divs:
                d['frame_nb'] = t
                d['filename'] = lst[t]
            divisions.extend(divs)
            already_seen |= set(infos[t]['areas'].keys())
            tracks_prev = tracks_cur
        for i, d in enumerate(divisions):
            d['division_id'] = i + 1

    deaths = []
    if detect_death:
        deaths = detect_deaths(infos, max_area_px=death_max_area_px,
                               max_area_fraction_of_track_max=death_max_area_fraction_of_track_max,
                               exclude_border_cells=exclude_border_cells,
                               exclude_border_cells_plus_one=death_exclude_border_cells_plus_one)
        for d in deaths:
            d['filename'] = lst[d['last_frame']]
    del infos

    # pass 3: save per frame outputs
    dying_tracks = set(d['track_id'] for d in deaths)
    for t, file in enumerate(lst):
        if early_stop.stop:
            return divisions, deaths
        progress(70 + t / len(lst) * 30)
        try:
            tracks = load_tracks(file)
            divisions_now = [d for d in divisions if d['frame_nb'] == t]
            dying_now = set(d['track_id'] for d in deaths if d['last_frame'] == t)
            _save_frame_outputs(file, tracks, divisions_now, dying_now, dying_tracks)
        except:
            traceback.print_exc()
            logger.error('Could not save divisions/deaths for ' + str(file))

    output_folder = os.path.dirname(lst[0])
    if detect_divs:
        _write_csv(os.path.join(output_folder, 'cell_divisions.csv'), divisions,
                   ['division_id', 'frame_nb', 'filename', 'mother_track_id', 'daughter1_track_id', 'daughter2_track_id',
                    'mother_area', 'daughter1_area', 'daughter2_area', 'score'])
    if detect_death:
        _write_csv(os.path.join(output_folder, 'cell_deaths.csv'), deaths,
                   ['track_id', 'first_frame', 'last_frame', 'filename', 'last_area', 'max_area'])
    logger.info(str(len(divisions)) + ' divisions and ' + str(len(deaths)) + ' deaths detected')
    progress(100)
    return divisions, deaths


if __name__ == '__main__':
    from batoolset.utils.loadlist import loadlist

    lst = loadlist('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/list.lst')
    divisions, deaths = detect_divisions_and_deaths(lst)
    print(divisions)
    print(deaths)
