"""
Detection of cell divisions and cell deaths (delamination/apoptosis) from tracked cells.

Works on the 'tracked_cells_resized.tif' images produced by the pyTA trackers (one RGB colour = one track ID,
membranes are white 0xFFFFFF, 0 is background). Extension of the TissueAnalyzer (Java) TrackCellsV1.find_cell_divisions
and TrackDyingCells.

A new track ID or a vanishing track ID is not necessarily a division or a death: it can be a tracking error (the cell
got a new ID), an over-segmentation (a cell transiently cut in two), an under-segmentation (a lost membrane: the cell
is merged with a neighbour for a few frames) or a cell entering/leaving the field. Every track start and track end
of the movie is therefore classified (see EVENT_CLASSES):

    1. tissue motion: every cell moves like the (median of the) nearest cells that keep their ID between the two
       frames, the pixels of new/vanishing cells are moved accordingly before being compared to the other frame.
    2. identity switches: a track that ends and a track that starts at the same place with the same size (directly or
       after a gap of up to max_gap frames, e.g. a cell transiently merged with a neighbour) are the same cell.
    3. splits: a new cell that mostly comes from one mother cell that also gave a large part of its pixels to a sister
       (the mother herself or another new cell) is a division if the two daughters are neighbours, persist, have
       plausible areas and the mother is not tiny; otherwise it is an over-segmentation and the fragment is considered
       part of its mother.
    4. ends: a cell (with its fragments) that vanishes is a death if it lived long enough, shrank a lot (relative to
       its own max area and to the median cell area) and is not at the border. Otherwise it is merged in a neighbour,
       leaving the field or lost.

Outputs, per frame (in the TA folder, visible in the pyTA preview):
    - dividing_cells.tif: daughter cells of divisions occurring at this frame painted blue (TA convention)
    - dying_cells.tif: cells whose track dies later in the movie (or now) painted blue (TA convention)
    - 'cell_fate' table in pyTA.db (one row per cell): local_id, track_id, is_newborn_daughter,
      sister_track_id, mother_track_id, division_id, is_dying (last frame of a dying track), will_die
And for the whole movie, next to the images: cell_divisions.csv, cell_deaths.csv and track_events.csv (the class
of every track start/end, useful to spot tracking and segmentation errors)
"""
import os
import traceback
import numpy as np
from scipy.spatial import cKDTree
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

# classes of track starts and ends (track_events.csv)
DIVISION = 'division'  # start: newborn daughter
IDENTITY_SWITCH = 'identity_switch'  # start/end: the cell exists before/after with another track ID (tracking error)
OVERSEGMENTATION = 'oversegmentation'  # start/end: a transient piece of a cell (segmentation error)
ENTERS_FIELD = 'enters_field'  # start: border cell
UNEXPLAINED_START = 'unexplained_start'  # start: a new cell coming from nowhere
DEATH = 'death'  # end: dying cell (delamination/apoptosis)
LEAVES_FIELD = 'leaves_field'  # end: border cell
MERGED = 'merged'  # end: the cell merged with a neighbour (lost membrane or tracking error)
LOST = 'lost'  # end: the cell vanished without shrinking and without being merged
SEGMENTATION_ARTEFACT = 'segmentation_artefact'  # start/end: an object that never reached the size of a cell
EVENT_CLASSES = (DIVISION, IDENTITY_SWITCH, OVERSEGMENTATION, ENTERS_FIELD, UNEXPLAINED_START, DEATH, LEAVES_FIELD,
                 MERGED, LOST, SEGMENTATION_ARTEFACT)


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


class _CellPixels:
    """Fast access to the pixels of every track of a frame (one sort per frame instead of one image scan per cell)."""

    def __init__(self, tracks):
        self.shape = tracks.shape
        flat = tracks.ravel()
        self.order = np.argsort(flat, kind='stable')
        ids, starts, counts = np.unique(flat[self.order], return_index=True, return_counts=True)
        self.index = {i: (s, c) for i, s, c in zip(ids.tolist(), starts.tolist(), counts.tolist())}

    def coords(self, track):
        """(ys, xs) of the pixels of a track."""
        s, c = self.index.get(track, (0, 0))
        return np.unravel_index(self.order[s:s + c], self.shape)

    def centroids(self):
        """dict track -> (y, x) (membrane and background excluded)."""
        out = {}
        for track in self.index:
            if track in (MEMBRANE, BACKGROUND):
                continue
            ys, xs = self.coords(track)
            out[track] = (float(ys.mean()), float(xs.mean()))
        return out


def _frame_info(tracks):
    neighbors, touching_background = get_neighbors(tracks)
    pixels = _CellPixels(tracks)
    areas = get_areas(tracks)
    return {'areas': areas, 'neighbors': neighbors, 'centroids': pixels.centroids(),
            'median_area': float(np.median(list(areas.values()))) if areas else 0.,
            'border': get_border_cells(tracks, neighbors, touching_background, plus_one=False),
            'border_plus_one': get_border_cells(tracks, neighbors, touching_background, plus_one=True)}


def _drop_remnants(infos, remnant_fraction_of_median=0.05, min_cell_area_fraction_of_median=0.15):
    """
    Forgets the tiny remnants (a few pixels) of real cells: a track ID that survives on a couple of pixels for a frame
    after the cell got another ID or vanished would otherwise look like a cell that shrank to nothing.
    Tracks that were always tiny (segmentation artefacts) are kept.
    """
    if remnant_fraction_of_median is None:
        return
    max_area = {}
    for info in infos:
        for track, area in info['areas'].items():
            max_area[track] = max(max_area.get(track, 0), area)
    for info in infos:
        tiny = remnant_fraction_of_median * info['median_area']
        real = (min_cell_area_fraction_of_median or 0) * info['median_area']
        for track in [c for c, a in info['areas'].items() if a < tiny and max_area[c] >= real]:
            del info['areas'][track]
            info['centroids'].pop(track, None)


class _LocalMotion:
    """
    Tissue motion between two frames estimated from the cells keeping their track ID. The displacements of the k
    nearest such cells (and their median) are candidates, the one that best superimposes the track IDs of the two frames
    around the point wins (robust to the cells whose ID was wrongly propagated by the tracker, that move 'a cell away').
    Without images (tracks_prev/tracks_cur None) the median displacement is used.
    """

    def __init__(self, info_prev, info_cur, tracks_prev=None, tracks_cur=None, k=10):
        persisting = [c for c in info_prev['centroids'] if c in info_cur['centroids']]
        self.k = k
        self.tracks_prev = tracks_prev
        self.tracks_cur = tracks_cur
        median_area = info_cur['median_area'] or info_prev['median_area'] or 1.
        self.radius = int(round(2 * np.sqrt(median_area)))
        if persisting:
            prev = np.array([info_prev['centroids'][c] for c in persisting])
            cur = np.array([info_cur['centroids'][c] for c in persisting])
            self.displacements = cur - prev
            self.tree_prev = cKDTree(prev)
            self.tree_cur = cKDTree(cur)
        else:
            self.displacements = None

    def _agreement(self, y, x, dy, dx, in_cur_frame):
        """fraction of the cell pixels around (y, x) that carry the same track ID in the other frame once moved."""
        here, there = (self.tracks_cur, self.tracks_prev) if in_cur_frame else (self.tracks_prev, self.tracks_cur)
        sign = -1 if in_cur_frame else 1
        h, w = here.shape
        y0, y1 = max(int(y) - self.radius, 0), min(int(y) + self.radius + 1, h)
        x0, x1 = max(int(x) - self.radius, 0), min(int(x) + self.radius + 1, w)
        patch = here[y0:y1, x0:x1]
        ys, xs = np.mgrid[y0:y1, x0:x1]
        ys = ys + int(round(sign * dy))
        xs = xs + int(round(sign * dx))
        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w) & (patch != MEMBRANE) & (patch != BACKGROUND)
        if not np.any(valid):
            return 0.
        return float(np.mean(patch[valid] == there[ys[valid], xs[valid]]))

    def displacement(self, y, x, in_cur_frame):
        """displacement (dy, dx) from the previous to the current frame of a point given in the previous (or current) frame."""
        if self.displacements is None:
            return 0., 0.
        tree = self.tree_cur if in_cur_frame else self.tree_prev
        k = min(self.k, len(self.displacements))
        _, idx = tree.query([y, x], k=k)
        candidates = self.displacements[np.atleast_1d(idx)]
        median = np.median(candidates, axis=0)
        if self.tracks_prev is None or self.tracks_cur is None:
            return float(median[0]), float(median[1])
        best = None
        for d in [median] + list(candidates):
            score = self._agreement(y, x, d[0], d[1], in_cur_frame)
            if best is None or score > best[0]:
                best = (score, d)
        return float(best[1][0]), float(best[1][1])


def _moved_overlap(pixels, track, dy, dx, other_tracks, valid_tracks):
    """
    dict track -> nb of pixels of 'track' landing on it in other_tracks once moved by (dy, dx) (membrane/outside and
    tracks not in valid_tracks ignored).
    """
    ys, xs = pixels.coords(track)
    ys = np.rint(ys + dy).astype(np.int64)
    xs = np.rint(xs + dx).astype(np.int64)
    h, w = other_tracks.shape
    inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    below = other_tracks[ys[inside], xs[inside]]
    ids, counts = np.unique(below, return_counts=True)
    return {i: c for i, c in zip(ids.tolist(), counts.tolist()) if i in valid_tracks}


def _best(overlap, area):
    """(track, fraction of area) of the biggest overlap, (None, 0) if none."""
    if not overlap or not area:
        return None, 0.
    track = max(overlap, key=overlap.get)
    return track, overlap[track] / area


class _Families:
    """Union find of the track IDs that belong to the same cell (identity switches and over-segmentation fragments)."""

    def __init__(self):
        self.parent = {}

    def find(self, a):
        root = a
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while self.parent.get(a, a) != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _measure_events(lst, infos, first, last, progress=None, min_source_fraction=0.2):
    """
    Pass over the frame pairs: for every new track at t, what was below it at t-1 (motion compensated); for every
    track ending at t-1, where its pixels go at t; and the same for the main source of every new track (possible mother).

    Returns:
        tuple: (starts, ends, motions) with starts[t][track] = overlap dict at t-1, ends[t][track] = overlap dict at t+1,
        mother_destinations[t][track] = overlap dict at t of a track of t-1 and motions[t] = _LocalMotion(t-1 -> t)
    """
    n = len(lst)
    starts = [dict() for _ in range(n)]
    ends = [dict() for _ in range(n)]
    mother_destinations = [dict() for _ in range(n)]
    motions = [None] * n
    tracks_prev = load_tracks(lst[0])
    pixels_prev = _CellPixels(tracks_prev)
    for t in range(1, n):
        if early_stop.stop:
            return None
        if progress is not None:
            progress(t)
        tracks_cur = load_tracks(lst[t])
        pixels_cur = _CellPixels(tracks_cur)
        motion = _LocalMotion(infos[t - 1], infos[t], tracks_prev, tracks_cur)
        motions[t] = motion
        for track, (y, x) in infos[t]['centroids'].items():
            if first[track] != t:
                continue
            dy, dx = motion.displacement(y, x, in_cur_frame=True)
            starts[t][track] = _moved_overlap(pixels_cur, track, -dy, -dx, tracks_prev, infos[t - 1]['areas'])
            # every cell below a good part of the new cell is a possible mother
            for mother, count in starts[t][track].items():
                if count < min_source_fraction * infos[t]['areas'][track] or mother in mother_destinations[t]:
                    continue
                my, mx = infos[t - 1]['centroids'][mother]
                mdy, mdx = motion.displacement(my, mx, in_cur_frame=False)
                mother_destinations[t][mother] = _moved_overlap(pixels_prev, mother, mdy, mdx, tracks_cur, infos[t]['areas'])
        for track, (y, x) in infos[t - 1]['centroids'].items():
            if last[track] != t - 1:
                continue
            dy, dx = motion.displacement(y, x, in_cur_frame=False)
            ends[t - 1][track] = _moved_overlap(pixels_prev, track, dy, dx, tracks_cur, infos[t]['areas'])
        # keep the motion for the gap closing but not the images (memory)
        motion.tracks_prev = motion.tracks_cur = None
        tracks_prev = tracks_cur
        pixels_prev = pixels_cur
    return starts, ends, mother_destinations, motions


def _predict_position(motions, y, x, t_from, t_to):
    """position at t_to of a point at (y, x) at t_from following the local tissue motion."""
    for t in range(t_from + 1, t_to + 1):
        dy, dx = motions[t].displacement(y, x, in_cur_frame=False)
        y += dy
        x += dx
    return y, x


def classify_track_events(infos, starts, ends, mother_destinations, motions, first, last,
                          min_daughter_area_ratio=0.25, mother_area_conservation=(0.5, 2.0),
                          require_daughter_persistence=True, min_daughter_lifetime=3,
                          min_mother_area_fraction_of_median=0.3, exclude_border_cells=True,
                          death_max_area_px=None, death_max_area_fraction_of_track_max=0.5,
                          death_max_area_fraction_of_median=0.4, death_min_small_frames=2, death_min_track_length=3,
                          death_exclude_border_cells_plus_one=True, max_gap=2, min_cell_area_fraction_of_median=0.15,
                          min_overlap=0.5, min_split_share=0.15, min_split_iou=0.5):
    """
    Classifies every track start and end (see module doc). Pure function of the per frame measurements.

    Returns:
        tuple: (divisions, deaths, events) lists of dicts; events has one row per track start/end with its class.
    """
    n = len(infos)
    families = _Families()
    start_class = {}
    end_class = {}
    related = {}  # (track, 'start'/'end') -> related track (mother, sister, previous/next id, absorbing cell)
    details = {}  # (track, 'start'/'end') -> why the event got its class (text)

    # 1. direct identity switches: the new cell mostly lies on a vanished cell whose pixels mostly go to the new cell
    for t in range(1, n):
        for track, overlap in starts[t].items():
            source, fraction = _best(overlap, infos[t]['areas'][track])
            if source is None or fraction < min_overlap or last[source] != t - 1 or (source, 'end') in end_class:
                continue
            destination, dest_fraction = _best(ends[t - 1].get(source, {}), infos[t - 1]['areas'][source])
            if destination == track and dest_fraction >= min_overlap:
                start_class[(track, 'start')] = IDENTITY_SWITCH
                end_class[(source, 'end')] = IDENTITY_SWITCH
                related[(track, 'start')] = source
                related[(source, 'end')] = track
                families.union(source, track)

    # 2. identity recovered after a gap (e.g. cell merged with a neighbour or not segmented for a few frames):
    # same predicted position and similar area
    if max_gap > 0:
        candidates = []
        for t_end in range(n):
            for track in ends[t_end]:
                if (track, 'end') in end_class:
                    continue
                y, x = infos[t_end]['centroids'][track]
                area = infos[t_end]['areas'][track]
                for t_start in range(t_end + 2, min(t_end + 2 + max_gap, n)):
                    py, px = _predict_position(motions, y, x, t_end, t_start)
                    radius = 0.5 * np.sqrt(infos[t_start]['median_area'])
                    for new, overlap in starts[t_start].items():
                        if (new, 'start') in start_class:
                            continue
                        ny, nx = infos[t_start]['centroids'][new]
                        distance = np.hypot(ny - py, nx - px)
                        ratio = infos[t_start]['areas'][new] / area
                        if distance <= radius and 0.5 <= ratio <= 2.:
                            candidates.append((t_start - t_end, distance, track, new))
        for _, _, track, new in sorted(candidates):
            if (track, 'end') in end_class or (new, 'start') in start_class:
                continue
            start_class[(new, 'start')] = IDENTITY_SWITCH
            end_class[(track, 'end')] = IDENTITY_SWITCH
            related[(new, 'start')] = track
            related[(track, 'end')] = new
            families.union(track, new)

    def family_alive(track, t):
        """True if the cell (any track of its family) exists at frame t."""
        root = families.find(track)
        return any(families.find(c) == root for c in infos[t]['areas'])

    # 3. splits: divisions or over-segmentation
    divisions = []
    for t in range(1, n):
        info_prev, info_cur = infos[t - 1], infos[t]
        for track in sorted(starts[t]):
            if (track, 'start') in start_class:
                continue
            area = info_cur['areas'][track]
            # the mother (moved to t) must match the union of the two daughters, whichever daughter kept her ID: the
            # sister is the mother herself if her ID continues, otherwise another new cell; both touch the new cell
            best = None
            for mother, count in starts[t][track].items():
                destinations = mother_destinations[t].get(mother)
                if destinations is None:
                    continue
                mother_area = info_prev['areas'][mother]
                for sister in info_cur['neighbors'].get(track, set()):
                    if sister not in info_cur['areas']:
                        continue
                    if not (sister == mother or (first[sister] == t and (sister, 'start') not in start_class)):
                        continue
                    share_track = destinations.get(track, 0) / mother_area
                    share_sister = destinations.get(sister, 0) / mother_area
                    if min(share_track, share_sister) < min_split_share:
                        continue
                    inter = destinations.get(track, 0) + destinations.get(sister, 0)
                    iou = inter / (mother_area + area + info_cur['areas'][sister] - inter)
                    if best is None or iou > best[0]:
                        best = (iou, mother, sister)
            if best is None or best[0] < min_split_iou:
                source, fraction_from_source = _best(starts[t][track], area)
                to_new = 0.
                if source is not None and source in mother_destinations[t]:
                    to_new = mother_destinations[t][source].get(track, 0) / info_prev['areas'][source]
                details[(track, 'start')] = 'no split (best mother/daughters IoU %.2f; %.2f of the cell from %s which gives it %.2f)' % (
                    best[0] if best else 0., fraction_from_source, source, to_new)
                # ID slide (tracking error): a cell of t-1 becomes the new cell 1:1 while its ID jumped to a neighbour
                # (whose own ID vanishes); unlike a division, the old cell is not shared between two cells
                source, fraction_from_source = _best(starts[t][track], area)
                if source is not None and fraction_from_source >= min_overlap and source in mother_destinations[t]:
                    source_area = info_prev['areas'][source]
                    to_new = mother_destinations[t][source].get(track, 0) / source_area
                    iou = to_new / (1. + area / source_area - to_new)
                    if to_new >= min_overlap and iou >= min_overlap:
                        start_class[(track, 'start')] = IDENTITY_SWITCH
                        related[(track, 'start')] = source
                        details[(track, 'start')] = 'ID slide: the cell of track %d got this new ID' % source
                        # the neighbour whose ID vanished and that now carries the ID of the source
                        for other, overlap in ends[t - 1].items():
                            if (other, 'end') in end_class:
                                continue
                            destination, dest_fraction = _best(overlap, info_prev['areas'][other])
                            if destination == source and dest_fraction >= min_overlap:
                                end_class[(other, 'end')] = IDENTITY_SWITCH
                                related[(other, 'end')] = source
                                details[(other, 'end')] = 'ID slide: this cell now carries the ID %d' % source
                continue
            fraction, mother, sister = best
            mother_area = info_prev['areas'][mother]
            area_sister = info_cur['areas'][sister]

            reasons = []
            if exclude_border_cells and (mother in info_prev['border'] or track in info_cur['border'] or sister in info_cur['border']):
                reasons.append('border')
            if min(area, area_sister) / max(area, area_sister) < min_daughter_area_ratio:
                reasons.append('daughter_area_ratio')
            if mother_area_conservation is not None:
                conservation = (area + area_sister) / mother_area
                if not mother_area_conservation[0] <= conservation <= mother_area_conservation[1]:
                    reasons.append('area_conservation')
            if min_mother_area_fraction_of_median is not None and mother_area < min_mother_area_fraction_of_median * info_prev['median_area']:
                reasons.append('mother_too_small')
            if sister not in info_cur['neighbors'].get(track, set()):
                reasons.append('daughters_not_neighbors')
            if require_daughter_persistence:
                for t2 in range(t + 1, min(t + min_daughter_lifetime, n)):
                    if not (family_alive(track, t2) and family_alive(sister, t2)):
                        reasons.append('daughters_not_persistent')
                        break
            if 'border' in reasons:
                start_class[(track, 'start')] = ENTERS_FIELD
                related[(track, 'start')] = mother
                continue
            if reasons:
                start_class[(track, 'start')] = OVERSEGMENTATION
                details[(track, 'start')] = ' '.join(reasons) + ' (IoU %.2f)' % fraction
                related[(track, 'start')] = mother
                families.union(mother, track)
                continue
            start_class[(track, 'start')] = DIVISION
            details[(track, 'start')] = 'IoU %.2f' % fraction
            related[(track, 'start')] = mother
            if sister != mother:
                start_class[(sister, 'start')] = DIVISION
                related[(sister, 'start')] = mother
                end_class[(mother, 'end')] = DIVISION
                related[(mother, 'end')] = track
            divisions.append({'frame_nb': t, 'mother_track_id': mother, 'daughter1_track_id': sister,
                              'daughter2_track_id': track, 'mother_area': mother_area, 'daughter1_area': area_sister,
                              'daughter2_area': area, 'score': round(fraction, 3)})

    # 4. ends, judged per cell (family): the cell ends when its last track ends
    family_last = {}
    family_first = {}
    for track in last:
        root = families.find(track)
        family_last[root] = max(family_last.get(root, -1), last[track])
        family_first[root] = min(family_first.get(root, n), first[track])
    family_area = {}
    for t, info in enumerate(infos):
        for track, area in info['areas'].items():
            key = (families.find(track), t)
            family_area[key] = family_area.get(key, 0) + area
    family_max_area = {}
    for (root, t), area in family_area.items():
        family_max_area[root] = max(family_max_area.get(root, 0), area)

    def is_artefact(track, t):
        """an object that never reached the size of a cell (e.g. a vertex segmented as a cell)"""
        return min_cell_area_fraction_of_median is not None and \
            family_max_area[families.find(track)] < min_cell_area_fraction_of_median * infos[t]['median_area']

    # remaining starts
    for t in range(1, n):
        for track in starts[t]:
            if (track, 'start') in start_class:
                continue
            if track in infos[t]['border']:
                start_class[(track, 'start')] = ENTERS_FIELD
            elif is_artefact(track, t):
                start_class[(track, 'start')] = SEGMENTATION_ARTEFACT
            else:
                start_class[(track, 'start')] = UNEXPLAINED_START
            related[(track, 'start')] = _best(starts[t][track], infos[t]['areas'][track])[0]

    deaths = []
    dead_cells = set()
    for t in range(n):
        for track, overlap in ends[t].items():
            if (track, 'end') in end_class:
                continue
            info = infos[t]
            root = families.find(track)
            destination, fraction = _best(overlap, info['areas'][track])
            related[(track, 'end')] = destination
            if family_last[root] > t:
                # another track of the same cell continues (over-segmentation fragment)
                end_class[(track, 'end')] = OVERSEGMENTATION
                continue
            if exclude_border_cells and track in info['border']:
                end_class[(track, 'end')] = LEAVES_FIELD
                continue
            if is_artefact(track, t):
                end_class[(track, 'end')] = SEGMENTATION_ARTEFACT
                continue
            area = family_area[(root, t)]
            is_death = not (death_exclude_border_cells_plus_one and track in info['border_plus_one'])
            if death_min_track_length is not None and t - family_first[root] + 1 < death_min_track_length:
                is_death = False
            if death_max_area_px is not None and area > death_max_area_px:
                is_death = False
            if death_max_area_fraction_of_track_max is not None and area > death_max_area_fraction_of_track_max * family_max_area[root]:
                is_death = False
            if death_max_area_fraction_of_median is not None:
                # a dying cell shrinks progressively: it must already be small in the frames before it vanishes (a
                # normal size cell disappearing at once is a tracking/segmentation error)
                for t2 in range(t, max(t - death_min_small_frames, family_first[root] - 1), -1):
                    if family_area.get((root, t2), 0) > death_max_area_fraction_of_median * infos[t2]['median_area']:
                        is_death = False
                        break
            if is_death:
                end_class[(track, 'end')] = DEATH
                if root in dead_cells:
                    # another fragment of the same dying cell vanishing at the same frame
                    continue
                dead_cells.add(root)
                members = sorted(c for c in last if families.find(c) == root)
                deaths.append({'track_id': track, 'first_frame': family_first[root], 'last_frame': t, 'last_area': area,
                               'max_area': family_max_area[root], 'family': members})
            elif destination is not None and fraction >= min_overlap:
                end_class[(track, 'end')] = MERGED
            else:
                end_class[(track, 'end')] = LOST

    events = []
    for (track, kind), cls in list(start_class.items()) + list(end_class.items()):
        t = first[track] if kind == 'start' else last[track]
        rel = related.get((track, kind))
        events.append({'frame_nb': t, 'track_id': track, 'event': kind, 'class': cls,
                       'related_track_id': rel if rel is not None else 0, 'area': infos[t]['areas'][track],
                       'cell_id': families.find(track), 'details': details.get((track, kind), '')})
    events.sort(key=lambda e: (e['frame_nb'], e['event'], e['track_id']))
    for i, d in enumerate(divisions):
        d['division_id'] = i + 1
    return divisions, deaths, events


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
                                min_daughter_lifetime=3, min_mother_area_fraction_of_median=0.3,
                                exclude_border_cells=True, death_max_area_px=None,
                                death_max_area_fraction_of_track_max=0.5, death_max_area_fraction_of_median=0.4,
                                death_min_small_frames=2, death_min_track_length=3,
                                death_exclude_border_cells_plus_one=True, max_gap=2,
                                min_cell_area_fraction_of_median=0.15, remnant_fraction_of_median=0.05,
                                progress_callback=None):
    """
    Detects cell divisions and cell deaths in a tracked movie, telling them apart from tracking and segmentation
    errors, and saves the results (see module doc).

    Args:
        lst (list): ordered list of the movie frames (images already segmented and tracked).
        detect_divs (bool): detect divisions.
        detect_death (bool): detect deaths.
        min_daughter_area_ratio (float): min ratio between the two daughter areas.
        mother_area_conservation (tuple): allowed range for (daughter1 + daughter2 area) / mother area, None to skip.
        require_daughter_persistence (bool): both daughters must exist for min_daughter_lifetime frames (removes
            transient over-segmentations).
        min_daughter_lifetime (int): number of frames (division frame included) the daughters must exist.
        min_mother_area_fraction_of_median (float, optional): the mother must be bigger than this fraction of the median
            cell area (a tiny cell cut in pieces is a fragmenting dying cell or a segmentation error). None to skip.
        exclude_border_cells (bool): ignore border cells (divisions and deaths).
        death_max_area_px (int, optional): absolute max area of a dying cell at its last frame.
        death_max_area_fraction_of_track_max (float, optional): max area of a dying cell at its last frame relative
            to the max area of its track.
        death_max_area_fraction_of_median (float, optional): max area of a dying cell during its last
            death_min_small_frames frames relative to the median cell area of the frame (dying cells shrink
            progressively, a normal size cell vanishing at once is a segmentation/tracking error). None to skip.
        death_min_small_frames (int): number of frames before vanishing during which a dying cell must be small.
        death_min_track_length (int, optional): a dying cell must have been tracked for at least that many frames.
        death_exclude_border_cells_plus_one (bool): also ignore the neighbours of border cells for deaths.
        max_gap (int): a cell that vanishes and reappears (with another track ID) within max_gap frames at the same
            place is the same cell (not a death followed by a birth). 0 to disable.
        min_cell_area_fraction_of_median (float, optional): objects whose area never reaches this fraction of the median
            cell area are segmentation artefacts (e.g. vertices segmented as cells), neither divisions nor deaths.
        remnant_fraction_of_median (float, optional): a track of a real cell present on less than this fraction of the
            median cell area is ignored in that frame (e.g. a track ID surviving on a couple of pixels after the cell got
            another ID). None to keep them.
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
        progress(t / len(lst) * 30)
        tracks = load_tracks(file)
        if tracks is None:
            return None, None
        infos.append(_frame_info(tracks))
    _drop_remnants(infos, remnant_fraction_of_median, min_cell_area_fraction_of_median)
    first = {}
    last = {}
    for t, info in enumerate(infos):
        for track in info['areas']:
            first.setdefault(track, t)
            last[track] = t

    # pass 2: motion compensated overlaps of every track start/end
    measures = _measure_events(lst, infos, first, last, progress=lambda t: progress(30 + t / len(lst) * 40))
    if measures is None:
        return None, None
    starts, ends, mother_destinations, motions = measures
    divisions, deaths, events = classify_track_events(
        infos, starts, ends, mother_destinations, motions, first, last,
        min_daughter_area_ratio=min_daughter_area_ratio, mother_area_conservation=mother_area_conservation,
        require_daughter_persistence=require_daughter_persistence, min_daughter_lifetime=min_daughter_lifetime,
        min_mother_area_fraction_of_median=min_mother_area_fraction_of_median, exclude_border_cells=exclude_border_cells,
        death_max_area_px=death_max_area_px, death_max_area_fraction_of_track_max=death_max_area_fraction_of_track_max,
        death_max_area_fraction_of_median=death_max_area_fraction_of_median,
        death_min_small_frames=death_min_small_frames, death_min_track_length=death_min_track_length,
        death_exclude_border_cells_plus_one=death_exclude_border_cells_plus_one, max_gap=max_gap,
        min_cell_area_fraction_of_median=min_cell_area_fraction_of_median)
    del infos
    if not detect_divs:
        divisions = []
    if not detect_death:
        deaths = []
    for d in divisions:
        d['filename'] = lst[d['frame_nb']]
    for d in deaths:
        d['filename'] = lst[d['last_frame']]
    for e in events:
        e['filename'] = lst[e['frame_nb']]

    # pass 3: save per frame outputs
    dying_tracks = set()
    for d in deaths:
        dying_tracks.update(d['family'])
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
    _write_csv(os.path.join(output_folder, 'track_events.csv'), events,
               ['frame_nb', 'filename', 'track_id', 'event', 'class', 'related_track_id', 'area', 'cell_id', 'details'])
    counts = {c: sum(1 for e in events if e['class'] == c) for c in EVENT_CLASSES}
    logger.info(str(len(divisions)) + ' divisions and ' + str(len(deaths)) + ' deaths detected; track events: ' + str(counts))
    progress(100)
    return divisions, deaths


if __name__ == '__main__':
    from batoolset.utils.loadlist import loadlist

    lst = loadlist('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/list.lst')
    divisions, deaths = detect_divisions_and_deaths(lst)
    print(divisions)
    print(deaths)
