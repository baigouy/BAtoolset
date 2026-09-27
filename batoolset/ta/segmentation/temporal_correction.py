"""
Correction of transient segmentation errors using time.

A segmentation error rarely happens at the same place in several consecutive frames, whereas real cells persist.
Every frame t is therefore compared (motion compensated) with the frame before (t-1, already corrected) and the
frames after (t+d, the error may last up to max_error_duration frames) and one more frame (t+d+1) that confirms that
the state after is the real one (what ultimately happens decides):

    - over-segmentation: several neighbouring cells of t are each mostly one cell of t-1, one cell of t+d and one cell
      of t+d+1, and together cover these cells --> they are merged (the spurious bond is removed).
    - under-segmentation: a cell of t contains several cells of t-1, of t+d and of t+d+1 (the same number, matching
      one another) --> it is split in as many cells, seeded by these cells moved to t, along the brightest (membrane)
      path of the raw image (watershed).

Real events are not changed: a division (one cell, then two cells that persist), a death or an extrusion (a cell
that vanishes for good), a T1 (the number of cells does not change).

The image has the last word (bond_contrast): a bond is not removed if it lies on a visible membrane, and a cell is not
split where there is no membrane signal (when the frames around are over-segmented differently, time alone would
wrongly cut the cell in the frame that was right).

Validated against hand-corrected masks (30 frames of a pupal notum movie segmented automatically, max error duration
4, 3 rounds): over-segmentations -40 %, 98 % of the merges and ~85 % of the splits right. Errors that persist longer
than max_error_duration cannot be seen this way.

The motion of every cell is the median displacement of the nearest cells that keep their track ID between the two
frames, so the movie must be tracked first. The masks (handCorrection.tif/.png) of the corrected frames are
overwritten (the previous versions are kept, see undo_segmentation_corrections); the cells must then be tracked
(and measured) again (see batoolset.ta.database.provenance). Outputs:
    - segmentation_corrections.csv next to the images: one row per correction
    - segmentation_corrections.tif in every TA folder: merged cells in red, split cells in green
"""
import json
import os
import shutil
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.spatial import cKDTree
from skimage.measure import label
from skimage.segmentation import watershed
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img, int24_to_RGB
from batoolset.ta.tracking.divisions_and_deaths import load_tracks, get_neighbors, MEMBRANE, BACKGROUND
from batoolset.ta.database.provenance import input_path, stale_reasons
from batoolset.tools.early_stopper_class import early_stop
from batoolset.tools.logger import TA_logger

logger = TA_logger()

MERGE = 'merge'
SPLIT = 'split'
MERGE_COLOR = 0xFF0000
SPLIT_COLOR = 0x00FF00
CORRECTIONS_IMAGE = 'segmentation_corrections.tif'
CORRECTIONS_CSV = 'segmentation_corrections.csv'
BACKUP_NAME = 'handCorrection_before_temporal_correction'
UNDO_LIST = 'segmentation_corrections_undo.json'
CSV_HEADER = ['frame_nb', 'filename', 'correction', 'y', 'x', 'cells_before', 'cells_after', 'area',
              'reference_frames', 'score', 'contrast', 'track_ids', 'round']


class _Cells:
    """Cells of a frame (image of track IDs): areas, centroids and a cell index image."""

    def __init__(self, labels):
        self.labels = labels
        h, w = labels.shape
        flat = labels.ravel()
        pos = np.flatnonzero((flat != MEMBRANE) & (flat != BACKGROUND))
        ids, inverse, counts = np.unique(flat[pos], return_inverse=True, return_counts=True)
        self.ids = ids
        self.index = {cell: k for k, cell in enumerate(ids.tolist())}
        self.cell_index = np.full(h * w, -1, np.int64)
        self.cell_index[pos] = inverse
        self.pos = pos
        self.areas = counts
        ys, xs = np.divmod(pos, w)
        if len(ids):
            self.centroids = np.stack([np.bincount(inverse, ys) / counts, np.bincount(inverse, xs) / counts], axis=1)
        else:
            self.centroids = np.zeros((0, 2))
        self.median_area = float(np.median(counts)) if len(counts) else 0.
        self._neighbors = None

    def area(self, cell):
        return int(self.areas[self.index[cell]])

    def neighbors(self):
        if self._neighbors is None:
            self._neighbors = get_neighbors(self.labels)[0]
        return self._neighbors


def _displacements(cells_t, cells_r, k=10):
    """
    Displacement (dy, dx) from frame t to frame r of every cell of t: median displacement of the k nearest cells whose
    track ID exists in both frames (robust to a few wrong IDs).
    """
    common = np.intersect1d(cells_t.ids, cells_r.ids)
    if len(common) == 0 or len(cells_t.ids) == 0:
        return np.zeros((len(cells_t.ids), 2))
    in_t = np.array([cells_t.index[c] for c in common.tolist()])
    in_r = np.array([cells_r.index[c] for c in common.tolist()])
    moves = cells_r.centroids[in_r] - cells_t.centroids[in_t]
    k = min(k, len(common))
    _, idx = cKDTree(cells_t.centroids[in_t]).query(cells_t.centroids, k=k)
    idx = np.asarray(idx).reshape(len(cells_t.ids), k)
    return np.median(moves[idx], axis=1)


def _pull_back(cells_t, cells_r, displacements):
    """Image of frame t giving, for every cell pixel, the track ID below it in frame r once moved (-1 elsewhere)."""
    h, w = cells_t.labels.shape
    pos = cells_t.pos
    ys, xs = np.divmod(pos, w)
    moves = displacements[cells_t.cell_index[pos]]
    ry = np.rint(ys + moves[:, 0]).astype(np.int64)
    rx = np.rint(xs + moves[:, 1]).astype(np.int64)
    inside = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
    below = cells_r.labels[ry[inside], rx[inside]].astype(np.int64)
    below[(below == MEMBRANE) | (below == BACKGROUND)] = -1
    out = np.full(h * w, -1, np.int64)
    out[pos[inside]] = below
    return out.reshape(h, w)


class _Overlap:
    """Motion compensated overlap between the cells of frame t and those of a reference frame r."""

    def __init__(self, cells_t, cells_r, frame_r):
        self.cells_t = cells_t
        self.cells_r = cells_r
        self.frame_r = frame_r
        self.warped = _pull_back(cells_t, cells_r, _displacements(cells_t, cells_r))
        flat_w = self.warped.ravel()
        valid = flat_w >= 0
        keys = (cells_t.labels.ravel()[valid].astype(np.int64) << 24) | flat_w[valid]
        unique, counts = np.unique(keys, return_counts=True)
        self.table = {}
        for key, count in zip(unique.tolist(), counts.tolist()):
            self.table.setdefault(key >> 24, {})[key & 0xFFFFFF] = count

    def count(self, cell, ref_cell):
        return self.table.get(cell, {}).get(ref_cell, 0)

    def best(self, cell):
        """(cell of r covering most of cell, fraction of cell covered by it)"""
        overlap = self.table.get(cell)
        if not overlap:
            return None, 0.
        ref_cell = max(overlap, key=overlap.get)
        return ref_cell, overlap[ref_cell] / self.cells_t.area(cell)

    def parts(self, cell, min_inside, min_area):
        """cells of r lying mostly (min_inside of their area) inside cell"""
        return sorted(p for p, n in self.table.get(cell, {}).items()
                      if self.cells_r.area(p) >= min_area and n >= min_inside * self.cells_r.area(p))


def _components(cells, neighbors):
    """groups of cells connected by shared membranes"""
    todo = set(cells)
    groups = []
    while todo:
        stack = [todo.pop()]
        group = set(stack)
        while stack:
            cell = stack.pop()
            for other in neighbors.get(cell, ()):
                if other in todo:
                    todo.discard(other)
                    group.add(other)
                    stack.append(other)
        groups.append(sorted(group))
    return groups


def find_merges(cells_t, overlaps, min_inside=0.6, min_cover=0.6, max_area_ratio=1.5):
    """
    Over-segmentations at frame t: groups of neighbouring cells that are each mostly the same cell in every reference
    frame and together cover it.

    Returns:
        list: dicts with 'cells' (track IDs at t), 'references' (the cell in each reference frame) and 'score'.
    """
    groups = {}
    for cell in cells_t.ids.tolist():
        key = []
        for overlap in overlaps:
            ref_cell, fraction = overlap.best(cell)
            if ref_cell is None or fraction < min_inside:
                break
            key.append(ref_cell)
        else:
            groups.setdefault(tuple(key), []).append(cell)
    merges = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        for component in _components(group, cells_t.neighbors()):
            if len(component) < 2:
                continue
            area = sum(cells_t.area(c) for c in component)
            score = 1.
            for overlap, ref_cell in zip(overlaps, key):
                ref_area = overlap.cells_r.area(ref_cell)
                covered = sum(overlap.count(c, ref_cell) for c in component) / ref_area
                ratio = area / ref_area
                if covered < min_cover or not 1. / max_area_ratio <= ratio <= max_area_ratio:
                    score = 0.
                    break
                score = min(score, covered)
            if score > 0:
                merges.append({'cells': component, 'references': list(key), 'score': score})
    return merges


def find_splits(cells_t, overlaps, min_inside=0.6, min_cover=0.6, max_area_ratio=1.5, min_part_area=0):
    """
    Under-segmentations at frame t: cells that contain the same number (>= 2) of cells in every reference frame, these
    cells covering it and matching one another (first reference <-> second reference).

    Returns:
        list: dicts with 'cell' (track ID at t), 'parts' (pairs of cells of the first two references), 'score'.
    """
    splits = []
    for cell in cells_t.ids.tolist():
        area = cells_t.area(cell)
        all_parts = []
        score = 1.
        for overlap in overlaps:
            parts = overlap.parts(cell, min_inside, min_part_area)
            if len(parts) < 2 or (all_parts and len(parts) != len(all_parts[0])):
                break
            covered = sum(overlap.count(cell, p) for p in parts) / area
            ratio = area / sum(overlap.cells_r.area(p) for p in parts)
            if covered < min_cover or not 1. / max_area_ratio <= ratio <= max_area_ratio:
                break
            score = min(score, covered)
            all_parts.append(parts)
        else:
            # which part of the frame before becomes which part of the frame after (pixels of the cell)
            in_cell = cells_t.labels == cell
            before = overlaps[0].warped[in_cell]
            after = overlaps[1].warped[in_cell]
            pairs = []
            for p in all_parts[0]:
                candidates = {q: int(np.count_nonzero((before == p) & (after == q))) for q in all_parts[1]}
                pairs.append((p, max(candidates, key=candidates.get)))
            if len(set(q for _, q in pairs)) == len(pairs):
                splits.append({'cell': cell, 'parts': pairs, 'score': score})
    return splits


def _prune_membranes(labels, y0, y1, x0, x1):
    """
    Membrane pixels of the box that touch a single cell (e.g. the remains of a removed bond) become part of that cell
    (never the pixels of the image border).
    """
    h, w = labels.shape
    y0 = max(y0, 1)
    x0 = max(x0, 1)
    y1 = min(y1, h - 1)
    x1 = min(x1, w - 1)
    changed = True
    while changed:
        changed = False
        ys, xs = np.nonzero(labels[y0:y1, x0:x1] == MEMBRANE)
        for y, x in zip((ys + y0).tolist(), (xs + x0).tolist()):
            around = set(np.unique(labels[y - 1:y + 2, x - 1:x + 2]).tolist())
            around.discard(MEMBRANE)
            if len(around) == 1 and BACKGROUND not in around:
                labels[y, x] = around.pop()
                changed = True


def _box(mask, margin, shape):
    ys, xs = np.nonzero(mask)
    return (max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, shape[0]),
            max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, shape[1]))


def merge_cells(labels, cells):
    """
    Merges cells (track IDs) of a frame into the first one, removing the bonds between them (in place).

    Returns:
        bool: False if the merged cell would not be in one piece (nothing changed).
    """
    group = np.isin(labels, cells)
    y0, y1, x0, x1 = _box(group, 2, labels.shape)
    crop = labels[y0:y1, x0:x1]
    backup = crop.copy()
    crop[group[y0:y1, x0:x1]] = cells[0]
    _prune_membranes(labels, y0, y1, x0, x1)
    if label(crop == cells[0], connectivity=1).max() == 1:
        return True
    # short bonds (a pixel or two, next to vertices): their membrane pixels also touch other cells; they can join the
    # merged cell if none of their 4 neighbours is another cell (cells are 4-connected)
    h, w = labels.shape
    ys, xs = np.nonzero(crop == MEMBRANE)
    for y, x in zip((ys + y0).tolist(), (xs + x0).tolist()):
        if not (0 < y < h - 1 and 0 < x < w - 1):
            continue
        around = backup[max(y - 1 - y0, 0):y + 2 - y0, max(x - 1 - x0, 0):x + 2 - x0]
        if np.count_nonzero(np.isin(np.unique(around), cells)) < 2:
            continue
        cross = {int(labels[y - 1, x]), int(labels[y + 1, x]), int(labels[y, x - 1]), int(labels[y, x + 1])}
        if cross <= {MEMBRANE, cells[0]}:
            labels[y, x] = cells[0]
    if label(crop == cells[0], connectivity=1).max() == 1:
        return True
    crop[...] = backup
    return False


def split_cell(labels, cell, seeds, new_ids, elevation=None):
    """
    Splits a cell of a frame (in place) with a watershed seeded by seeds (boolean images, one per new cell) on the
    elevation image (bright membranes), or halfway between the seeds without elevation.

    Returns:
        bool: False if the cell could not be split (nothing changed).
    """
    region = labels == cell
    y0, y1, x0, x1 = _box(region, 2, labels.shape)
    crop_region = region[y0:y1, x0:x1]
    markers = np.zeros(crop_region.shape, np.int32)
    for k, seed in enumerate(seeds):
        seed = seed[y0:y1, x0:x1] & crop_region
        eroded = binary_erosion(seed)
        markers[(eroded if eroded.any() else seed) & (markers == 0)] = k + 1
    if len(np.unique(markers[markers > 0])) != len(seeds):
        return False
    if elevation is None:
        crop_elevation = np.zeros(crop_region.shape)
    else:
        crop_elevation = elevation[y0:y1, x0:x1]
    split = watershed(crop_elevation, markers, mask=crop_region, watershed_line=True)
    if label(split > 0, connectivity=1).max() != len(seeds):
        return False
    crop = labels[y0:y1, x0:x1]
    crop[crop_region & (split == 0)] = MEMBRANE
    for k, new_id in enumerate(new_ids):
        crop[split == k + 1] = new_id
    _prune_membranes(labels, y0, y1, x0, x1)
    return True


def bond_contrast(elevation, bond, cell, labels):
    """
    Membrane signal of a bond relative to its cell: (bond - cell interior) / (cell membranes - cell interior), from the
    medians of the (smoothed) raw image. ~1 for a real bond, ~0 for a line crossing the cytoplasm. None if not
    measurable.

    Args:
        elevation (ndarray): smoothed raw image.
        bond (ndarray): boolean image of the bond pixels.
        cell (ndarray): boolean image of the cell (the union of the cells the bond separates, bond included).
        labels (ndarray): image of track IDs (membranes = MEMBRANE).
    """
    if elevation is None or not bond.any():
        return None
    y0, y1, x0, x1 = _box(cell, 3, labels.shape)
    crop_cell = cell[y0:y1, x0:x1]
    crop_bond = bond[y0:y1, x0:x1]
    crop_elevation = elevation[y0:y1, x0:x1]
    around = binary_dilation(crop_cell, np.ones((3, 3), bool)) & ~crop_cell & (labels[y0:y1, x0:x1] == MEMBRANE)
    interior = binary_erosion(crop_cell, iterations=3) & ~binary_dilation(crop_bond, iterations=3)
    if not around.any() or not interior.any():
        return None
    inside = np.median(crop_elevation[interior])
    membranes = np.median(crop_elevation[around])
    if membranes <= inside:
        return None
    return float((np.median(crop_elevation[crop_bond]) - inside) / (membranes - inside))


def _elevation(filename, channel, shape):
    """smoothed raw image (membrane signal), None if it cannot be used"""
    try:
        raw = np.asarray(Img(filename)).astype(np.float64)
    except Exception:
        return None
    if raw.ndim == 3:
        if channel is not None and 0 <= channel < raw.shape[-1]:
            raw = raw[..., channel]
        else:
            raw = raw.max(axis=-1)
    if raw.shape != tuple(shape):
        return None
    # the watershed floods from the seeds: the membranes must be the highest points
    return gaussian_filter(raw, 1.)


def mask_path(filename):
    return input_path(filename, 'handCorrection')


def _save_mask(filename, original, corrected, backup=True):
    """Writes the membranes that changed into the mask of the image, keeping the previous mask (if backup)."""
    path = mask_path(filename)
    mask = np.asarray(Img(path)).copy()
    changed = (original == MEMBRANE) != (corrected == MEMBRANE)
    became_membrane = changed & (corrected == MEMBRANE)
    became_cell = changed & (corrected != MEMBRANE)
    mask[became_membrane] = 255
    mask[became_cell] = 0
    if backup:
        TA_path = smart_name_parser(filename, ordered_output='TA')
        shutil.copyfile(path, os.path.join(TA_path, BACKUP_NAME + os.path.splitext(path)[1]))
    Img(mask, dimensions='hwc' if mask.ndim == 3 else 'hw').save(path)


def _undo_list(lst):
    path = os.path.join(os.path.dirname(lst[0]), UNDO_LIST)
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _save_corrections_image(filename, labels, merged, split):
    out = np.zeros(labels.shape, np.uint32)
    if merged:
        out[np.isin(labels, list(merged))] = MERGE_COLOR
    if split:
        out[np.isin(labels, list(split))] = SPLIT_COLOR
    out[labels == MEMBRANE] = MEMBRANE
    TA_path = smart_name_parser(filename, ordered_output='TA')
    Img(int24_to_RGB(out), dimensions='hwc').save(os.path.join(TA_path, CORRECTIONS_IMAGE), mode='raw')


def _rounded(value):
    return '' if value is None else round(value, 3)


def correct_segmentation_over_time(lst, max_error_duration=4, min_inside=0.6, min_cover=0.6, max_area_ratio=1.5,
                                   min_part_area_fraction_of_median=0.1, channel=None, use_raw_image=True,
                                   max_merge_contrast=0.45, min_split_contrast=0.35, dry_run=False, round_nb=1,
                                   progress_callback=None):
    """
    Detects and corrects the transient over- and under-segmentations of a tracked movie (see module doc).

    Args:
        lst (list): ordered list of the movie frames (segmented and tracked).
        max_error_duration (int): longest error corrected (in frames).
        min_inside (float): min fraction of a cell lying in the cell it matches in a reference frame.
        min_cover (float): min fraction of a reference cell covered by the cells matching it.
        max_area_ratio (float): max area ratio between the corrected cell(s) and the reference cell(s).
        min_part_area_fraction_of_median (float): cells smaller than this fraction of the median area are not
            re-created by a split.
        channel (int, optional): channel of the raw image containing the membranes (split lines follow it); None = max of
            the channels.
        use_raw_image (bool): split along the membrane signal of the raw image (else halfway between the cells) and
            check the bond contrasts.
        max_merge_contrast (float, optional): a bond is not removed if its membrane signal is higher than this (see
            bond_contrast; the bonds removed by right merges are ~0.15, wrong ones up to ~0.8). None to skip.
        min_split_contrast (float, optional): a cell is not split if the membrane signal along the new bond is lower
            than this (right splits >= ~0.45; the lines drawn through the cytoplasm when the reference frames are
            over-segmented are < ~0.35). None to skip.
        dry_run (bool): only report (csv + segmentation_corrections.tif), do not change the masks.
        round_nb (int): > 1 when called again after re-tracking (see correct_segmentation_and_track): the csv is
            appended to and undo restores the masks as they were before round 1.
        progress_callback: Qt signal emitting an int percentage (optional).

    Returns:
        list: one dict per correction (see CSV_HEADER).
    """
    if not lst:
        return []
    stale = [t for t, f in enumerate(lst) if stale_reasons(f, 'tracking')]
    if stale:
        raise ValueError('The masks of image(s) ' + ', '.join(str(t) for t in stale[:10]) +
                         ' changed since the cells were tracked, please track the cells again first')

    def progress(value):
        try:
            if progress_callback is not None:
                progress_callback.emit(int(value))
        except Exception:
            pass

    n = len(lst)
    frames = {}
    originals = {}

    def frame(t):
        if t not in frames:
            tracks = load_tracks(lst[t])
            if tracks is None:
                raise ValueError('Image ' + str(t) + ' is not tracked, please track the cells first')
            frames[t] = tracks
            originals[t] = tracks.copy()
        return frames[t]

    cells_cache = {}

    def cells(t):
        if t not in cells_cache:
            cells_cache[t] = _Cells(frame(t))
        return cells_cache[t]

    next_id = [MEMBRANE]

    def new_id():
        # an ID absent from the frames in use (the cells get their final IDs when tracked again)
        next_id[0] -= 1
        while any(np.any(labels == next_id[0]) for labels in frames.values()):
            next_id[0] -= 1
        return next_id[0]

    elevations = {}

    def load_elevation(t):
        """smoothed raw image of frame t (None without raw image)"""
        if not use_raw_image:
            return None
        if t not in elevations:
            elevations.clear()
            elevations[t] = _elevation(lst[t], channel, frame(t).shape)
            if elevations[t] is None:
                logger.warning('Raw image of image ' + str(t) + ' not usable: cells are split halfway between their '
                               'parts and the bond contrasts are not checked')
        return elevations[t]

    corrections = []
    changed_frames = []
    finished = set()
    already_backed_up = set(_undo_list(lst)) if round_nb > 1 else set()

    def finish(t):
        """saves frame t (no longer needed) and forgets it"""
        if t in finished:
            return
        finished.add(t)
        merged = {c['track_ids'][0] for c in corrections if c['frame_nb'] == t and c['correction'] == MERGE}
        split = set()
        for c in corrections:
            if c['frame_nb'] == t and c['correction'] == SPLIT:
                split.update(c['track_ids'])
        if merged or split or round_nb == 1:
            _save_corrections_image(lst[t], frame(t), merged, split)
        if merged or split:
            changed_frames.append(t)
            if not dry_run:
                # later rounds keep the mask saved before the first round
                _save_mask(lst[t], originals[t], frame(t), backup=os.path.basename(lst[t]) not in already_backed_up)
        frames.pop(t, None)
        originals.pop(t, None)
        cells_cache.pop(t, None)

    for t in range(n):
        if early_stop.stop:
            break
        progress(100 * t / n)
        frame(t)
        if 1 <= t < n - 1:
            done = set()  # cells of t already corrected
            for duration in range(1, max_error_duration + 1):
                after = t + duration
                if after >= n:
                    break
                confirm = after + 1 if after + 1 < n else (t - 2 if t >= 2 else None)
                if confirm is None:
                    break
                references = [t - 1, after, confirm]
                cells_t = cells(t)
                overlaps = [_Overlap(cells_t, cells(r), r) for r in references]
                reference_text = '/'.join(str(r) for r in references)
                for merge in find_merges(cells_t, overlaps, min_inside, min_cover, max_area_ratio):
                    group = merge['cells']
                    if done & set(group):
                        continue
                    areas = [cells_t.area(c) for c in group]
                    # the cell keeps the ID of its biggest piece
                    biggest = group[int(np.argmax(areas))]
                    group = [biggest] + [c for c in group if c != biggest]
                    y, x = cells_t.centroids[cells_t.index[group[0]]]
                    elevation = load_elevation(t)
                    before = frame(t).copy()
                    if not merge_cells(frame(t), group):
                        logger.warning('Could not merge cells ' + str(group) + ' of image ' + str(t))
                        continue
                    contrast = bond_contrast(elevation, (before == MEMBRANE) & (frame(t) != MEMBRANE),
                                             frame(t) == group[0], frame(t))
                    if max_merge_contrast is not None and contrast is not None and contrast > max_merge_contrast:
                        # the removed bond is a visible membrane: not an over-segmentation
                        frame(t)[...] = before
                        continue
                    done.update(group)
                    corrections.append({'frame_nb': t, 'filename': lst[t], 'correction': MERGE, 'y': int(round(y)),
                                        'x': int(round(x)), 'cells_before': len(group), 'cells_after': 1,
                                        'area': int(sum(areas)), 'reference_frames': reference_text,
                                        'score': round(merge['score'], 3), 'contrast': _rounded(contrast),
                                        'track_ids': group})
                min_part_area = min_part_area_fraction_of_median * cells_t.median_area
                for split in find_splits(cells_t, overlaps, min_inside, min_cover, max_area_ratio, min_part_area):
                    cell = split['cell']
                    if cell in done:
                        continue
                    elevation = load_elevation(t)
                    seeds = [(overlaps[0].warped == p) & (overlaps[1].warped == q) for p, q in split['parts']]
                    # the cell keeps its ID for the part that had it before (if any), new IDs for the others
                    ids = [cell if p == cell else None for p, _ in split['parts']]
                    if cell not in ids:
                        ids[0] = cell
                    ids = [i if i is not None else new_id() for i in ids]
                    y, x = cells_t.centroids[cells_t.index[cell]]
                    area = cells_t.area(cell)
                    before = frame(t).copy()
                    if not split_cell(frame(t), cell, seeds, ids, elevation):
                        logger.warning('Could not split cell ' + str(cell) + ' of image ' + str(t))
                        continue
                    contrast = bond_contrast(elevation, (before != MEMBRANE) & (frame(t) == MEMBRANE), before == cell,
                                             before)
                    if min_split_contrast is not None and contrast is not None and contrast < min_split_contrast:
                        # no membrane where the cell would be cut: the references are the ones over-segmented
                        frame(t)[...] = before
                        continue
                    done.add(cell)
                    corrections.append({'frame_nb': t, 'filename': lst[t], 'correction': SPLIT, 'y': int(round(y)),
                                        'x': int(round(x)), 'cells_before': 1, 'cells_after': len(ids), 'area': area,
                                        'reference_frames': reference_text, 'score': round(split['score'], 3),
                                        'contrast': _rounded(contrast), 'track_ids': ids})
                cells_cache.pop(t, None)
        if t >= 3:
            # t-1 and t-2 (confirmation frame at the end of the movie) stay in memory, with their corrections
            finish(t - 3)
    for t in sorted(frames):
        finish(t)

    for c in corrections:
        c['track_ids'] = ' '.join(str(i) for i in c['track_ids'])
        c['round'] = round_nb
    folder = os.path.dirname(lst[0])
    csv_path = os.path.join(folder, CORRECTIONS_CSV)
    new_file = round_nb == 1 or not os.path.isfile(csv_path)
    with open(csv_path, 'w' if new_file else 'a', encoding='utf-8') as f:
        if new_file:
            f.write(','.join(CSV_HEADER) + '\n')
        for c in corrections:
            f.write(','.join(str(c[h]) for h in CSV_HEADER) + '\n')
    if not dry_run and (changed_frames or round_nb == 1):
        names = sorted(set(already_backed_up) | {os.path.basename(lst[t]) for t in changed_frames})
        with open(os.path.join(folder, UNDO_LIST), 'w', encoding='utf-8') as f:
            json.dump(names, f)
    nb_merges = sum(1 for c in corrections if c['correction'] == MERGE)
    logger.info('%d over-segmentation(s) merged and %d under-segmentation(s) split in %d image(s)%s' % (
        nb_merges, len(corrections) - nb_merges, len(changed_frames), ' (dry run, masks unchanged)' if dry_run else
        ', please track the cells again'))
    progress(100)
    return corrections


def tracks_missing_or_stale(lst):
    """True if a frame is not tracked or its mask changed since it was tracked."""
    for filename in lst:
        if not os.path.isfile(smart_name_parser(filename, ordered_output='tracked_cells_resized.tif')):
            return True
        if stale_reasons(filename, 'tracking'):
            return True
    return False


def correct_segmentation_and_track(lst, track_function, rounds=3, progress_callback=None, **parameters):
    """
    Tracks the cells if needed, then alternates correct_segmentation_over_time and tracking (a better tracking gives a
    better motion estimate, and corrected frames reveal the errors of their neighbours) until nothing is corrected or
    after rounds rounds. The masks and the tracks are up to date at the end (except in dry run: one round, no tracking
    after it).

    Args:
        lst (list): ordered list of the movie frames (segmented).
        track_function (callable): track_function(lst) tracks the cells of the movie.
        rounds (int): max number of correction rounds.
        progress_callback: Qt signal emitting an int percentage (optional).
        **parameters: parameters of correct_segmentation_over_time.

    Returns:
        list: the corrections of all rounds (dicts with a 'round' key).
    """
    dry_run = parameters.get('dry_run', False)
    if tracks_missing_or_stale(lst):
        logger.info('Tracking the cells before correcting the segmentation')
        track_function(lst)
    corrections = []
    for round_nb in range(1, max(rounds, 1) + 1):
        if early_stop.stop:
            break
        found = correct_segmentation_over_time(lst, round_nb=round_nb, progress_callback=progress_callback,
                                               **parameters)
        corrections.extend(found)
        if dry_run or not found:
            break
        logger.info('Round %d: %d correction(s), tracking the cells again' % (round_nb, len(found)))
        track_function(lst)
    return corrections


def undo_segmentation_corrections(lst):
    """
    Restores the masks as they were before the last correct_segmentation_over_time (not dry run; before its first round
    for correct_segmentation_and_track).

    Returns:
        int: number of masks restored.
    """
    folder = os.path.dirname(lst[0])
    undo_path = os.path.join(folder, UNDO_LIST)
    if not os.path.isfile(undo_path):
        return 0
    with open(undo_path, encoding='utf-8') as f:
        names = set(json.load(f))
    restored = 0
    for filename in lst:
        if os.path.basename(filename) not in names:
            continue
        path = mask_path(filename)
        backup = os.path.join(smart_name_parser(filename, ordered_output='TA'), BACKUP_NAME + os.path.splitext(path)[1])
        if os.path.isfile(backup):
            shutil.copyfile(backup, path)
            restored += 1
    os.remove(undo_path)
    return restored
