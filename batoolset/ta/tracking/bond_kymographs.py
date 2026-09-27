"""
Bond kymographs and bond pixel intensities.

Port of TissueAnalyzer (Java) 'Create bond kymographs' (BondKymograph) and 'Plot bd px intensities' (testPixelValues).

For every tracked bond (see bonds_and_vertices.py, 'Track bonds and vertices' must be run first), the pixels of the bond
are ordered from one vertex to the other and their intensities in the original image make one line of the kymograph
(one line per frame). The bond is flipped when needed so that its ends stay on the same side from frame to frame.

Outputs, next to the images:
    - BondKymographs/<bond track id>.tif: height = frames where the bond exists, width = bond length (pixels), one
      channel per channel of the original image
    - bond_pixel_intensities.csv: frame_nb, bond track id, channel and the ordered pixel values of every bond
"""
import os
import numpy as np
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img
from batoolset.tools.early_stopper_class import early_stop
from batoolset.ta.database.sql import TAsql
from batoolset.ta.tracking.bonds_and_vertices import get_bond_pixels, membrane_contacts
from batoolset.ta.tracking.divisions_and_deaths import load_tracks
from batoolset.tools.logger import TA_logger

logger = TA_logger()

ALIGN_LEFT = 'left'
ALIGN_CENTER = 'center'
ALIGN_RIGHT = 'right'
STRETCH = 'stretch'
_NEIGHBOURS = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))  # 4-connected first


def order_bond_pixels(ys, xs):
    """
    Orders the pixels of a bond (a 1 px wide 8-connected line) from one end to the other.

    Returns:
        ndarray: N x 2 (y, x) ordered coordinates of the longest walk from an end of the bond.
    """
    pixels = {(int(y), int(x)) for y, x in zip(ys, xs)}
    if len(pixels) <= 1:
        return np.asarray(sorted(pixels), dtype=np.int64).reshape(-1, 2)

    def neighbours(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in _NEIGHBOURS if (p[0] + dy, p[1] + dx) in pixels]

    def walk(start):
        path = [start]
        seen = {start}
        cur = start
        while True:
            nxt = [n for n in neighbours(cur) if n not in seen]
            if not nxt:
                return path
            # prefer the neighbour that has the fewest unvisited neighbours itself (follows the line, not a shortcut)
            cur = min(nxt, key=lambda n: sum(m not in seen for m in neighbours(n)))
            seen.add(cur)
            path.append(cur)

    degree = {p: len(neighbours(p)) for p in pixels}
    min_degree = min(degree.values())
    ends = sorted(p for p, d in degree.items() if d == min_degree)
    # start from the end farthest from the centroid (robust when there is no degree 1 pixel)
    centroid = np.mean(np.asarray(list(pixels), dtype=float), axis=0)
    start = max(ends, key=lambda p: (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
    path = walk(start)
    path = max((path, walk(path[-1])), key=len)
    return np.asarray(path, dtype=np.int64)


def _bond_track_ids(filename):
    """(cell_track_id1, cell_track_id2) -> bond track id from the 'bond_tracks' table of the frame."""
    db_path = smart_name_parser(filename, ordered_output='pyTA.db')
    if not os.path.isfile(db_path):
        return None
    db = TAsql(filename_or_connection=db_path)
    try:
        if 'bond_tracks' not in db.get_tables():
            return None
        rows = db.run_SQL_command_and_get_results(
            'SELECT DISTINCT cell_track_id1, cell_track_id2, track_id FROM bond_tracks WHERE track_id IS NOT NULL')
    finally:
        db.close()
    return {(int(a), int(b)): int(t) for a, b, t in rows}


def _load_original(filename):
    img = np.asarray(Img(filename))
    if img.ndim == 2:
        img = img[..., np.newaxis]
    return img


def _flip_if_needed(line, ends, prev_ends):
    """Flips the line if its ends match the previous ends better in the reverse order (TissueAnalyzer isMinDist)."""
    if prev_ends is None:
        return line, ends
    straight = ((ends[0] - prev_ends[0]) ** 2).sum() + ((ends[1] - prev_ends[1]) ** 2).sum()
    flipped = ((ends[1] - prev_ends[0]) ** 2).sum() + ((ends[0] - prev_ends[1]) ** 2).sum()
    if flipped < straight:
        return line[::-1], ends[::-1]
    return line, ends


def build_kymograph(lines, alignment=ALIGN_LEFT):
    """
    Stacks the lines (one per frame, N x channels) of a bond into a kymograph (frames x max length x channels).
    """
    width = max(len(line) for line in lines)
    channels = lines[0].shape[-1]
    kymo = np.zeros((len(lines), width, channels), dtype=lines[0].dtype)
    for row, line in enumerate(lines):
        n = len(line)
        if n == 0:
            continue
        if alignment == STRETCH and n != width:
            positions = np.linspace(0, n - 1, width)
            kymo[row] = np.stack([np.interp(positions, np.arange(n), line[:, c]) for c in range(channels)],
                                 axis=-1).astype(kymo.dtype)
            continue
        if alignment == ALIGN_RIGHT:
            pad = width - n
        elif alignment == ALIGN_CENTER:
            pad = (width - n) // 2
        else:
            pad = 0
        kymo[row, pad:pad + n] = line
    return kymo


def create_bond_kymographs(lst, alignment=ALIGN_LEFT, min_nb_of_frames=0, save_kymographs=True,
                           save_pixel_intensities=True, progress_callback=None):
    """
    Creates the bond kymographs and/or the bond pixel intensities file (see module doc).

    Args:
        lst (list): the ordered list of images of the movie (bonds tracked with pyTA).
        alignment (str): 'left', 'center', 'right' or 'stretch' (all lines resampled to the longest one).
        min_nb_of_frames (int): kymographs are only saved for bonds present in more frames than this.
        save_kymographs (bool): save the BondKymographs/*.tif files.
        save_pixel_intensities (bool): save bond_pixel_intensities.csv.
        progress_callback: optional Qt signal receiving a percentage.

    Returns:
        dict: bond track id -> kymograph (frames x length x channels)
    """
    if not lst:
        return {}
    lines = {}
    prev_ends = {}
    parent = smart_name_parser(lst[0], ordered_output='parent')
    csv_file = None
    if save_pixel_intensities:
        csv_file = open(os.path.join(parent, 'bond_pixel_intensities.csv'), 'w', encoding='utf-8')
        csv_file.write('frame_nb,filename,bond_track_id,channel,pixel_values\n')
    try:
        missing = 0
        for i, filename in enumerate(lst):
            if early_stop.stop:
                return {}
            if progress_callback is not None:
                progress_callback.emit(int(i / len(lst) * 90))
            ids = _bond_track_ids(filename)
            tracks = load_tracks(filename)
            if ids is None or tracks is None:
                missing += 1
                continue
            original = _load_original(filename)
            if original.shape[:2] != tracks.shape:
                logger.error('Image and tracks sizes differ for ' + str(filename))
                continue
            for pair, (ys, xs) in get_bond_pixels(tracks, membrane_contacts(tracks)).items():
                track_id = ids.get(pair)
                if track_id is None:
                    continue
                path = order_bond_pixels(ys, xs)
                line = original[path[:, 0], path[:, 1]]
                ends = path[[0, -1]].astype(float)
                line, ends = _flip_if_needed(line, ends, prev_ends.get(track_id))
                prev_ends[track_id] = ends
                lines.setdefault(track_id, []).append(line)
                if csv_file is not None:
                    for c in range(line.shape[-1]):
                        csv_file.write('%d,%s,%d,%d,%s\n' % (i, os.path.basename(filename), track_id, c,
                                                             ' '.join(str(v) for v in line[:, c].tolist())))
        if missing == len(lst):
            logger.error('No tracked bonds found, please run "Track bonds and vertices" first')
            return {}
    finally:
        if csv_file is not None:
            csv_file.close()

    kymographs = {}
    output_folder = os.path.join(parent, 'BondKymographs')
    for n, (track_id, bond_lines) in enumerate(sorted(lines.items())):
        if early_stop.stop:
            break
        if progress_callback is not None:
            progress_callback.emit(90 + int(n / max(1, len(lines)) * 10))
        if len(bond_lines) <= min_nb_of_frames:
            continue
        kymo = build_kymograph(bond_lines, alignment)
        kymographs[track_id] = kymo
        if save_kymographs:
            os.makedirs(output_folder, exist_ok=True)
            out = kymo[..., 0] if kymo.shape[-1] == 1 else kymo
            Img(out, dimensions='hw' if out.ndim == 2 else 'hwc').save(
                os.path.join(output_folder, str(track_id) + '.tif'), mode='raw')
    logger.info(str(len(kymographs)) + ' bond kymographs')
    return kymographs
