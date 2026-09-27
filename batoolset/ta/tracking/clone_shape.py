"""
Shape of clones: clone roughness, boundary roughness between two clones and clone wiggliness.

Port of TissueAnalyzer (Java) 'Clone roughness' (CloneRoughnessPluginContinuous + CloneRoughness, code by Maryam
Aliee), 'Boundary roughness' (BondLengthInterfaceVsClonesContinuous + BoundaryRougness) and 'Clone wiggliness'
(Wiggliness, idea by Paul Villoutreix).

The interface of a clone is the set of membrane pixels touching a cell of the clone and a cell of the other clone (or
any cell outside the clone). Each connected piece of interface is ordered from one end to the other (open interface:
the clone is cut by the image border or the interface is between two clones) or all around (closed interface).
Distances are divided by 'scale' (TissueAnalyzer default 20 px, about one bond length) so that results are in bond
length units.

    - clone roughness: for every pair of interface points, the chord length L and the arc length s along the
      interface between them (for closed interfaces the shortest way around); output = mean s per chord length bin
      (L0 = 10 bins per unit), for all interface pixels and for the interface vertices only
    - boundary roughness: the interface is rotated so that its ends lie on the x axis, resampled every ds = 0.05 unit
      and the roughness is the standard deviation of the height over windows of increasing length (every 20 * ds)
    - wiggliness: the angle at every vertex of the interface between the previous and the next vertex (pi = straight)

Outputs, per frame (in the TA folder, visible in the pyTA preview): clone_interface_XXX(_and_YYY).tif (the interface
in red over the membranes) and the 'wiggliness_...' table in pyTA.db (TissueAnalyzer columns); for the whole movie,
next to the images: clone_roughness_XXX.csv, boundary_roughness_XXX_and_YYY.csv and wiggliness_...csv.
"""
import math
import os
import numpy as np
from scipy import ndimage
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img, int24_to_RGB
from batoolset.tools.early_stopper_class import early_stop
from batoolset.ta.database.sql import TAsql
from batoolset.ta.tracking.bonds_and_vertices import membrane_contacts
from batoolset.ta.tracking.bond_kymographs import order_bond_pixels
from batoolset.ta.tracking.clones import load_clone, save_subclone_ids
from batoolset.ta.tracking.divisions_and_deaths import load_tracks
from batoolset.tools.logger import TA_logger

logger = TA_logger()

MEMBRANE = 0xFFFFFF
BACKGROUND = 0
INTERFACE_COLOR = 0xFF0000


# ---------------------------------------------------------------- interfaces
def interface_pixels(tracks, cells_1, cells_2=None):
    """
    Membrane pixels touching a cell of cells_1 and a cell of cells_2 (any other cell if cells_2 is None).

    Returns:
        tuple: ys, xs of the interface pixels and a boolean array telling which ones are vertices (>= 3 cells around)
    """
    cells_1 = np.asarray(sorted(cells_1), dtype=np.uint32)
    ys, xs, cells, nb_cells, _ = membrane_contacts(tracks)
    valid = cells != np.uint32(0xFFFFFFFF)
    in_1 = np.isin(cells, cells_1) & valid
    if cells_2 is None:
        in_2 = valid & ~in_1
    else:
        in_2 = np.isin(cells, np.asarray(sorted(cells_2), dtype=np.uint32)) & valid & ~in_1
    keep = in_1.any(axis=1) & in_2.any(axis=1)
    return ys[keep], xs[keep], nb_cells[keep] >= 3


def split_interface(tracks, ys, xs, is_vertex):
    """
    Splits the interface into connected pieces (8-connectivity) and orders their pixels.

    Returns:
        list: dicts with 'contour' (N x 2 ordered y, x), 'closed' (bool), 'vertices' (M x 2 ordered y, x, adjacent vertex
        pixels merged) and 'center' (mean y, x)
    """
    h, w = tracks.shape
    mask = np.zeros((h, w), bool)
    mask[ys, xs] = True
    vertex_mask = np.zeros((h, w), bool)
    vertex_mask[ys[is_vertex], xs[is_vertex]] = True
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    pieces = []
    for lab in range(1, n + 1):
        py, px = np.nonzero(labels == lab)
        if py.size < 2:
            continue
        touches_edge = py.min() <= 1 or px.min() <= 1 or py.max() >= h - 2 or px.max() >= w - 2
        path = order_bond_pixels(py, px)
        closed = (not touches_edge) and len(path) > 3 and max(abs(path[0] - path[-1])) <= 1
        pieces.append({'contour': path, 'closed': bool(closed), 'center': (float(py.mean()), float(px.mean())),
                       'vertices': _ordered_vertices(path, vertex_mask)})
    pieces.sort(key=lambda p: -len(p['contour']))
    return pieces


def _ordered_vertices(path, vertex_mask):
    """Vertices along the ordered path, adjacent vertex pixels (2 px vertices) merged into their mean position."""
    groups = []
    for y, x in path:
        if not vertex_mask[y, x]:
            continue
        if groups and max(abs(groups[-1][-1][0] - y), abs(groups[-1][-1][1] - x)) <= 1:
            groups[-1].append((y, x))
        else:
            groups.append([(y, x)])
    return np.asarray([np.mean(g, axis=0) for g in groups], dtype=float).reshape(-1, 2)


def interface_image(tracks, pieces):
    """Membranes in white and the interface pieces in red (TissueAnalyzer colors)."""
    out = np.zeros(tracks.shape, np.uint32)
    out[tracks == MEMBRANE] = MEMBRANE
    for piece in pieces:
        out[piece['contour'][:, 0], piece['contour'][:, 1]] = INTERFACE_COLOR
    return out


# ---------------------------------------------------------------- measures
def clone_roughness(points, scale=20., closed=True, L0=10.):
    """
    TissueAnalyzer CloneRoughness.CRoughness: mean arc length along the interface as a function of the chord length.

    Args:
        points (ndarray): N x 2 ordered interface points (y, x) in pixels.
        scale (float): unit length in pixels (TissueAnalyzer default 20, about one bond length).
        closed (bool): closed (clone inside the image) or open interface.
        L0 (float): number of chord length bins per unit.

    Returns:
        list: (chord length, mean arc length) for every chord length bin (bins without data are 0 as in TissueAnalyzer)
    """
    pts = np.asarray(points, dtype=float) / scale
    n = len(pts)
    if n < 3:
        return []
    bl = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    meanbl = bl.mean()
    bl = np.append(bl, np.sqrt(((pts[0] - pts[-1]) ** 2).sum()))
    sums = {}
    counts = {}
    lmax = 1.
    if closed:
        ext = np.concatenate([pts, pts])
        ext_bl = np.concatenate([bl, bl])
        cum = np.concatenate([[0.], np.cumsum(ext_bl)])
        starts = np.arange(n)
        half_perimeter = 0.5 * meanbl * n
        for i in range(1, n // 2):
            s = cum[starts + i] - cum[starts]
            chord = np.sqrt(((ext[starts + i] - ext[starts]) ** 2).sum(axis=1))
            sp = np.where(s > half_perimeter, meanbl * n - s, s)
            _accumulate(sums, counts, (chord * L0).astype(int), sp)
            lmax = max(lmax, chord.max())
    else:
        cum = np.concatenate([[0.], np.cumsum(bl[:-1])])
        limit = 5. + 0.5 * meanbl * n
        for i in range(1, n):
            starts = np.arange(n - i)
            s = cum[starts + i] - cum[starts]
            chord = np.sqrt(((pts[starts + i] - pts[starts]) ** 2).sum(axis=1))
            ok = s < limit
            if ok.any():
                _accumulate(sums, counts, (chord[ok] * L0).astype(int), s[ok])
                lmax = max(lmax, chord[ok].max())
    last = int(lmax * L0) - 1
    return [(k / L0, sums[k] / counts[k] if counts.get(k) else 0.) for k in range(last)]


def _accumulate(sums, counts, keys, values):
    uniq, inverse = np.unique(keys, return_inverse=True)
    s = np.bincount(inverse, weights=values)
    c = np.bincount(inverse)
    for k, sv, cv in zip(uniq.tolist(), s.tolist(), c.tolist()):
        sums[k] = sums.get(k, 0.) + sv
        counts[k] = counts.get(k, 0) + cv


def boundary_roughness(points, scale=20., ds=0.05, every=20):
    """
    TissueAnalyzer BoundaryRougness.interfaceRoughness: roughness of an (open) interface as a function of the window
    length. The interface is rotated so that the line joining its ends is the x axis (the Java code used the slope
    instead of the sine, which only matters for tilted interfaces), resampled every ds units along x (the height of
    the pixels falling in a bin is averaged, segments are linearly interpolated), then for windows of l + 1 samples the
    roughness is sqrt(sum over windows of the squared deviations to the window mean / ((l + 1) x nb of windows)).

    Returns:
        list: (window length, roughness) every 'every' samples
    """
    pts = np.asarray(points, dtype=float) / scale
    if len(pts) < 3:
        return []
    # Java: x = position along the boundary = image y, y = perpendicular = image x
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    angle = math.atan2(y[-1] - y[0], x[-1] - x[0])
    c, s = math.cos(angle), math.sin(angle)
    xr = c * x + s * y
    yr = c * y - s * x
    xr = np.abs(xr - xr[0])
    yr = yr - yr[0]
    num_k = int(int(xr[-1]) / ds)  # as in TissueAnalyzer (the end position is truncated to an integer first)
    if num_k < 3:
        return []
    size = int(xr.max() / ds) + 2
    h = np.zeros(size)
    count = np.zeros(size)
    for i in range(len(xr) - 1):
        dx = xr[i + 1] - xr[i]
        dy = yr[i + 1] - yr[i]
        if abs(dx) < ds:
            for xi, yi in ((xr[i], yr[i]), (xr[i + 1], yr[i + 1])):
                k = int(xi / ds)
                h[k] += yi
                count[k] += 1
        elif xr[i] < xr[i + 1]:
            ks = np.arange(int(xr[i] / ds), int(xr[i + 1] / ds))
            h[ks] += yr[i] + (ks * ds - xr[i]) * dy / dx
            count[ks] += 1
        else:
            ks = np.arange(int(xr[i + 1] / ds) + 1, int(xr[i] / ds))
            h[ks] += yr[i + 1] + (ks * ds - xr[i + 1]) * dy / dx
            count[ks] += 1
    h = h[:num_k]
    count = count[:num_k]
    h[count > 0] /= count[count > 0]
    c1 = np.concatenate([[0.], np.cumsum(h)])
    c2 = np.concatenate([[0.], np.cumsum(h * h)])
    out = []
    for l in range(1, num_k - 1):
        if l % every:
            continue
        nb_windows = num_k - l - 1
        starts = np.arange(nb_windows)
        sum1 = c1[starts + l + 1] - c1[starts]
        sum2 = c2[starts + l + 1] - c2[starts]
        squared_deviations = (sum2 - sum1 * sum1 / (l + 1)).sum()
        out.append((l * ds, math.sqrt(max(0., squared_deviations) / ((l + 1) * nb_windows))))
    return out


def wiggliness(vertices, closed, contour_length):
    """
    TissueAnalyzer Wiggliness: angle (radians, in [0, pi], pi = straight) at every vertex between the previous and the
    next vertex along the interface (extremities of open interfaces excluded).

    Returns:
        list: (x, y, angle, contour_length)
    """
    out = []
    n = len(vertices)
    for k in range(n):
        if not closed and (k == 0 or k == n - 1):
            continue
        if n < 3:
            continue
        cur = vertices[k]
        nxt = vertices[(k + 1) % n]
        prev = vertices[(k - 1) % n]
        angle = math.atan2(nxt[0] - cur[0], nxt[1] - cur[1]) - math.atan2(prev[0] - cur[0], prev[1] - cur[1])
        if angle < 0:
            angle += 2 * math.pi
        if angle > math.pi:
            angle = 2 * math.pi - angle
        out.append((float(cur[1]), float(cur[0]), angle, contour_length))
    return out


def contour_length(contour, closed):
    pts = np.asarray(contour, dtype=float)
    length = float(np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1)).sum())
    if closed and len(pts) > 1:
        length += float(np.sqrt(((pts[0] - pts[-1]) ** 2).sum()))
    return length


# ---------------------------------------------------------------- movies
def _clone_label(clone_nb_1, clone_nb_2=None):
    if clone_nb_2 is None or clone_nb_2 < 0 or clone_nb_2 == clone_nb_1:
        return '%03d' % clone_nb_1
    return '%03d_and_%03d' % (clone_nb_1, clone_nb_2)


def _frame_interface(filename, clone_nb_1, clone_nb_2=None, save_image=True):
    tracks = load_tracks(filename)
    if tracks is None:
        return None
    cells_1 = load_clone(filename, clone_nb_1, tracks)
    if cells_1 is None:
        logger.error('Clone ' + str(clone_nb_1) + ' does not exist for ' + str(filename))
        return None
    cells_2 = None
    if clone_nb_2 is not None and clone_nb_2 >= 0 and clone_nb_2 != clone_nb_1:
        cells_2 = load_clone(filename, clone_nb_2, tracks)
        if cells_2 is None:
            logger.error('Clone ' + str(clone_nb_2) + ' does not exist for ' + str(filename))
            return None
    if not cells_1:
        pieces = []
    else:
        ys, xs, is_vertex = interface_pixels(tracks, cells_1, cells_2)
        pieces = split_interface(tracks, ys, xs, is_vertex)
    if save_image:
        TA_path = smart_name_parser(filename, ordered_output='TA')
        Img(int24_to_RGB(interface_image(tracks, pieces)), dimensions='hwc').save(
            os.path.join(TA_path, 'clone_interface_' + _clone_label(clone_nb_1, clone_nb_2) + '.tif'), mode='raw')
    return pieces


def _progress(progress_callback, value):
    if progress_callback is not None:
        progress_callback.emit(int(value))


def compute_clone_roughness(lst, clone_nb=0, scale=20., store_subclone_ids=False, progress_callback=None):
    """
    Clone roughness of every piece of the clone outline in every frame, saved in clone_roughness_XXX.csv next to the
    images. With store_subclone_ids, the sub-clone ID of every clone cell is also saved in the 'local_clone_id_XXX'
    table of pyTA.db (as the TissueAnalyzer option).

    Returns:
        list: rows (frame_nb, filename, piece, closed, center_x, center_y, points, chord_length, mean_arc_length)
    """
    rows = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        if store_subclone_ids:
            save_subclone_ids(filename, clone_nb)
        pieces = _frame_interface(filename, clone_nb)
        if not pieces:
            continue
        for p, piece in enumerate(pieces):
            for kind, pts in (('all_pixels', piece['contour']), ('vertices_only', piece['vertices'])):
                for chord, arc in clone_roughness(pts, scale, piece['closed']):
                    rows.append((i, os.path.basename(filename), p + 1, int(piece['closed']), piece['center'][1],
                                 piece['center'][0], kind, chord, arc))
    parent = smart_name_parser(lst[0], ordered_output='parent')
    with open(os.path.join(parent, 'clone_roughness_' + _clone_label(clone_nb) + '.csv'), 'w', encoding='utf-8') as f:
        f.write('frame_nb,filename,interface_piece,closed,center_x,center_y,points,chord_length,mean_arc_length\n')
        for row in rows:
            f.write('%d,%s,%d,%d,%.2f,%.2f,%s,%.4f,%.6g\n' % row)
    return rows


def compute_boundary_roughness(lst, clone_nb_1=0, clone_nb_2=1, scale=20., progress_callback=None):
    """
    Roughness of the longest open interface between two clones in every frame, saved in
    boundary_roughness_XXX_and_YYY.csv next to the images.

    Returns:
        list: rows (frame_nb, filename, window_length, roughness)
    """
    clone_nb_1, clone_nb_2 = min(clone_nb_1, clone_nb_2), max(clone_nb_1, clone_nb_2)
    rows = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        pieces = _frame_interface(filename, clone_nb_1, clone_nb_2)
        open_pieces = [p for p in (pieces or []) if not p['closed']]
        if not open_pieces:
            continue
        for window, roughness in boundary_roughness(open_pieces[0]['contour'], scale):
            rows.append((i, os.path.basename(filename), window, roughness))
    parent = smart_name_parser(lst[0], ordered_output='parent')
    with open(os.path.join(parent, 'boundary_roughness_' + _clone_label(clone_nb_1, clone_nb_2) + '.csv'), 'w',
              encoding='utf-8') as f:
        f.write('frame_nb,filename,window_length,roughness\n')
        for row in rows:
            f.write('%d,%s,%.4f,%.6g\n' % row)
    return rows


WIGGLINESS_COLUMNS = ['central_vx_x', 'central_vx_y', 'angle_between_vectors', 'contour_length']


def compute_wiggliness(lst, clone_nb_1=0, clone_nb_2=-1, progress_callback=None):
    """
    Wiggliness of the outline of a clone (clone_nb_2 < 0) or of the interface between two clones in every frame: table
    in pyTA.db (TissueAnalyzer name and columns) and csv next to the images.

    Returns:
        list: rows (frame_nb, filename, interface piece, x, y, angle, contour_length)
    """
    two_clones = clone_nb_2 is not None and clone_nb_2 >= 0 and clone_nb_2 != clone_nb_1
    if two_clones:
        name = 'wiggliness_interface_between_clones_%d_and_%d' % (clone_nb_1, clone_nb_2)
    else:
        name = 'wiggliness_clone%d' % clone_nb_1
    rows = []
    for i, filename in enumerate(lst):
        if early_stop.stop:
            break
        _progress(progress_callback, i / len(lst) * 100)
        pieces = _frame_interface(filename, clone_nb_1, clone_nb_2 if two_clones else None)
        if pieces is None:
            continue
        table = {c: [] for c in WIGGLINESS_COLUMNS}
        for p, piece in enumerate(pieces):
            length = contour_length(piece['contour'], piece['closed'])
            for x, y, angle, length in wiggliness(piece['vertices'], piece['closed'], length):
                for col, value in zip(WIGGLINESS_COLUMNS, (x, y, angle, length)):
                    table[col].append(value)
                rows.append((i, os.path.basename(filename), p + 1, x, y, angle, length))
        db = TAsql(filename_or_connection=smart_name_parser(filename, ordered_output='pyTA.db'))
        try:
            db.drop_table(name)
            if table['central_vx_x']:
                db.create_and_append_table(name, table)
        finally:
            db.close()
    parent = smart_name_parser(lst[0], ordered_output='parent')
    with open(os.path.join(parent, name + '.csv'), 'w', encoding='utf-8') as f:
        f.write('frame_nb,filename,interface_piece,' + ','.join(WIGGLINESS_COLUMNS) + '\n')
        for row in rows:
            f.write('%d,%s,%d,%.2f,%.2f,%.6f,%.2f\n' % row)
    return rows
