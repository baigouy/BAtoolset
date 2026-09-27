"""
Manual track corrections, applied to tracked_cells_resized.tif of the images of a movie (Tracking tab of pyTA):
    - swap_tracks: exchange two track IDs from a frame on (fixes an identity switch)
    - split_track: give a new track ID to a cell from a frame on (two cells were wrongly linked)
    - merge_tracks: give the ID of a track to another one (a track was broken in two)
    - reconnect_lost_track: merge a cell with the track that disappeared just before, where it overlaps it most
    - undo_last_edit: restores the tracks as they were before the last edit
After an edit, the local ID <-> track ID table (cell_tracks) of the changed images is updated; the results that
depend on the tracks (divisions, T1s, bonds, clones...) are then out of date and must be computed again.
"""
import json
import os
import random
import shutil
import numpy as np
from batoolset.files.tools import smart_name_parser
from batoolset.img import Img, int24_to_RGB
from batoolset.ta.tracking.divisions_and_deaths import load_tracks
from batoolset.tools.logger import TA_logger

logger = TA_logger()

MEMBRANE = 0xFFFFFF
BACKGROUND = 0
UNDO_NAME = 'tracked_cells_resized_before_edit.tif'
UNDO_LIST = 'track_edit_undo.json'


class TrackEditError(ValueError):
    pass


def _tracks_path(filename):
    return smart_name_parser(filename, ordered_output='tracked_cells_resized.tif')


def _undo_list_path(lst):
    return os.path.join(smart_name_parser(lst[0], ordered_output='parent'), UNDO_LIST)


def _load_all(lst, from_frame=0):
    tracks = {}
    for t in range(from_frame, len(lst)):
        tr = load_tracks(lst[t])
        if tr is None:
            raise TrackEditError('Image ' + str(t) + ' is not tracked, please track the cells first')
        tracks[t] = tr
    return tracks


def _save_edit(lst, new_tracks):
    """Saves the edited frames {frame: tracks}, keeping the previous version for undo_last_edit."""
    for t in new_tracks:
        path = _tracks_path(lst[t])
        shutil.copyfile(path, os.path.join(os.path.dirname(path), UNDO_NAME))
    with open(_undo_list_path(lst), 'w', encoding='utf-8') as f:
        json.dump([os.path.basename(lst[t]) for t in sorted(new_tracks)], f)
    for t, tracks in new_tracks.items():
        Img(int24_to_RGB(tracks), dimensions='hwc').save(_tracks_path(lst[t]), mode='raw')
    _update_cell_tracks(lst, sorted(new_tracks))


def _update_cell_tracks(lst, frames):
    from batoolset.ta.tracking.local_to_track_correspondance import add_localID_to_trackID_correspondance_in_DB
    to_update = [lst[t] for t in frames if os.path.isfile(smart_name_parser(lst[t], ordered_output='cells.tif'))]
    if to_update:
        add_localID_to_trackID_correspondance_in_DB(to_update)


def all_track_ids(lst):
    ids = set()
    for filename in lst:
        tracks = load_tracks(filename)
        if tracks is not None:
            ids.update(int(v) for v in np.unique(tracks))
    return ids - {MEMBRANE, BACKGROUND}


def new_track_id(used, rng=None):
    """A random track ID (int24, not 0 nor 0xFFFFFF) not in used."""
    rng = rng or random.Random()
    while True:
        candidate = rng.randint(1, MEMBRANE - 1)
        if candidate not in used:
            return candidate


def _check_present(tracks, track_id, frame):
    if not np.any(tracks == track_id):
        raise TrackEditError('Track ' + str(track_id) + ' is not in image ' + str(frame))


def swap_tracks(lst, id_a, id_b, from_frame):
    """Exchanges the IDs id_a and id_b in all images from from_frame on. Returns the changed frames."""
    if id_a == id_b:
        raise TrackEditError('Please select two different cells')
    tracks = _load_all(lst, from_frame)
    _check_present(tracks[from_frame], id_a, from_frame)
    _check_present(tracks[from_frame], id_b, from_frame)
    changed = {}
    for t, tr in tracks.items():
        a = tr == id_a
        b = tr == id_b
        if a.any() or b.any():
            tr = tr.copy()
            tr[a] = id_b
            tr[b] = id_a
            changed[t] = tr
    _save_edit(lst, changed)
    return sorted(changed)


def split_track(lst, track_id, from_frame, new_id=None):
    """Gives a new ID to track_id from from_frame on. Returns (new_id, changed frames)."""
    tracks = _load_all(lst, from_frame)
    _check_present(tracks[from_frame], track_id, from_frame)
    if new_id is None:
        new_id = new_track_id(all_track_ids(lst))
    changed = {}
    for t, tr in tracks.items():
        sel = tr == track_id
        if sel.any():
            tr = tr.copy()
            tr[sel] = new_id
            changed[t] = tr
    _save_edit(lst, changed)
    return new_id, sorted(changed)


def merge_tracks(lst, keep_id, merge_id, from_frame=0):
    """
    Replaces merge_id by keep_id from from_frame on. Refused (TrackEditError) if both IDs are present in the same
    image (they are two different cells there). Returns the changed frames.
    """
    if keep_id == merge_id:
        raise TrackEditError('Please select two different tracks')
    tracks = _load_all(lst, from_frame)
    both = [t for t, tr in tracks.items() if np.any(tr == keep_id) and np.any(tr == merge_id)]
    if both:
        raise TrackEditError('Tracks ' + str(keep_id) + ' and ' + str(merge_id) + ' are both present in image(s) '
                             + ', '.join(str(t) for t in both[:10]) + ': they are different cells')
    changed = {}
    for t, tr in tracks.items():
        sel = tr == merge_id
        if sel.any():
            tr = tr.copy()
            tr[sel] = keep_id
            changed[t] = tr
    if not changed:
        raise TrackEditError('Track ' + str(merge_id) + ' is not present from image ' + str(from_frame) + ' on')
    _save_edit(lst, changed)
    return sorted(changed)


def lost_track_for(lst, frame, track_id, max_gap=3):
    """
    The track that disappeared at most max_gap images before image frame (present in image frame - g, absent from
    image frame - g + 1 to image frame) and whose last position overlaps most the cell track_id of image frame (the
    most recent one when several overlap as much). None if there is none.
    """
    if frame <= 0:
        return None
    current = load_tracks(lst[frame])
    if current is None:
        return None
    cell = current == track_id
    # tracks present in an image between frame - gap + 1 and frame (did not disappear at frame - gap)
    seen_later = set(int(v) for v in np.unique(current))
    best = None
    for gap in range(1, min(max_gap, frame) + 1):
        previous = load_tracks(lst[frame - gap])
        if previous is None:
            break
        present = set(int(v) for v in np.unique(previous))
        lost = present - seen_later - {MEMBRANE, BACKGROUND}
        if lost:
            under = previous[cell]
            under = under[np.isin(under, list(lost))]
            if under.size:
                ids, counts = np.unique(under, return_counts=True)
                if best is None or counts.max() > best[1]:
                    best = (int(ids[counts.argmax()]), int(counts.max()))
        seen_later |= present
    return None if best is None else best[0]


def reconnect_lost_track(lst, frame, track_id, max_gap=3):
    """
    Gives the cell track_id of image frame (and its following images) the ID of the track lost at most max_gap images
    before, where it overlaps it most (see lost_track_for). Returns (lost track ID, changed frames).
    """
    lost = lost_track_for(lst, frame, track_id, max_gap=max_gap)
    if lost is None:
        raise TrackEditError('No track that disappeared in the ' + str(max_gap) + ' images before image ' +
                             str(frame) + ' overlaps cell ' + str(track_id))
    if np.any(load_tracks(lst[frame - 1]) == track_id):
        # the cell had its ID before: only its future is given the lost ID, its past keeps its own ID
        tracks = _load_all(lst, frame)
        both = [t for t, tr in tracks.items() if np.any(tr == lost) and np.any(tr == track_id)]
        if both:
            raise TrackEditError('Track ' + str(lost) + ' reappears with cell ' + str(track_id) + ' in image(s) ' +
                                 ', '.join(str(t) for t in both[:10]))
        changed = {}
        for t, tr in tracks.items():
            sel = tr == track_id
            if sel.any():
                tr = tr.copy()
                tr[sel] = lost
                changed[t] = tr
        _save_edit(lst, changed)
        return lost, sorted(changed)
    return lost, merge_tracks(lst, lost, track_id, from_frame=frame)


def undo_last_edit(lst):
    """Restores the tracks changed by the last edit. Returns the restored frames."""
    path = _undo_list_path(lst)
    if not os.path.isfile(path):
        raise TrackEditError('Nothing to undo')
    with open(path, encoding='utf-8') as f:
        names = set(json.load(f))
    restored = []
    for t, filename in enumerate(lst):
        if os.path.basename(filename) not in names:
            continue
        backup = os.path.join(os.path.dirname(_tracks_path(filename)), UNDO_NAME)
        if os.path.isfile(backup):
            os.replace(backup, _tracks_path(filename))
            restored.append(t)
    os.remove(path)
    _update_cell_tracks(lst, restored)
    return restored
