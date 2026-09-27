"""
Out-of-date results detection.

Every analysis stage records, per image, a fingerprint (hash of the content) of the files it used, in
<TA folder>/pyTA_provenance.json. A result is out of date when one of these files changed since the stage was run,
e.g. a mask corrected after 'Measure cell properties', or cells tracked again after 'Detect T1s'.

Stages and the files they depend on (names in the TA folder of the image, 'raw' = the image itself, 'handCorrection'
= handCorrection.tif, or handCorrection.png when there is no tif):
"""
import datetime
import hashlib
import json
import os
from collections import OrderedDict
from batoolset.files.tools import smart_name_parser
from batoolset.tools.logger import TA_logger

logger = TA_logger()

PROVENANCE_FILE = 'pyTA_provenance.json'

STAGES = OrderedDict([
    ('measurements', {'label': 'Measure cell properties', 'inputs': ['handCorrection', 'raw']}),
    ('tracking', {'label': 'Track cells', 'inputs': ['handCorrection', 'cells.tif']}),
    ('divisions_and_deaths', {'label': 'Detect cell divisions and deaths', 'inputs': ['tracked_cells_resized.tif']}),
    ('T1s', {'label': 'Detect T1s', 'inputs': ['tracked_cells_resized.tif']}),
    ('bonds_and_vertices', {'label': 'Track bonds and vertices',
                            'inputs': ['tracked_cells_resized.tif', 'cells.tif', 'bonds.tif', 'vertices.tif']}),
])

__doc__ += '\n'.join('    %s: %s' % (stage, ', '.join(info['inputs'])) for stage, info in STAGES.items())


def _TA_path(filename):
    return smart_name_parser(filename, ordered_output='TA')


def input_path(filename, name):
    """Path of an input file of a stage (see the module doc for the names)."""
    if name == 'raw':
        return filename
    TA_path = _TA_path(filename)
    if name == 'handCorrection':
        tif = os.path.join(TA_path, 'handCorrection.tif')
        png = os.path.join(TA_path, 'handCorrection.png')
        return tif if os.path.isfile(tif) or not os.path.isfile(png) else png
    return os.path.join(TA_path, name)


def fingerprint(path):
    """Hash of the content of a file, None if it does not exist."""
    if not os.path.isfile(path):
        return None
    h = hashlib.blake2b(digest_size=16)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_provenance(filename):
    path = os.path.join(_TA_path(filename), PROVENANCE_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        logger.warning('Could not read ' + str(path))
        return {}


def record_stage(filename, stage):
    """Records that stage was just run on filename, with the fingerprints of its input files."""
    if stage not in STAGES:
        raise ValueError('Unknown stage ' + str(stage))
    TA_path = _TA_path(filename)
    if not os.path.isdir(TA_path):
        return
    data = load_provenance(filename)
    data[stage] = {'time': datetime.datetime.now().isoformat(timespec='seconds'),
                   'inputs': {name: fingerprint(input_path(filename, name)) for name in STAGES[stage]['inputs']}}
    path = os.path.join(TA_path, PROVENANCE_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def record_stage_for_list(lst, stage):
    for filename in lst:
        record_stage(filename, stage)


def stale_reasons(filename, stage):
    """
    Returns the input files of stage that changed since it was run on filename ([] when up to date), or None when
    the stage was never recorded for this image.
    """
    record = load_provenance(filename).get(stage)
    if record is None:
        return None
    changed = []
    for name, old in record.get('inputs', {}).items():
        if fingerprint(input_path(filename, name)) != old:
            changed.append(name)
    return changed


def stale_frames(lst, stage):
    """
    Returns [(frame_nb, filename, changed input names)] for the images where stage is out of date.
    """
    out = []
    for frame_nb, filename in enumerate(lst):
        changed = stale_reasons(filename, stage)
        if changed:
            out.append((frame_nb, filename, changed))
    return out


def _frames_text(frames, max_shown=10):
    shown = ', '.join(str(f) for f in frames[:max_shown])
    if len(frames) > max_shown:
        shown += '... (' + str(len(frames)) + ' images)'
    return shown


def staleness_report(lst, stages=None):
    """
    Returns a human readable list of the out of date results (one line per stage), [] when everything is up to date.
    """
    lines = []
    for stage in stages or STAGES:
        stale = stale_frames(lst, stage)
        if not stale:
            continue
        changed = sorted({name for _, _, names in stale for name in names})
        lines.append('"%s" is out of date for image(s) %s: %s changed since it was run' % (
            STAGES[stage]['label'], _frames_text([f for f, _, _ in stale]), ', '.join(changed)))
    return lines


def warn_if_stale(lst, stages=None):
    """Logs a warning for every out of date stage; returns the report lines."""
    lines = staleness_report(lst, stages)
    for line in lines:
        logger.warning(line + ' --> please run it again')
    return lines
