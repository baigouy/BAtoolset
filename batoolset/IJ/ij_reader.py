"""
ij_reader.py
============
PyImageJ utility for reading Bio-Formats files (LIF, LSM, TIF, GIF, etc.).

Provides:
    - Series listing and selection
    - LUT extraction per channel
    - Metadata extraction (physical voxel sizes, channel names, acquisition date)
    - ROI / Overlay extraction with mask conversion
    - Image display via matplotlib

Usage (standalone):
    python ij_reader.py

Usage (as a module):
    from ij_reader import init_ij, list_and_open_series, get_luts, to_metadata
    ij = init_ij()
    imp  = list_and_open_series('/path/to/file.lif', series_index=0)
    luts = get_luts(imp)
    meta = to_metadata(imp, luts=luts)
"""
import sys
import logging as _logging

from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()# none of the label drawings is great --> ignore t for now or maybe add it to the vectorial draw pane but do not allow edit and save of this and move it with the image!!!
import scyjava as sj
import jpype
import numpy as np
from batoolset.tools.logger import TA_logger  # logging
logger = TA_logger()  # logging_level=TA_logger.DEBUG
import os
import traceback


import threading as _threading

# at module level in ij_reader.py (add after imports)
_ij_gateway = None  # lazy singleton: None = not yet, False = init failed, else the gateway
# Guards gateway creation AND every Bio-Formats read: init_ij() is called from a
# startup thread while images may already be loading on a background loader
# thread, and imagej.init() must never run twice / concurrently in one process.
ij_lock = _threading.RLock()

# module-level callback — GUI can set this before first image open
_on_fiji_init_callback = None  # callable() or None



def set_fiji_init_callback(callback):
    """
    Register a callback to be called just before Fiji is initialised.
    Useful for showing a GUI warning about slow first-time download.

    Parameters
    ----------
    callback : callable  Called with no arguments before imagej.init().
    """
    global _on_fiji_init_callback
    _on_fiji_init_callback = callback

# ---------------------------------------------------------------------------
# Console logging helpers (non-GUI mode)
# ---------------------------------------------------------------------------

def _attach_console_logging():
    """
    Route jgo/imagej/scyjava/cjdk DEBUG output to stdout so the user can
    see download progress when there is no GUI dialog.
    Returns a list of (logger, handler, prev_level) for cleanup.
    """
    handler = _logging.StreamHandler(sys.stdout)
    handler.setLevel(_logging.DEBUG)
    handler.setFormatter(_logging.Formatter('[%(name)s] %(message)s'))
    patched = []
    for name in ('jgo', 'imagej', 'scyjava', 'cjdk'):
        lg = _logging.getLogger(name)
        prev_level = lg.level
        lg.setLevel(_logging.DEBUG)
        lg.addHandler(handler)
        patched.append((lg, handler, prev_level))
    return patched


def _detach_console_logging(patched):
    for lg, handler, prev_level in patched:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------



def silence_java_logging():
    """
    Suppress SciJava / Bio-Formats console noise.
    Must be called BEFORE init_ij() / imagej.init().
    """
    scyjava_config = sj.config
    scyjava_config.add_option('-Djava.awt.headless=true')    # prevent DefaultPlatformService AWT init failure
    scyjava_config.add_option('-Dscijava.log.level=ERROR')   # suppress SciJava logs
    scyjava_config.add_option('-Dloci.bioformats.printStackTraces=false')

def _cleanup_duplicate_jgo_fiji_envs():
    """
    Remove stale duplicate Fiji jgo environment directories.

    jgo creates a content-addressed directory under ~/.jgo/envs/sc/fiji/fiji/<hash>/
    for each dependency resolution.  When imagej.init() is interrupted or fails
    mid-download, an incomplete directory is left behind.  On the next launch jgo
    may resolve to a different hash, so multiple entries accumulate — causing
    SLF4J multiple-binding warnings and potential classpath conflicts.

    Strategy: keep the single directory with the most files (most complete
    download); delete the rest.  If all directories have the same file count
    the newest one (by mtime) is kept.
    """
    fiji_env_root = os.path.expanduser('~/.jgo/envs/sc/fiji/fiji')
    if not os.path.isdir(fiji_env_root):
        return
    entries = [
        os.path.join(fiji_env_root, d)
        for d in os.listdir(fiji_env_root)
        if os.path.isdir(os.path.join(fiji_env_root, d))
    ]
    if len(entries) <= 1:
        return

    def _score(path):
        # count files recursively; tie-break on modification time
        n = sum(len(files) for _, _, files in os.walk(path))
        mtime = os.path.getmtime(path)
        return (n, mtime)

    entries.sort(key=_score, reverse=True)
    keep = entries[0]
    for stale in entries[1:]:
        try:
            import shutil
            shutil.rmtree(stale)
            logger.info(f'Removed stale Fiji jgo env: {stale}')
        except Exception as e:
            logger.warning(f'Could not remove stale Fiji jgo env {stale}: {e}')


def _is_fiji_cached():
    """Return True if Fiji has already been downloaded to the Maven/jgo cache."""
    jgo_cache = os.path.expanduser('~/.jgo')
    if os.path.exists(jgo_cache):
        for root, dirs, files in os.walk(jgo_cache):
            if 'fiji' in root.lower():
                return True
    m2_cache = os.path.expanduser('~/.m2/repository/sc/fiji')
    return os.path.exists(m2_cache)

def silence_bioformats_logging():
    """
    Suppress Bio-Formats log output via DebugTools.
    Must be called AFTER the JVM has started (i.e. after init_ij()).
    Levels: OFF, ERROR, WARN, INFO, DEBUG, TRACE.
    """
    DebugTools = sj.jimport('loci.common.DebugTools')
    DebugTools.setRootLevel('ERROR')  # options: OFF, ERROR, WARN, INFO, DEBUG, TRACE


def init_ij_threaded(parent_widget=None, on_done=None, on_error=None):
    """
    Initialise Fiji in a background thread so the GUI stays responsive.

    Parameters
    ----------
    parent_widget : QWidget or None
    on_done       : callable(ij)   called in the main thread when done
    on_error      : callable(str)  called in the main thread on failure
    """
    import threading
    from qtpy.QtCore import QTimer

    result = {'ij': None, 'error': None, 'done': False}

    def _run():
        try:
            result['ij'] = init_ij(parent_widget=parent_widget)
        except Exception as e:
            result['error'] = str(e)
        finally:
            result['done'] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Poll from the main thread via QTimer so Qt stays alive
    def _poll():
        if result['done']:
            if result['error'] is not None:
                if on_error:
                    on_error(result['error'])
            else:
                # init_ij() has already stored the gateway singleton
                if on_done:
                    on_done(result['ij'])
        else:
            QTimer.singleShot(200, _poll)

    QTimer.singleShot(200, _poll)

def _clear_jgo_cache():
    """Remove the jgo cache to recover from maven cache corruption."""
    import shutil
    jgo_cache = os.path.expanduser('~/.jgo')
    if os.path.isdir(jgo_cache):
        shutil.rmtree(jgo_cache)
        logger.info('Cleared corrupted jgo cache — retrying Fiji init')


def reset_fiji_cache():
    """
    Delete all Fiji-related download caches so the next init_ij() call
    re-downloads everything from scratch. Useful for testing.

    Removes:
      - ~/.jgo          (jgo environment / classpath cache)
      - ~/.m2/repository/sc/fiji  (Maven artifact cache for Fiji)
      - %LOCALAPPDATA%/cjdk/cache (cjdk JDK download cache, Windows)
      - ~/.cjdk/cache             (cjdk JDK download cache, Linux/macOS)
    """
    import shutil
    removed = []

    paths = [
        os.path.expanduser('~/.jgo'),
        os.path.expanduser('~/.m2/repository/sc/fiji'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'cjdk', 'cache'),
        os.path.expanduser('~/.cjdk/cache'),
    ]

    for path in paths:
        if path and os.path.isdir(path):
            try:
                shutil.rmtree(path)
                removed.append(path)
                print(f'Removed: {path}')
            except Exception as e:
                print(f'Could not remove {path}: {e}')

    if removed:
        print('Fiji cache reset — next init_ij() will re-download from scratch.')
    else:
        print('Nothing to remove — Fiji cache was already empty.')


def _use_conda_java():
    """
    Point jgo to an existing Java so cjdk does not download its own.

    jgo's JAVA_HOME check has a Windows bug (looks for 'java' not 'java.exe'),
    so it always falls through to cjdk on Windows even when JAVA_HOME is set.
    Fix: add the java bin dir to PATH so shutil.which('java') finds it,
    which is jgo's second check and works cross-platform.
    Also monkey-patch cjdk.java_home as a belt-and-suspenders fallback.
    """
    import sys
    import shutil
    java_bin = 'java.exe' if os.name == 'nt' else 'java'

    # Candidate locations vary by platform:
    #   Windows: conda puts the JDK under <prefix>\Library\lib\jvm (openjdk) or \Library
    #   Linux/macOS: conda puts java directly under <prefix>/bin
    if os.name == 'nt':
        candidates = [
            os.path.join(sys.prefix, 'Library', 'lib', 'jvm'),
            os.path.join(sys.prefix, 'Library'),
        ]
    else:
        candidates = [
            sys.prefix,
        ]
    java_home = None
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, 'bin', java_bin)):
            java_home = candidate
            break

    # Fall back to whatever java is already on PATH
    if java_home is None:
        java_cmd = shutil.which('java')
        if java_cmd:
            java_home = str(os.path.dirname(os.path.dirname(java_cmd)))

    if java_home is None:
        logger.warning('No Java found — cjdk will download its own (run: conda install -c conda-forge openjdk=11)')
        return

    bin_dir = os.path.join(java_home, 'bin')
    os.environ['JAVA_HOME'] = java_home

    # Prepend bin_dir to PATH so jgo's shutil.which('java') finds it
    path = os.environ.get('PATH', '')
    if bin_dir not in path:
        os.environ['PATH'] = bin_dir + os.pathsep + path

    # Monkey-patch cjdk as a fallback in case shutil.which still misses
    try:
        import cjdk
        _jh = java_home
        cjdk.java_home = lambda *args, **kwargs: _jh
    except ImportError:
        pass

    logger.info(f'Java set to: {java_home}')


def init_ij(parent_widget=None):
    """Return the process-wide ImageJ gateway, initialising Fiji on first call.

    Idempotent and thread-safe: later/concurrent callers get the same gateway
    (previously the gateway returned at startup was discarded, so the first
    image read initialised Fiji a second time). Raises if a previous init
    failed."""
    global _ij_gateway
    with ij_lock:
        if _ij_gateway is False:
            raise RuntimeError('Fiji init previously failed; skipping')
        if _ij_gateway is not None:
            return _ij_gateway
        try:
            _ij_gateway = _init_ij_unlocked(parent_widget)
        except Exception:
            _ij_gateway = False  # prevent endless retries
            raise
        return _ij_gateway


def _init_ij_unlocked(parent_widget=None):
    _cleanup_duplicate_jgo_fiji_envs()
    _use_conda_java()
    silence_java_logging()

    gui_mode = _on_fiji_init_callback is not None
    cached = _is_fiji_cached()

    if not gui_mode:
        if not cached:
            print('Fiji not found in local cache — downloading now (this may take several minutes on first run)...', flush=True)
        else:
            print('Initializing Fiji...', flush=True)

    # Fire callback BEFORE the blocking imagej.init() call
    if gui_mode:
        _on_fiji_init_callback()

    # In non-GUI mode route library debug logs to stdout so the user sees progress
    patched = _attach_console_logging() if not gui_mode else []
    import imagej
    try:
        try:
            ij = imagej.init('sc.fiji:fiji', mode='headless')
        except AssertionError:
            # jgo maven cache corruption (cached_file already exists) — wipe and retry
            _clear_jgo_cache()
            ij = imagej.init('sc.fiji:fiji', mode='headless')
    finally:
        _detach_console_logging(patched)

    if not gui_mode:
        print('Fiji initialized successfully.', flush=True)
    return ij

def _java_imports():
    """
    Return a dict of commonly used Java classes.
    Called after the JVM is started; results are not cached so each call
    resolves the classes fresh (JPype handles the caching internally).
    """
    return {
        'ImporterOptions': sj.jimport('loci.plugins.in.ImporterOptions'),
        'ImportProcess'  : sj.jimport('loci.plugins.in.ImportProcess'),
        'ImagePlusReader': sj.jimport('loci.plugins.in.ImagePlusReader'),
        'CompositeImage' : sj.jimport('ij.CompositeImage'),
        'LUT'            : sj.jimport('ij.process.LUT'),
        'IndexColorModel': sj.jimport('java.awt.image.IndexColorModel'),
        'ByteProcessor'  : sj.jimport('ij.process.ByteProcessor'),
        'RoiManager'     : sj.jimport('ij.plugin.frame.RoiManager'),
        'IJ'             : sj.jimport('ij.IJ'),
    }

# Mapping from ImageJ ROI type integer to human-readable name.
ROI_TYPES = {
    0: 'polygon',
    1: 'rect',
    2: 'oval',
    3: 'line',
    4: 'freeline',
    5: 'polyline',
    6: 'no_roi',
    7: 'freehand',
    8: 'traced',
    9: 'angle',
    10: 'point',
}



def get_rois(imp):
    """
    Extract all ROIs from an ImagePlus with full details.

    Collects ROIs from three sources:
      - The single ROI attached directly to the image (imp.getRoi())
      - The image Overlay (multiple ROIs drawn on top)
      - The ROI Manager (if open and populated)

    Parameters
    ----------
    imp : ImagePlus

    Returns
    -------
    dict with keys:
        'single_roi'  : roi_info dict or None
        'overlay'     : list of roi_info dicts
        'roi_manager' : list of roi_info dicts

    Each roi_info dict contains:
        roi           : raw Java ROI object (use directly with roi_to_mask())
        index         : int
        name          : str
        type          : str  (e.g. 'polygon', 'rect', 'oval', ...)
        type_int      : int  (ImageJ internal type code)
        z, c, t       : int  hyperstack position (1-indexed, 0 = all planes)
        position      : int  flat stack position
        bounds        : dict {x, y, w, h}
        stroke_color  : str hex or None
        fill_color    : str hex or None
        stroke_width  : float
        coords        : dict  type-dependent coordinate data (see get_roi_coords)
        group         : int or None
    """
    RoiManager  = sj.jimport('ij.plugin.frame.RoiManager')

    def color_to_hex(c):
        if c is None:
            return None
        return '#{:02x}{:02x}{:02x}'.format(
            int(c.getRed()), int(c.getGreen()), int(c.getBlue()))

    def get_roi_coords(roi):
        """
              Extract ROI coordinates in a type-appropriate format.

              Returns a dict whose keys depend on ROI type:
                polygon / freehand / polyline : {'x': [...], 'y': [...]}
                line                          : {'x1', 'y1', 'x2', 'y2', 'width'}
                point                         : {'x': [...], 'y': [...]}
                rect / oval                   : {'x', 'y', 'w', 'h'}  (bounding box)
              Returns {} on any error.
        """
        try:
            roi_type = int(roi.getType())
            if roi_type in (0, 4, 5, 7, 8):
                poly = roi.getPolygon()
                return {
                    'x': [int(x) for x in poly.xpoints[:poly.npoints]],
                    'y': [int(y) for y in poly.ypoints[:poly.npoints]],
                }
            elif roi_type == 3:
                return {
                    'x1': int(roi.x1), 'y1': int(roi.y1),
                    'x2': int(roi.x2), 'y2': int(roi.y2),
                    'width': float(roi.getStrokeWidth()),
                }
            elif roi_type == 10:
                pts = roi.getContainedPoints()
                return {
                    'x': [int(p.x) for p in pts],
                    'y': [int(p.y) for p in pts],
                }
            else:
                b = roi.getBounds()
                return {
                    'x': int(b.x), 'y': int(b.y),
                    'w': int(b.width), 'h': int(b.height),
                }
        except:
            return {}

    def extract_roi_info(roi, index):
        """Build a roi_info dict from a raw Java ROI object."""
        roi_type = int(roi.getType())
        bounds   = roi.getBounds()
        stroke_c = roi.getStrokeColor()
        fill_c   = roi.getFillColor()
        return {
            'roi'         : roi,          # <-- raw Java ROI for mask generation etc.
            'index'       : index,
            'name'        : str(roi.getName()) if roi.getName() else f'roi_{index}',
            'type'        : ROI_TYPES.get(roi_type, f'unknown_{roi_type}'),
            'type_int'    : roi_type,
            'z'           : int(roi.getZPosition()),
            'c'           : int(roi.getCPosition()),
            't'           : int(roi.getTPosition()),
            'position'    : int(roi.getPosition()),
            'bounds'      : {
                'x': int(bounds.x), 'y': int(bounds.y),
                'w': int(bounds.width), 'h': int(bounds.height),
            },
            'stroke_color': color_to_hex(stroke_c),
            'fill_color'  : color_to_hex(fill_c),
            'stroke_width': float(roi.getStrokeWidth()),
            'coords'      : get_roi_coords(roi),
            'group'       : int(roi.getGroup()) if hasattr(roi, 'getGroup') else None,
        }

    result = {
        'single_roi' : None,
        'overlay'    : [],
        'roi_manager': [],
    }

    single = imp.getRoi()
    if single is not None:
        result['single_roi'] = extract_roi_info(single, 0)

    overlay = imp.getOverlay()
    if overlay is not None:
        n = overlay.size()
        print(f"Found {n} ROIs in overlay")
        for i in range(n):
            result['overlay'].append(extract_roi_info(overlay.get(i), i))
    else:
        # print("No overlay found")
        pass

    rm = RoiManager.getInstance()
    if rm is not None:
        rois = rm.getRoisAsArray()
        print(f"Found {len(rois)} ROIs in ROI manager")
        for i, roi in enumerate(rois):
            info = extract_roi_info(roi, i)
            info['name'] = str(rm.getName(i))
            result['roi_manager'].append(info)
    else:
        # print("ROI manager empty or not open")
        pass

    return result

def print_rois(imp):
    """
    Print a human-readable summary of all ROIs attached to an ImagePlus.

    Calls get_rois(imp) internally and prints one line per ROI showing
    index, name, type, hyperstack position, colours, stroke width, and bounds.

    Parameters
    ----------
    imp : ImagePlus
    """
    rois = get_rois(imp)

    def print_roi(r, source):
        print(f"  [{source}] #{r['index']} '{r['name']}' "
              f"type={r['type']} "
              f"z={r['z']} c={r['c']} t={r['t']} "
              f"stroke={r['stroke_color']} fill={r['fill_color']} "
              f"stroke_width={r['stroke_width']} "
              f"bounds={r['bounds']} "
              f"coords_keys={list(r['coords'].keys())}")

    if rois['single_roi']:
        print("Single ROI:")
        print_roi(rois['single_roi'], 'single')

    if rois['overlay']:
        print(f"Overlay ({len(rois['overlay'])} ROIs):")
        for r in rois['overlay']:
            print_roi(r, 'overlay')


    if rois['roi_manager']:
        print(f"ROI Manager ({len(rois['roi_manager'])} ROIs):")
        for r in rois['roi_manager']:
            print_roi(r, 'manager')


def roi_to_mask(roi, width, height):
    """
    Convert a raw Java ImageJ ROI to a boolean numpy mask.

    Parameters
    ----------
    roi    : Java ROI object  The 'roi' key from a roi_info dict,
                               e.g. rois['overlay'][0]['roi'].
    width  : int  Image width  (imp.getWidth()).
    height : int  Image height (imp.getHeight()).

    Returns
    -------
    np.ndarray, dtype=bool, shape=(height, width)
    True inside the ROI, False outside.
    """
    ByteProcessor = sj.jimport('ij.process.ByteProcessor')
    bp = ByteProcessor(width, height)
    bp.setColor(255)
    bp.fill(roi)
    mask = np.array(bp.getPixels(), dtype=np.uint8).reshape(height, width)
    return mask > 0



def _parse_series(filepath):
    """
    Parse a Bio-Formats file and return series metadata.

    Opens the file with ImportProcess to read the header without loading
    pixel data. Silences Bio-Formats logging before parsing.

    Parameters
    ----------
    filepath : str  Path to any Bio-Formats-supported file.

    Returns
    -------
    opts    : ImporterOptions  Configured with the file path.
    process : ImportProcess    Already executed; ready to query.
    n       : int              Number of series in the file.
    names   : list[str]        Human-readable label for each series.
    """
    with ij_lock:  # re-entrant: callers may already hold it
        silence_bioformats_logging()
        J = _java_imports()
        opts = J['ImporterOptions']()
        opts.setId(filepath)
        process = J['ImportProcess'](opts)
        process.execute()
        n = process.getSeriesCount()
        names = [process.getSeriesLabel(i) for i in range(n)]
    return opts, process, n, names

def _open_from_opts(opts, series_index, n):
    """
    Open a specific series from an already-configured ImporterOptions.

    Activates only the requested series, then opens it via ImagePlusReader.
    Falls back to IJ.openImage() for plain formats (TIF, PNG, ...) where
    ImagePlusReader returns an empty array.

    Parameters
    ----------
    opts         : ImporterOptions  From _parse_series().
    series_index : int              0-based series to open.
    n            : int              Total number of series in the file.

    Returns
    -------
    imp : ImagePlus
    """
    with ij_lock:  # re-entrant: callers may already hold it
        J = _java_imports()
        for i in range(n):
            opts.setSeriesOn(i, i == series_index)
        process2 = J['ImportProcess'](opts)
        process2.execute()
        reader = J['ImagePlusReader'](process2)
        imps = reader.openImagePlus()
        if imps is None or len(imps) == 0:
            return J['IJ'].openImage(opts.getId())
        return imps[0]


def count_series(filepath):
    """
    Return the number of series in a Bio-Formats file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    int
    """
    _, _, n, _ = _parse_series(filepath)
    return n


def list_series(filepath):
    """
    Return the list of series labels in a Bio-Formats file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    list[str]
    """
    _, _, _, names = _parse_series(filepath)
    return names


def open_series(filepath, series_index):
    """
    Open a specific series from a Bio-Formats file.

    Parameters
    ----------
    filepath     : str  Path to the image file.
    series_index : int  0-based index of the series to open.

    Returns
    -------
    imp : ImagePlus
    """
    opts, _, n, _ = _parse_series(filepath)
    return _open_from_opts(opts, series_index, n)


def list_and_open_series(filepath, series_index):
    """
    Print all series names, then open and return the selected one.

    Combines listing and opening in a single file-header parse,
    avoiding a second read of the file header.

    Parameters
    ----------
    filepath     : str  Path to the image file.
    series_index : int  0-based index of the series to open.

    Returns
    -------
    imp : ImagePlus
    """
    opts, _, n, names = _parse_series(filepath)
    if False:
        # for debug
        print(f"Found {n} series:")
        for i, name in enumerate(names):
            print(f"  [{i}] {name}")
    return _open_from_opts(opts, series_index, n)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def get_metadata(filepath, series_index=0):
    """
    Extract full OME metadata for a series using Bio-Formats.

    Reads physical pixel sizes, channel names, and objective magnification
    from the OME metadata store embedded in the file.

    Parameters
    ----------
    filepath     : str
    series_index : int  0-based series index (clamped to valid range).

    Returns
    -------
    dict with keys:
        name, series_index,
        size_x, size_y, size_z, size_c, size_t,
        bit_depth, is_rgb, is_interleaved,
        pixel_size_x, pixel_size_y, pixel_size_z, pixel_unit,
        channel_names, magnification
    """
    opts, _, n, names = _parse_series(filepath)
    series_index = max(0, min(series_index, n - 1))

    for i in range(n):
        opts.setSeriesOn(i, i == series_index)
    J = _java_imports()
    process = J['ImportProcess'](opts)
    process.execute()
    reader = process.getReader()
    reader.setSeries(series_index)
    meta = reader.getMetadataStore()

    try:
        phys_x = meta.getPixelsPhysicalSizeX(series_index)
        phys_y = meta.getPixelsPhysicalSizeY(series_index)
        phys_z = meta.getPixelsPhysicalSizeZ(series_index)
        pixel_size_x = phys_x.value().doubleValue() if phys_x else None
        pixel_size_y = phys_y.value().doubleValue() if phys_y else None
        pixel_size_z = phys_z.value().doubleValue() if phys_z else None
        pixel_unit   = str(phys_x.unit().getSymbol()) if phys_x else None
    except Exception:
        pixel_size_x = pixel_size_y = pixel_size_z = pixel_unit = None

    info = {
        'name'          : names[series_index],
        'series_index'  : series_index,
        'size_x'        : int(reader.getSizeX()),
        'size_y'        : int(reader.getSizeY()),
        'size_z'        : int(reader.getSizeZ()),
        'size_c'        : int(reader.getSizeC()),
        'size_t'        : int(reader.getSizeT()),
        'bit_depth'     : int(reader.getBitsPerPixel()),
        'is_rgb'        : bool(reader.isRGB()),
        'is_interleaved': bool(reader.isInterleaved()),
        'pixel_size_x'  : pixel_size_x,
        'pixel_size_y'  : pixel_size_y,
        'pixel_size_z'  : pixel_size_z,
        'pixel_unit'    : pixel_unit,
    }

    try:
        info['channel_names'] = [
            str(meta.getChannelName(series_index, c))
            for c in range(reader.getSizeC())
        ]
    except Exception:
        info['channel_names'] = None

    try:
        obj = meta.getObjectiveNominalMagnification(0, 0)
        info['magnification'] = float(str(obj)) if obj else None
    except Exception:
        info['magnification'] = None

    return info


def get_imp_metadata(imp):
    """
    Extract basic metadata directly from an already-opened ImagePlus.

    Faster than get_metadata() because it does not re-parse the file.
    Physical sizes come from the ImageJ Calibration object, which is
    populated by Bio-Formats when the image was opened.

    Parameters
    ----------
    imp : ImagePlus

    Returns
    -------
    dict with keys:
        size_x, size_y, size_z, size_c, size_t,
        bit_depth,
        pixel_size_x, pixel_size_y, pixel_size_z, pixel_unit
    """
    cal = imp.getCalibration()
    print(cal)
    return {
        'size_x'      : imp.getWidth(),
        'size_y'      : imp.getHeight(),
        'size_z'      : imp.getNSlices(),
        'size_c'      : imp.getNChannels(),
        'size_t'      : imp.getNFrames(),
        'bit_depth'   : imp.getBitDepth(),
        'pixel_size_x': cal.pixelWidth,
        'pixel_size_y': cal.pixelHeight,
        'pixel_size_z': cal.pixelDepth,
        'pixel_unit'  : str(cal.getUnit()),
    }

def print_metadata(filepath, series_index):
    """
    Print a human-readable metadata summary for one series.

    Parameters
    ----------
    filepath     : str
    series_index : int
    """
    info = get_metadata(filepath, series_index)
    print(f"Series [{info['series_index']}]: {info['name']}")
    print(f"  Dimensions : X={info['size_x']} Y={info['size_y']} "
          f"Z={info['size_z']} C={info['size_c']} T={info['size_t']}")
    print(f"  Bit depth  : {info['bit_depth']}")
    print(f"  Pixel size : X={info['pixel_size_x']} Y={info['pixel_size_y']} "
          f"Z={info['pixel_size_z']} {info['pixel_unit']}")
    print(f"  Magnification: {info['magnification']}x")
    print(f"  Channels   : {info['channel_names']}")
    print(f"  RGB        : {info['is_rgb']}")
    print(info)


def to_metadata(imp, luts=None, filepath=None, series_index=0):
    """
    Build a standard metadata dict from an open ImagePlus.

    Dimensions are reported in tdhwc order with size-1 axes dropped,
    so a single-channel Z-stack gives dimensions=(d, h, w) with
    dim_order=('d', 'h', 'w') rather than (t, d, h, w, c).

    Parameters
    ----------
    imp          : ImagePlus
    luts         : dict, optional   Output of get_luts(imp).
    filepath     : str, optional    When provided, the acquisition date is
                                    read from the OME metadata store.
    series_index : int              Series index used when filepath is given.

    Returns
    -------
    dict with keys:
        dimensions    : tuple  e.g. (13, 2048, 2048) for a Z-stack
        dim_order     : tuple  e.g. ('d', 'h', 'w')
        bits          : int
        vx, vy, vz    : float  physical voxel sizes
        AR            : float  vz/vx aspect ratio
        LUTs          : dict or None  from get_luts()
        cur_d         : int  current Z position (0-indexed)
        cur_t         : int  current T position (0-indexed)
        Overlays      : list  of roi_info dicts from get_rois()
        ROI           : roi_info dict or None
        timelapse     : float or None  frame interval
        creation_time : str or None
        pixel_unit    : str  e.g. 'µm'
        time_unit     : str  e.g. 'sec'
    """
    cal = imp.getCalibration()

    t = imp.getNFrames()
    d = imp.getNSlices()
    h = imp.getHeight()
    w = imp.getWidth()
    c = imp.getNChannels()

    dim_map = {}
    if t > 1: dim_map['t'] = t
    if d > 1: dim_map['d'] = d
    dim_map['h'] = h
    dim_map['w'] = w
    if c > 1: dim_map['c'] = c

    dimensions = tuple(dim_map.values())
    dim_order  = tuple(dim_map.keys())

    vx = float(cal.pixelWidth)
    vy = float(cal.pixelHeight)
    vz = float(cal.pixelDepth)
    ar = vz / vx if vx > 0 else None

    creation_time = None
    if filepath is not None:
        try:
            opts, _, n, _ = _parse_series(filepath)
            for i in range(n):
                opts.setSeriesOn(i, i == series_index)
            J = _java_imports()
            process = J['ImportProcess'](opts)
            process.execute()
            ct = process.getReader().getMetadataStore().getImageAcquisitionDate(series_index)
            creation_time = str(ct) if ct is not None else None
        except Exception:
            creation_time = None

    roi_data = get_rois(imp)

    return {
        'dimensions'   : dimensions,
        'dim_order'    : dim_order,
        'bits'         : imp.getBitDepth(),
        'vx'           : vx,
        'vy'           : vy,
        'vz'           : vz,
        'AR'           : ar,
        'LUTs'         : luts,
        'cur_d'        : imp.getZ() - 1,
        'cur_t'        : imp.getT() - 1,
        'Overlays'     : roi_data['overlay'],
        'ROI'          : roi_data['single_roi'],
        'timelapse'    : float(cal.frameInterval) if cal.frameInterval > 0 else None,
        'creation_time': creation_time,
        'pixel_unit'   : str(cal.getUnit()),
        'time_unit'    : str(cal.getTimeUnit()),
    }


def print_metadata_std(meta):
    """
    Pretty-print a standard metadata dict produced by to_metadata().

    Parameters
    ----------
    meta : dict  Output of to_metadata().
    """
    dims = {k: v for k, v in zip(meta['dim_order'], meta['dimensions'])}
    print(f"Dimensions ({' '.join(meta['dim_order'])}): "
          + ' '.join(f"{k}={v}" for k, v in dims.items()))
    print(f"Bits       : {meta['bits']}")
    print(f"Voxel size : vx={meta['vx']} vy={meta['vy']} vz={meta['vz']} "
          f"[{meta['pixel_unit']}]  AR={meta['AR']:.4f}")
    print(f"Timelapse  : {meta['timelapse']} {meta['time_unit']}")
    print(f"cur_d={meta['cur_d']}  cur_t={meta['cur_t']}")
    print(f"LUTs       : {None if meta['LUTs'] is None else {k: v.shape for k, v in enumerate(meta['LUTs'])}}")
    print(f"Overlays   : {len(meta['Overlays'])} ROIs")
    print(f"ROI        : {meta['ROI']}")
    print(f"Created    : {meta['creation_time']}")


# ---------------------------------------------------------------------------
# Public API — LUTs
# ---------------------------------------------------------------------------

def get_luts(imp):
    """
    Extract per-channel LUTs from an ImagePlus.

    For CompositeImage, fetches each channel's LUT via getChannelLut(c),
    which correctly returns the per-channel LUT regardless of which channel
    is currently active. For a plain ImagePlus, reads getLuts()[0].

    LUT colours are unpacked from packed ARGB integers returned by
    IndexColorModel.getRGBs() to avoid JPype overload ambiguity with
    the inherited ColorModel.getRed(int) / ColorModel.getRed(Object) pair.

    Parameters
    ----------
    imp : ImagePlus

    Returns
    -------
    dict : {channel (1-indexed): np.ndarray shape (256, 3) uint8}
           Each row is [R, G, B] for that LUT index value.
    """
    J = _java_imports()

    def lut_to_array(raw_lut):
        cm = J['IndexColorModel'] @ raw_lut
        rgb_array = jpype.JArray(jpype.JInt)(256)
        cm.getRGBs(rgb_array)
        rgb = np.array([int(x) for x in rgb_array], dtype=np.int32)
        r = ((rgb >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb >>  8) & 0xFF).astype(np.uint8)
        b = ((rgb      ) & 0xFF).astype(np.uint8)
        return np.stack([r, g, b], axis=1)

    luts = {}
    if isinstance(imp, J['CompositeImage']):
        for c in range(1, imp.getNChannels() + 1):
            luts[c] = lut_to_array(imp.getChannelLut(c))
    else:
        # print('luts2')
        luts[1] = lut_to_array(imp.getLuts()[0])
        # print('luts',luts)

    luts = list(luts.values())
    return luts

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    if True:
        from batoolset.IJ.ij_reader import reset_fiji_cache

        reset_fiji_cache()
        sys.exit(0)



    ij = init_ij()

    FILE = './260327_better_dissections_adults_MISC.lif'
    FILE = './260304_test_muscle_clones1.lif'
    FILE = './260304_test_muscle_clones1.lif - FRT19RFPnls_hsLP_FRT19_H2AGFP_hs_eggs_L1_Bcg_3b_40X.tif'
    FILE = './Image16.lsm'
    FILE = './armGFP_2_manue_10percent_laser_speed6.lsm'
    FILE = '/home/aigouy/Bureau/sample_images_FIJI/2chZT.lsm'
    FILE = '/home/aigouy/Bureau/sample_images_FIJI/150707_WTstack.lsm'  # all seems ok including channels and etc... --> all very good --> I can finalize pyFigures rapidly then and maybe publish it in the mean time ???
    FILE = '/home/aigouy/Bureau/sample_images_FIJI/tst_ROI_in_someZ_positions.tif'  # all seems ok including channels and etc... --> all very good --> I can finalize pyFigures rapidly then and maybe publish it in the mean time ???
    FILE = '/home/aigouy/Bureau/sample_images_quickfigure/all chan in order.zvi'  # all seems ok including channels and etc... --> all very good --> I can finalize pyFigures rapidly then and maybe publish it in the mean time ???
    # FILE = '/home/aigouy/Bureau/sample_images_FIJI/particles.gif'  # all seems ok including channels and etc... --> all very good --> I can finalize pyFigures rapidly then and maybe publish it in the mean time ???

    INDEX = 16 # 2


    print('='*20)
    print('count', count_series(FILE))
    print('='*20)

    # --- Main ---
    imp = list_and_open_series(FILE, series_index=INDEX)
    luts = get_luts(imp)
    # lust = get_luts_simple(imp)

    image = ij.py.from_java(imp)
    print("Shape:", image.shape) # --> this is ok, but maybe try also with different LUTs and so on... --> TODO
    print("LUTs per channel:", {k: v.shape for k, v in enumerate(luts)})

    # Show the first channel's LUT color at max intensity
    for ch, lut_array in enumerate(luts):
        print(f"  Channel {ch} max color (RGB): {lut_array[255]}")


    # --- Usage ---
    # imp = list_and_open_series(FILE, series_index=INDEX)
    # luts = get_luts(imp)
    # show_image(imp, luts) # if image has 2 channels then it does not work

    print_metadata(FILE, series_index=INDEX)
    info = get_metadata(FILE, series_index=INDEX)
    print(info['pixel_size_x'])  # e.g. 0.2756 µm


    # Usage
    meta = to_metadata(imp, luts=luts, filepath=FILE, series_index=INDEX)
    print_metadata_std(meta)

    # Access dimensions easily
    # t, d, h, w, c = meta['dimensions']
    print('dimensions',meta['dimensions'])

    print('*'*20)
    print(get_imp_metadata(imp))

    # franchement c'est un très bon début et me fait gagner plein de temps --> mais peut etre finaliser --> all seems ok


    # very good I should probably remove the crappy java hub stuff because it is very hard to use and complex and rely on that and it may not do errors, I need open big amounts of files just to see what it does
    # see also how it goes with the memory


    # all seems to be fine, let's try several other images and see what to do with that ???



    # also hack this code to do extraction of tif files and or max projs for a given file --> TODO
    # see what is the best way to integrate this in the rest of my code
    # also need to hack the dimensions so that they fit my stuff and see how to do that with the image of the other code --> TODO and maybe also offer the possibility to change image ??? --> think what is the best way to do that
    # gather all my images and get this




    # now ROIs
    # Usage
    # Usage
    rois = get_rois(imp)
    w, h = imp.getWidth(), imp.getHeight()
    for r in rois['overlay']:
        # Correct — pass the raw Java ROI
        mask = roi_to_mask(r['roi'], w, h)
        print(f"ROI '{r['name']}' mask — True pixels: {mask.sum()}")


    if False:
        # Usage — extract the Java ROI from the overlay directly
        overlay = imp.getOverlay()
        if overlay is not None:
            w, h = imp.getWidth(), imp.getHeight()
            for i in range(overlay.size()):
                roi  = overlay.get(i)       # raw Java ROI
                mask = roi_to_mask(roi, w, h)
                print(f"ROI {i} mask — True pixels: {mask.sum()}")

    # I can read the ROIs so I could convert them to pyfigures format so that they can be represented on the image --> then see if it perfectly fits with imageJ or not


    # this is all good now

    # it all seems good