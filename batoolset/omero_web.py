"""Images from an OMERO server, through OMERO.web (no omero-py / Ice needed).

    meta, arr = read_omero('https://omero.example.org/webclient/img_detail/1234/')
    meta, arr = read_omero('https://omero.example.org/webclient/?show=image-1234#z=0-9&t=0')

Any OMERO.web image link works: the webclient (``img_detail/ID``,
``?show=image-ID``), OMERO.iviewer (``iviewer/?images=ID``) or webgateway
(``imgData/ID``) URL of the image, i.e. what the browser shows when the image
is open. The server address is everything before ``/webclient``,
``/webgateway``, ``/iviewer`` or ``/api``.

How the pixels are read: OMERO.web serves rendered planes, not raw ones. Each
channel of each plane is rendered in greyscale over the channel's full data
range (``window.min`` to ``window.max`` of ``imgData``) and mapped back to that
range, so values keep their scale with 256 levels per channel (exact for 8-bit
images; a 16-bit image is quantised to 256 levels of its own range, which is
plenty for display in a figure but not for measurements — use the original
file or OME-Zarr for those). Pixel sizes, channel names, colours and display
windows come from OMERO.

Raw pixels (optional): with omero-py installed (``pip install omero-py``;
needs zeroc-ice) and a login, the planes are read raw, in the image's own data
type (16-bit, float...), through OMERO's Ice API — on the web host's port 4064,
OMERO_HOST / OMERO_PORT if set, or the server's websocket endpoint
(wss://host/omero-ws). Anything that fails there falls back to OMERO.web.
Public data without a login is always read through OMERO.web.

Planes: by default every Z and T plane is read if the image fits a fixed budget
(512 MB, 8192 px per side, 2000 rendered planes); otherwise choose with a URL
fragment, e.g. ``#z=5``, ``#z=0-9&t=0``, ``#t=0,10,20&c=0,2``.

Private data: if the server asks for a login, :class:`OmeroLoginRequired` is
raised. Log in with :func:`login` (the GUI asks for the user name and password;
scripts can set OMERO_USERNAME / OMERO_PASSWORD in the environment). The session
lives in memory only: credentials are never written to a figure or a file.
"""
from __future__ import annotations

import atexit
import io
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlsplit

import numpy as np

DEFAULT_MAX_BYTES = 512 * 1024 ** 2
DEFAULT_MAX_SIDE = 8192
MAX_PLANES = 2000
_PATTERNS = (
    r"/(?:webclient|webgateway)/img_detail/(\d+)",
    r"/webgateway/imgData/(\d+)",
    r"/webgateway/render_image(?:_region)?/(\d+)",
    r"/api/v0/m/images/(\d+)",
)
_APP_SEGMENTS = ("/webclient", "/webgateway", "/iviewer", "/api/")
_DTYPES = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16,
           "int32": np.int32, "uint32": np.uint32, "float": np.float32, "double": np.float64,
           "bit": np.uint8}
_SESSIONS = {}          # server base URL -> logged-in requests.Session (memory only)
_CREDENTIALS = {}       # server base URL -> (user, password), memory only, for omero-py
_RAW_CONNS = {}         # server base URL -> connected omero-py BlitzGateway
_RAW_LOCK = threading.Lock()


class OmeroLoginRequired(PermissionError):
    """The server needs a login before this image can be read."""

    def __init__(self, base):
        super().__init__(f"the OMERO server {base} asks for a login")
        self.base = base


# ---------------------------------------------------------------------------
# addresses
# ---------------------------------------------------------------------------
def parse_omero_url(url):
    """(server base URL, image id) for an OMERO.web image link, or None."""
    if not isinstance(url, str) or not re.match(r"^https?://", url.strip(), re.I):
        return None
    clean = url.strip().split("#", 1)[0]
    parts = urlsplit(clean)
    path = parts.path
    image_id = None
    for pat in _PATTERNS:
        m = re.search(pat, path)
        if m:
            image_id = int(m.group(1))
            break
    if image_id is None:
        q = parse_qs(parts.query)
        for show in q.get("show", []):
            m = re.search(r"\bimage-(\d+)", show)
            if m:
                image_id = int(m.group(1))
                break
        if image_id is None and "/iviewer" in path:
            ids = (q.get("images") or q.get("image") or [""])[0]
            m = re.match(r"(\d+)", ids)
            if m:
                image_id = int(m.group(1))
    if image_id is None:
        return None
    cut = len(path)
    for seg in _APP_SEGMENTS:
        i = path.find(seg)
        if i != -1:
            cut = min(cut, i)
    base = f"{parts.scheme}://{parts.netloc}{path[:cut]}".rstrip("/")
    return base, image_id


def is_omero_url(path) -> bool:
    return parse_omero_url(path) is not None


def canonical_url(base, image_id):
    return f"{base}/webclient/img_detail/{int(image_id)}/"


def _selection(url):
    """{'z': [...], 't': [...], 'c': [...]} from a '#z=0-4&t=0&c=0,2' fragment."""
    if "#" not in (url or ""):
        return {}
    out = {}
    for key, value in parse_qs(url.split("#", 1)[1]).items():
        if key not in ("z", "t", "c"):
            continue
        idx = []
        for part in value[0].split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                idx.extend(range(int(a), int(b) + 1))
            else:
                idx.append(int(part))
        out[key] = idx
    return out


# ---------------------------------------------------------------------------
# sessions and login
# ---------------------------------------------------------------------------
def _new_session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "batoolset-omero-web"
    return s


def session_for(base):
    s = _SESSIONS.get(base)
    if s is None:
        s = _new_session()
        _SESSIONS[base] = s
        user = os.environ.get("OMERO_USERNAME") or os.environ.get("OMERO_USER")
        password = os.environ.get("OMERO_PASSWORD")
        if user and password:
            try:
                login(base, user, password)
            except Exception:
                pass
    return _SESSIONS[base]


def login(base, username, password, server_id=None):
    """Log in to OMERO.web's JSON API; the session is kept in memory for *base*.
    Raises PermissionError if the server refuses the credentials."""
    s = _new_session()
    r = s.get(f"{base}/api/v0/token/", timeout=60)
    r.raise_for_status()
    token = r.json().get("data")
    if server_id is None:
        r = s.get(f"{base}/api/v0/servers/", timeout=60)
        r.raise_for_status()
        servers = r.json().get("data") or [{"id": 1}]
        server_id = servers[0].get("id", 1)
    r = s.post(f"{base}/api/v0/login/", timeout=60,
               data={"username": username, "password": password, "server": server_id},
               headers={"X-CSRFToken": token or "", "Referer": f"{base}/"})
    ok = r.status_code == 200 and r.json().get("success", False)
    if not ok:
        try:
            msg = r.json().get("message")
        except ValueError:
            msg = None
        raise PermissionError(msg or f"OMERO login refused ({r.status_code})")
    _SESSIONS[base] = s
    _CREDENTIALS[base] = (username, password)      # memory only: for omero-py raw reads
    return True


def logout(base):
    _SESSIONS.pop(base, None)
    _CREDENTIALS.pop(base, None)
    conn = _RAW_CONNS.pop(base, None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# raw pixels with omero-py (optional)
# ---------------------------------------------------------------------------
def raw_reader_installed() -> bool:
    try:
        import omero.gateway  # noqa: F401
        return True
    except Exception:
        return False


def _raw_hosts(base):
    """(host, port) candidates for the OMERO server behind OMERO.web *base*:
    OMERO_HOST/OMERO_PORT if set, the web host on the Ice port 4064, then the
    websocket endpoint OMERO.web servers often expose (wss://host/omero-ws)."""
    out = []
    if os.environ.get("OMERO_HOST"):
        out.append((os.environ["OMERO_HOST"], int(os.environ.get("OMERO_PORT") or 4064)))
    parts = urlsplit(base)
    out.append((parts.hostname, 4064))
    out.append((f"wss://{parts.netloc}/omero-ws", 443))
    return out


def _credentials(base):
    creds = _CREDENTIALS.get(base)
    if creds:
        return creds
    user = os.environ.get("OMERO_USERNAME") or os.environ.get("OMERO_USER")
    password = os.environ.get("OMERO_PASSWORD")
    return (user, password) if user and password else None


def _close_raw_connections():
    """At exit: close the Ice sessions (else Ice warns 'communicator not
    destroyed during global destruction' and the server keeps the session)."""
    for base in list(_RAW_CONNS):
        conn = _RAW_CONNS.pop(base, None)
        try:
            conn.close()
        except Exception:
            pass


atexit.register(_close_raw_connections)


def raw_connection(base):
    """A connected omero-py BlitzGateway for *base*, or None (omero-py not
    installed, not logged in, or the server unreachable through Ice)."""
    if not raw_reader_installed():
        return None
    with _RAW_LOCK:
        conn = _RAW_CONNS.get(base)
        if conn is not None:
            try:
                if conn.isConnected() and conn.keepAlive():
                    return conn
            except Exception:
                pass
            _RAW_CONNS.pop(base, None)
        creds = _credentials(base)
        if not creds:
            return None
        from omero.gateway import BlitzGateway
        for host, port in _raw_hosts(base):
            try:
                conn = BlitzGateway(creds[0], creds[1], host=host, port=port, secure=True)
                if conn.connect():
                    _RAW_CONNS[base] = conn
                    return conn
            except Exception:
                continue
    return None


def _read_raw(conn, image_id, zs, ts, cs, h, w, dtype):
    """Raw planes -> array (t, z, h, w, c) in the image's own data type."""
    with _RAW_LOCK:
        image = conn.getObject("Image", image_id)
        if image is None:
            raise FileNotFoundError(f"image {image_id} not found (or not visible to this user)")
        pixels = image.getPrimaryPixels()
        zct = [(z, c, t) for t in ts for z in zs for c in cs]
        planes = list(pixels.getPlanes(zct))
    arr = np.zeros((len(ts), len(zs), h, w, len(cs)), dtype=dtype)
    k = 0
    for ti in range(len(ts)):
        for zi in range(len(zs)):
            for ci in range(len(cs)):
                arr[ti, zi, :, :, ci] = planes[k]
                k += 1
    return arr


def _get(base, url, **params):
    s = session_for(base)
    r = s.get(url, params=params or None, timeout=120, allow_redirects=False)
    if r.status_code in (301, 302, 303, 401, 403):    # sent to the login page / refused
        raise OmeroLoginRequired(base)
    if r.status_code == 404:
        raise FileNotFoundError(f"not found on the OMERO server: {url}")
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
def image_data(url):
    """OMERO.web's imgData for the image (sizes, pixel sizes, channels...)."""
    parsed = parse_omero_url(url)
    if parsed is None:
        raise ValueError(f"not an OMERO image link: {url}")
    base, image_id = parsed
    r = _get(base, f"{base}/webgateway/imgData/{image_id}/")
    try:
        data = r.json()
    except ValueError:
        raise OmeroLoginRequired(base) from None
    if "size" not in data:
        raise IOError(f"OMERO could not read image {image_id}: "
                      f"{data.get('Exception') or data.get('message') or 'no pixel data'}")
    return data


def selection_problem(url, size) -> str | None:
    """What is wrong with the '#z=..&t=..&c=..' fragment of *url* for an
    image of *size* ({'z': n, 't': n, 'c': n}), or None."""
    if "#" not in (url or ""):
        return None
    frag = url.split("#", 1)[1]
    for key, values in parse_qs(frag, keep_blank_values=True).items():
        if key not in ("z", "t", "c"):
            return f"unknown plane selector {key!r} after '#': use z, t and c (e.g. #z=0-4&t=0&c=0,2)"
        n = int(size[key])
        for part in values[0].split(","):
            part = part.strip()
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
            if not m:
                return f"{key}={values[0]!r}: use numbers and ranges, e.g. {key}=0-4 or {key}=0,2"
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) is not None else a
            if b < a:
                return f"{key}={part}: a range goes from low to high ({key}={b}-{a})"
            if b >= n:
                return f"{key}={part} out of range: the image has {n} {key.upper()} plane(s), 0-{n - 1}"
    return None


def describe(url) -> dict:
    """What read_omero would read, from the metadata only (no pixels).
    ValueError for a plane selection the image does not have."""
    base, image_id = parse_omero_url(url)
    d = image_data(url)
    size = d["size"]
    problem = selection_problem(url, size)
    if problem:
        raise ValueError(problem)
    ps = d.get("pixel_size") or {}
    sel = _selection(url)
    planes = {k: sel.get(k, list(range(int(size[k])))) for k in ("z", "t", "c")}
    return {"server": base, "image_id": image_id, "url": canonical_url(base, image_id),
            "name": (d.get("meta") or {}).get("imageName"),
            "pixels_type": (d.get("meta") or {}).get("pixelsType"),
            "size": {k: int(size[k]) for k in ("width", "height", "z", "t", "c")},
            "pixel_size_um": {k: ps.get(k) for k in ("x", "y", "z")},
            "channels": [{"label": c.get("label"), "color": c.get("color"),
                          "window": c.get("window")} for c in d.get("channels", [])],
            "selected": planes,
            "rendered_planes": len(planes["z"]) * len(planes["t"]) * len(planes["c"]),
            "raw_reader": ("omero-py (raw pixels)" if raw_reader_installed() and _credentials(base)
                           else "omero-py installed; raw pixels after a login" if raw_reader_installed()
                           else "OMERO.web (256 levels per channel; pip install omero-py for raw pixels)")}


def size_of(url):
    """(width, height) of the image, from the metadata."""
    size = image_data(url)["size"]
    return int(size["width"]), int(size["height"])


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------
def _render_plane(base, image_id, z, t, c, lo, hi, w, h):
    """One channel of one plane, greyscale over [lo, hi], as uint8 (h, w)."""
    from PIL import Image
    spec = f"{c + 1}|{lo}:{hi}$FFFFFF"
    params = {"c": spec, "m": "g", "format": "png", "p": "normal", "ia": "0"}
    if w * h > 3000 * 3000:
        # big planes: OMERO.web renders large images in regions
        out = np.zeros((h, w), np.uint8)
        step = 2048
        for y in range(0, h, step):
            for x in range(0, w, step):
                rw, rh = min(step, w - x), min(step, h - y)
                r = _get(base, f"{base}/webgateway/render_image_region/{image_id}/{z}/{t}/",
                         region=f"{x},{y},{rw},{rh}", **params)
                tile = np.asarray(Image.open(io.BytesIO(r.content)).convert("L"))
                out[y:y + tile.shape[0], x:x + tile.shape[1]] = tile[:rh, :rw]
        return out
    r = _get(base, f"{base}/webgateway/render_image/{image_id}/{z}/{t}/", **params)
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("L"))


def read_omero(url, max_bytes=DEFAULT_MAX_BYTES, max_side=DEFAULT_MAX_SIDE, workers=8, raw=True):
    """Read an OMERO image -> (metadata, array in batoolset order t d h w c).

    With omero-py installed and a login (GUI prompt, login(), or
    OMERO_USERNAME / OMERO_PASSWORD), the raw pixels are read in their own data
    type; otherwise (or with raw=False) OMERO.web renders 256 levels per channel."""
    base, image_id = parse_omero_url(url)
    d = image_data(url)
    size = d["size"]
    w, h = int(size["width"]), int(size["height"])
    sel = _selection(url)
    zs = sel.get("z", list(range(int(size["z"]))))
    ts = sel.get("t", list(range(int(size["t"]))))
    cs = sel.get("c", list(range(int(size["c"]))))
    for key, idx, n in (("z", zs, size["z"]), ("t", ts, size["t"]), ("c", cs, size["c"])):
        bad = [i for i in idx if not 0 <= i < int(n)]
        if bad or not idx:
            raise ValueError(f"{key} must be within 0-{int(n) - 1} (got {idx})")
    pixels_type = (d.get("meta") or {}).get("pixelsType", "uint8")
    dtype = np.dtype(_DTYPES.get(pixels_type, np.float32))
    n_planes = len(zs) * len(ts) * len(cs)
    n_bytes = w * h * n_planes * dtype.itemsize
    if max(w, h) > max_side:
        raise ValueError(f"image {image_id} is {w} x {h} px, larger than {max_side} px: "
                         "export it from OMERO as OME-Zarr (multi-resolution) to use it in a figure")
    if n_bytes > max_bytes or n_planes > MAX_PLANES:
        raise ValueError(f"image {image_id} has {size['z']} Z x {size['t']} T x {size['c']} C planes "
                         f"of {w} x {h} px ({n_bytes / 1024 ** 2:.0f} MB, {n_planes} renders): choose "
                         "planes with a fragment, e.g. #z=0-9&t=0 or #t=0")
    chans = d.get("channels", [])
    windows = []
    for c in cs:
        win = (chans[c].get("window") if c < len(chans) else None) or {}
        lo = win.get("min", 0)
        hi = win.get("max", 255)
        if hi <= lo:
            hi = lo + 1
        windows.append((lo, hi))

    session_for(base)                     # log in from the environment once, before threads
    arr = None
    reader = "omero.web"
    if raw:
        # raw pixels in the image's own type (16-bit, float...) with omero-py,
        # when installed and logged in; else OMERO.web renders below
        conn = raw_connection(base)
        if conn is not None:
            try:
                arr = _read_raw(conn, image_id, zs, ts, cs, h, w, dtype)
                reader = "omero-py"
            except FileNotFoundError:
                raise
            except Exception:
                arr = None
    if arr is None:
        jobs = [(ti, zi, ci, t, z, c) for ti, t in enumerate(ts) for zi, z in enumerate(zs)
                for ci, c in enumerate(cs)]
        arr = np.zeros((len(ts), len(zs), h, w, len(cs)), dtype=np.float64)

        def fetch(job):
            ti, zi, ci, t, z, c = job
            lo, hi = windows[ci]
            plane = _render_plane(base, image_id, z, t, c, lo, hi, w, h).astype(np.float64)
            arr[ti, zi, :plane.shape[0], :plane.shape[1], ci] = lo + plane * ((hi - lo) / 255.0)

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            list(pool.map(fetch, jobs))
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            arr = np.clip(np.round(arr), info.min, info.max)
        arr = arr.astype(dtype)

    letters = "tdhwc"
    squeeze = tuple(i for i, L in enumerate(letters) if L in "tdc" and arr.shape[i] == 1)
    arr = np.squeeze(arr, axis=squeeze) if squeeze else arr
    dims = "".join(L for i, L in enumerate(letters) if i not in squeeze)
    shape = dict(zip(dims, arr.shape))
    ps = d.get("pixel_size") or {}
    names = [chans[c].get("label") if c < len(chans) else None for c in cs]
    colors = [chans[c].get("color") if c < len(chans) else None for c in cs]
    meta = {"w": shape.get("w"), "h": shape.get("h"), "d": shape.get("d"), "t": shape.get("t"),
            "c": shape.get("c"), "bits": dtype.itemsize * 8, "dimensions": dims,
            "vx": ps.get("x"), "vy": ps.get("y"), "vz": ps.get("z"),
            "AR": None, "LUTs": None, "times": None, "Overlays": None, "ROI": None,
            "timelapse": None, "creation_time": None,
            "channel_names": names if any(names) else None,
            "channel_colors": colors if all(colors) else None,
            "channel_windows": [((chans[c].get("window") or {}).get("start"),
                                 (chans[c].get("window") or {}).get("end"))
                                if c < len(chans) else None for c in cs],
            "omero": {"server": base, "image_id": image_id, "name": (d.get("meta") or {}).get("imageName"),
                      "z": zs, "t": ts, "c": cs, "pixels_type": pixels_type, "reader": reader,
                      "quantised_to_8_bits": reader != "omero-py" and
                      (dtype.itemsize > 1 or not np.issubdtype(dtype, np.integer))}}
    if meta["vx"] and meta["vz"]:
        meta["AR"] = meta["vz"] / meta["vx"]
    if shape.get("c") and meta["channel_colors"]:
        from batoolset.ome_zarr import color_luts
        luts = color_luts(meta["channel_colors"], shape.get("c"))
        if luts is not None:
            meta["LUTs"] = luts
    return meta, arr
