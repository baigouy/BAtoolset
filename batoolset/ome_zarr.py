"""OME-Zarr (OME-NGFF) reading, local or from an http(s) URL.

    meta, arr = read_ome_zarr('https://.../image.zarr')          # auto level
    meta, arr = read_ome_zarr('/data/image.ome.zarr', level=1)
    meta, arr = read_ome_zarr('https://.../image.zarr#level=2')  # level in the URL

The image is returned as a numpy array in batoolset's dimension order
(t, d, h, w, c — absent axes dropped) with the usual metadata keys (w, h, d, t,
c, bits, vx, vy, vz in µm, dimensions) plus channel names / colours / display
windows from the ``omero`` block and an ``ome_zarr`` entry describing the
pyramid level that was read.

OME-Zarr images are pyramids: by default the largest level that fits a memory
budget is read (a 100 000 px slide becomes the level a figure can use), unless
a level is given.

Remote images are read through a small read-only zarr store built on
``requests`` (one HTTP GET per chunk, byte ranges for sharded v0.5 data) rather
than fsspec/aiohttp: requests uses certifi, so it works on Windows machines
where a broken certificate in the system store makes aiohttp fail at import.

Needs the ``zarr`` package (v3 reads both Zarr v2 / OME-NGFF 0.4 and Zarr v3 /
OME-NGFF 0.5; zarr 2 reads 0.4 only).
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import numpy as np

_UNIT_TO_UM = {"micrometer": 1.0, "micron": 1.0, "um": 1.0, "µm": 1.0,
               "nanometer": 1e-3, "nm": 1e-3, "millimeter": 1e3, "mm": 1e3,
               "centimeter": 1e4, "meter": 1e6, "angstrom": 1e-4}
DEFAULT_MAX_BYTES = 512 * 1024 ** 2
DEFAULT_MAX_SIDE = 8192


def is_zarr_path(path) -> bool:
    """True for a local or remote path that names a .zarr store."""
    if not isinstance(path, str):
        return False
    p = _strip_fragment(path).rstrip("/\\").lower()
    return p.endswith(".zarr")


def is_url(path) -> bool:
    return isinstance(path, str) and re.match(r"^https?://", path, re.I) is not None


def _strip_fragment(path):
    return path.split("#", 1)[0]


def _level_from_fragment(path):
    m = re.search(r"#.*\blevel=(\d+)", path or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# read-only HTTP store (zarr 3 / zarr 2)
# ---------------------------------------------------------------------------
def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "batoolset-ome-zarr"
    return s


def _http_get(session, url, headers=None):
    r = session.get(url, headers=headers or {}, timeout=60)
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    return r.content


def _make_http_store(url):
    """A read-only zarr store over http(s) for the installed zarr version."""
    import zarr
    base = _strip_fragment(url).rstrip("/") + "/"
    session = _session()
    if int(zarr.__version__.split(".")[0]) < 3:
        from collections.abc import Mapping

        class _HTTPMap(Mapping):                      # zarr 2: a mapping of keys -> bytes
            def __getitem__(self, key):
                data = _http_get(session, base + key)
                if data is None:
                    raise KeyError(key)
                return data

            def __contains__(self, key):
                try:
                    self[key]
                    return True
                except KeyError:
                    return False

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0
        return _HTTPMap()

    from zarr.abc.store import (Store, RangeByteRequest, OffsetByteRequest, SuffixByteRequest)
    from zarr.core.buffer import default_buffer_prototype

    class _HTTPStore(Store):
        supports_writes = False
        supports_deletes = False
        supports_partial_writes = False
        supports_listing = False

        def __init__(self):
            super().__init__(read_only=True)

        def __eq__(self, other):
            return isinstance(other, _HTTPStore) and other is self

        def _fetch(self, key, byte_range):
            headers = {}
            if isinstance(byte_range, RangeByteRequest):
                headers["Range"] = f"bytes={byte_range.start}-{byte_range.end - 1}"
            elif isinstance(byte_range, OffsetByteRequest):
                headers["Range"] = f"bytes={byte_range.offset}-"
            elif isinstance(byte_range, SuffixByteRequest):
                headers["Range"] = f"bytes=-{byte_range.suffix}"
            return _http_get(session, base + key, headers)

        async def get(self, key, prototype=None, byte_range=None):
            prototype = prototype or default_buffer_prototype()
            data = await asyncio.to_thread(self._fetch, key, byte_range)
            return None if data is None else prototype.buffer.from_bytes(data)

        async def get_partial_values(self, prototype, key_ranges):
            return [await self.get(k, prototype, r) for k, r in key_ranges]

        async def exists(self, key):
            return (await asyncio.to_thread(self._fetch, key, None)) is not None

        async def set(self, key, value):
            raise PermissionError("read-only store")

        async def delete(self, key):
            raise PermissionError("read-only store")

        async def list(self):
            return
            yield

        async def list_prefix(self, prefix):
            return
            yield

        async def list_dir(self, prefix):
            return
            yield

        def __repr__(self):
            return f"HTTPStore({base})"

    return _HTTPStore()


def open_group(path):
    """Open the zarr group at *path* (local path or http(s) URL), read-only."""
    import zarr
    if is_url(path):
        store = _make_http_store(path)
        try:
            return zarr.open_group(store=store, mode="r")
        except Exception:
            # zarr 3 needs the format when it cannot list: try v2 (OME-NGFF 0.4) then v3
            for fmt in (2, 3):
                try:
                    return zarr.open_group(store=store, mode="r", zarr_format=fmt)
                except Exception:
                    pass
            raise
    return zarr.open_group(_strip_fragment(path), mode="r")


# ---------------------------------------------------------------------------
# OME-NGFF metadata
# ---------------------------------------------------------------------------
def ome_attrs(group):
    """The OME metadata dict (v0.5 nests it under 'ome'), or None if not OME-Zarr."""
    attrs = dict(group.attrs)
    if "ome" in attrs and isinstance(attrs["ome"], dict):
        attrs = attrs["ome"]
    return attrs if attrs.get("multiscales") else None


def _axes(ms):
    axes = ms.get("axes")
    if not axes:                               # OME-NGFF < 0.4: always tczyx
        return [{"name": n} for n in "tczyx"]
    return [a if isinstance(a, dict) else {"name": a} for a in axes]


def _scales(ms, ds, ndim):
    s = np.ones(ndim)
    for tr in (ds.get("coordinateTransformations") or []) + (ms.get("coordinateTransformations") or []):
        if tr.get("type") == "scale" and len(tr.get("scale", [])) == ndim:
            s = s * np.asarray(tr["scale"], dtype=float)
    return s


def describe(path) -> dict:
    """Pyramid levels, axes, pixel sizes and channels without reading pixels."""
    g = open_group(path)
    ome = ome_attrs(g)
    if ome is None:
        raise ValueError(f"{path} is a zarr store but not an OME-Zarr image (no 'multiscales')")
    ms = ome["multiscales"][0]
    axes = _axes(ms)
    levels = []
    for k, ds in enumerate(ms["datasets"]):
        arr = g[ds["path"]]
        levels.append({"level": k, "path": ds["path"], "shape": list(arr.shape), "dtype": str(arr.dtype),
                       "scale": [float(v) for v in _scales(ms, ds, len(arr.shape))]})
    chans = []
    for ch in (ome.get("omero") or {}).get("channels", []) or []:
        win = ch.get("window") or {}
        chans.append({"label": ch.get("label"), "color": ch.get("color"),
                      "start": win.get("start"), "end": win.get("end")})
    return {"axes": [a.get("name") for a in axes], "units": [a.get("unit") for a in axes],
            "levels": levels, "channels": chans, "name": ms.get("name"),
            "version": ome.get("version") or ms.get("version")}


def choose_level(levels, max_bytes=DEFAULT_MAX_BYTES, max_side=DEFAULT_MAX_SIDE, axes=None):
    """Index of the largest pyramid level within the memory and size budget
    (the smallest level if none fits)."""
    yx = [i for i, a in enumerate(axes or []) if a in ("y", "x")] or [-2, -1]
    for lv in levels:
        shape = lv["shape"]
        nbytes = int(np.prod(shape)) * np.dtype(lv["dtype"]).itemsize
        if nbytes <= max_bytes and max(shape[i] for i in yx) <= max_side:
            return lv["level"]
    return levels[-1]["level"]


def read_ome_zarr(path, level=None, max_bytes=None, max_side=DEFAULT_MAX_SIDE):
    """Read one pyramid level of an OME-Zarr image -> (metadata, array tdhwc)."""
    if level is None:
        level = _level_from_fragment(path)
    info = describe(path)
    g = open_group(path)
    if max_bytes is None:
        # a fixed budget (not a share of free RAM): the same figure must always
        # read the same level, or its panel sizes / pixel size would change
        max_bytes = DEFAULT_MAX_BYTES
    if level is None:
        level = choose_level(info["levels"], max_bytes, max_side, info["axes"])
    if not 0 <= level < len(info["levels"]):
        raise ValueError(f"level {level} out of range (the image has {len(info['levels'])} levels)")
    lv = info["levels"][level]
    arr = np.asarray(g[lv["path"]][...])

    names = [str(n).lower() for n in info["axes"]]
    order = {"t": "t", "z": "d", "y": "h", "x": "w", "c": "c"}
    keep = []                                   # (source axis, batoolset letter)
    for letter_src, letter in order.items():
        if letter_src in names:
            keep.append((names.index(letter_src), letter))
    # target order t d h w c; drop size-1 axes except y / x
    target = [k for k in keep if k[1] in "td"] + [k for k in keep if k[1] in "hw"] + [k for k in keep if k[1] == "c"]
    arr = np.transpose(arr, [k[0] for k in target] + [i for i in range(arr.ndim) if i not in [k[0] for k in target]])
    letters = [k[1] for k in target]
    squeeze = [i for i, L in enumerate(letters) if arr.shape[i] == 1 and L not in "hw"]
    arr = np.squeeze(arr, axis=tuple(squeeze)) if squeeze else arr
    letters = [L for i, L in enumerate(letters) if i not in squeeze]
    while arr.ndim > len(letters):              # unexpected extra axes: drop them
        arr = arr[0]
    dims = "".join(letters)

    def pixel_size(src_name):
        if src_name not in names:
            return None
        i = names.index(src_name)
        unit = (info["units"][i] or "micrometer").lower()
        return float(lv["scale"][i]) * _UNIT_TO_UM.get(unit, 1.0)

    shape = dict(zip(letters, arr.shape))
    chans = info["channels"]
    meta = {"w": shape.get("w"), "h": shape.get("h"), "d": shape.get("d"), "t": shape.get("t"),
            "c": shape.get("c"), "bits": arr.dtype.itemsize * 8, "dimensions": dims,
            "vx": pixel_size("x"), "vy": pixel_size("y"), "vz": pixel_size("z"),
            "AR": None, "LUTs": None, "times": None, "Overlays": None, "ROI": None,
            "timelapse": None, "creation_time": None,
            "channel_names": [c["label"] for c in chans] or None,
            "channel_colors": [c["color"] for c in chans] or None,
            "channel_windows": [(c["start"], c["end"]) for c in chans] or None,
            "ome_zarr": {"level": level, "levels": len(info["levels"]),
                         "full_shape": info["levels"][0]["shape"], "level_shape": lv["shape"],
                         "axes": info["axes"], "version": info["version"], "name": info["name"]}}
    if meta["vx"] and meta["vz"]:
        meta["AR"] = meta["vz"] / meta["vx"]
    luts = color_luts(meta["channel_colors"], shape.get("c"))
    if luts is not None:
        meta["LUTs"] = luts
    return meta, arr


def color_luts(colors, n_channels):
    """OME channel colours ('RRGGBB') -> ImageJ-style LUTs (one 3 x 256 uint8
    black-to-colour ramp per channel), or None if they do not match the channels."""
    if not colors or not n_channels or len(colors) != n_channels:
        return None
    ramp = np.arange(256, dtype=float) / 255.0
    luts = []
    for col in colors:
        try:
            c = str(col).lstrip("#")[:6]
            rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
        except (ValueError, TypeError):
            return None
        luts.append(np.stack([np.round(ramp * v) for v in rgb]).astype(np.uint8))
    return luts


def level_size(path):
    """(level, width, height) of the pyramid level read_ome_zarr(path) would
    read, from the metadata only (no pixels)."""
    info = describe(path)
    level = _level_from_fragment(path)
    if level is None:
        level = choose_level(info["levels"], DEFAULT_MAX_BYTES, DEFAULT_MAX_SIDE, info["axes"])
    shape = info["levels"][level]["shape"]
    names = [str(n).lower() for n in info["axes"]]
    w = shape[names.index("x")] if "x" in names else shape[-1]
    h = shape[names.index("y")] if "y" in names else shape[-2]
    return level, int(w), int(h)


def is_ome_zarr(path) -> bool:
    """True when *path* is a zarr store carrying OME multiscales metadata."""
    if not is_zarr_path(path):
        return False
    try:
        return ome_attrs(open_group(path)) is not None
    except Exception:
        return False


def write_ome_zarr(path, arr, axes="tczyx", scale=None, channel_names=None, levels=2,
                   channel_colors=None):
    """Write a small OME-NGFF 0.4 image (zarr v2 format) — used by the tests and
    to create sample data. *arr* must follow *axes*."""
    import zarr
    try:
        g = zarr.open_group(path, mode="w", zarr_format=2)
    except TypeError:
        g = zarr.open_group(path, mode="w")
    scale = list(scale or [1.0] * arr.ndim)
    datasets = []
    cur = np.asarray(arr)
    for k in range(levels):
        try:
            g.create_array(str(k), data=cur, chunks=cur.shape)
        except AttributeError:
            g.array(str(k), cur, chunks=cur.shape)
        s = [v * (2 ** k if a in "yx" else 1) for v, a in zip(scale, axes)]
        datasets.append({"path": str(k), "coordinateTransformations": [{"type": "scale", "scale": s}]})
        sl = tuple(slice(None, None, 2) if a in "yx" else slice(None) for a in axes)
        cur = cur[sl]
    ax = []
    for a in axes:
        d = {"name": a, "type": {"t": "time", "c": "channel"}.get(a, "space")}
        if a in "zyx":
            d["unit"] = "micrometer"
        ax.append(d)
    g.attrs["multiscales"] = [{"version": "0.4", "axes": ax, "datasets": datasets, "name": os.path.basename(path)}]
    if channel_names and "c" in axes:
        colors = channel_colors or ["FFFFFF"] * len(channel_names)
        g.attrs["omero"] = {"channels": [{"label": n, "color": c, "window": {"start": 0, "end": 255}}
                                         for n, c in zip(channel_names, colors)]}
    return path
