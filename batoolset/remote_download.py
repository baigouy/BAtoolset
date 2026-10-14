"""Local copies of remote images, for consolidation ("download remote images").

A consolidated figure normally keeps its remote images as links (http(s)
images, OME-Zarr URLs, OMERO links). With the download option they are saved
into the bundle instead:

- an OMERO link -> a local OME-Zarr of exactly what the panel shows (its
  planes, the raw pixels when omero-py is installed, channel names, colours,
  display windows and pixel size), so the panel looks the same;
- a remote OME-Zarr -> a local OME-Zarr of the pyramid level the figure
  reads, so the panel keeps its size and pixel size;
- any other http(s) image -> the file itself.

The source URL is recorded in the local OME-Zarr (attribute
``pyfigures_source``).
"""
from __future__ import annotations

import os
import re
from urllib.parse import unquote, urlsplit

import numpy as np

_BATOOLSET_TO_OME = {"t": "t", "c": "c", "d": "z", "h": "y", "w": "x"}


def is_remote(path) -> bool:
    return isinstance(path, str) and path.lower().startswith(("http://", "https://"))


def _safe(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "image"


def local_name(url) -> str:
    """The file or folder name a downloaded *url* gets in the bundle."""
    from batoolset import omero_web, ome_zarr
    if omero_web.is_omero_url(url):
        _, image_id = omero_web.parse_omero_url(url)
        return f"omero_image_{image_id}.ome.zarr"
    path = unquote(urlsplit(url.split("#", 1)[0]).path).rstrip("/")
    base = _safe(os.path.basename(path))
    if ome_zarr.is_zarr_path(url) and not base.lower().endswith(".zarr"):
        base += ".zarr"
    return base


def _write_local_zarr(dest, meta, arr, source_url):
    """*arr* in batoolset order (dims in meta['dimensions']) -> OME-Zarr at *dest*."""
    import zarr
    dims = meta.get("dimensions") or "hw"
    ome_axes = "".join(_BATOOLSET_TO_OME[d] for d in dims)
    # OME-NGFF order: t, c, z, y, x
    order = [ome_axes.index(a) for a in "tczyx" if a in ome_axes]
    axes = "".join(ome_axes[i] for i in order)
    data = np.ascontiguousarray(np.transpose(arr, order))
    try:
        g = zarr.open_group(dest, mode="w", zarr_format=2)
    except TypeError:
        g = zarr.open_group(dest, mode="w")
    chunks = tuple(min(n, 1024) if a in "yx" else 1 for a, n in zip(axes, data.shape))
    try:
        g.create_array("0", data=data, chunks=chunks)
    except AttributeError:
        g.array("0", data, chunks=chunks)
    size = {"x": meta.get("vx"), "y": meta.get("vy") or meta.get("vx"), "z": meta.get("vz")}
    scale = [float(size.get(a) or 1.0) for a in axes]
    ax = []
    for a in axes:
        d = {"name": a, "type": {"t": "time", "c": "channel"}.get(a, "space")}
        if a in "zyx":
            d["unit"] = "micrometer"
        ax.append(d)
    g.attrs["multiscales"] = [{"version": "0.4", "axes": ax, "name": os.path.basename(dest),
                               "datasets": [{"path": "0", "coordinateTransformations":
                                             [{"type": "scale", "scale": scale}]}]}]
    if "c" in axes:
        n = data.shape[axes.index("c")]
        names = meta.get("channel_names") or [f"Channel {i + 1}" for i in range(n)]
        colors = meta.get("channel_colors") or ["FFFFFF"] * n
        windows = meta.get("channel_windows") or [None] * n
        channels = []
        for i in range(n):
            ch = np.take(data, i, axis=axes.index("c"))
            lo, hi = (windows[i] if i < len(windows) and windows[i] else (None, None))
            lo = float(ch.min()) if lo is None else float(lo)
            hi = float(ch.max()) if hi is None else float(hi)
            channels.append({"label": str(names[i] if i < len(names) else f"Channel {i + 1}"),
                             "color": str(colors[i] if i < len(colors) else "FFFFFF"),
                             "window": {"start": lo, "end": hi, "min": float(ch.min()), "max": float(ch.max())},
                             "active": True})
        g.attrs["omero"] = {"channels": channels}
    g.attrs["pyfigures_source"] = source_url
    return dest


def download(url, dest):
    """Save the remote image *url* at *dest* (a path ending in local_name(url)).
    Returns *dest*. Raises on failure (the caller keeps the link)."""
    from batoolset import omero_web, ome_zarr
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if omero_web.is_omero_url(url):
        meta, arr = omero_web.read_omero(url)
        return _write_local_zarr(dest, meta, arr, url)
    if ome_zarr.is_zarr_path(url):
        meta, arr = ome_zarr.read_ome_zarr(url)
        return _write_local_zarr(dest, meta, arr, url)
    import requests
    r = requests.get(url.split("#", 1)[0], timeout=120)
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest
