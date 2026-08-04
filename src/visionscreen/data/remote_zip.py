"""Read selected members out of a huge remote ZIP without downloading it.

The OpenEDS 2019 archive is ~47 GB, but the semantic-segmentation subset is a
small fraction of it. ZIP stores its central directory at the end of the file,
so with HTTP range requests one can read the index, then fetch only the byte
ranges of the wanted members. This turns "download 47 GB" into "download the
part you actually need".

Requires the server to honour `Range` (Hugging Face's CDN does).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "visionscreen-research/0.5"}


class HTTPRangeFile(io.RawIOBase):
    """A seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str, chunk: int = 1 << 20):
        self.url = url
        self._pos = 0
        self._chunk = chunk
        self.size = self._content_length()

    def _content_length(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            # HF redirects to a CDN that reports the true size in x-linked-size
            linked = r.headers.get("x-linked-size")
            return int(linked or n)

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        headers = {**UA, "Range": f"bytes={start}-{end}"}
        req = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    # --- io plumbing ---
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self._pos
        data = self._fetch(self._pos, size)
        self._pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(HTTPRangeFile(url))


def extract_members(url: str, out_dir: Path, predicate, limit: int | None = None,
                    progress_every: int = 250) -> dict:
    """Extract only members for which `predicate(name) -> bool`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zf = open_remote_zip(url)
    names = [n for n in zf.namelist() if predicate(n)]
    if limit:
        names = names[:limit]

    written, errors, total_bytes = 0, [], 0
    for i, name in enumerate(names):
        dest = out_dir / Path(name).name
        if dest.exists() and dest.stat().st_size > 0:
            written += 1
            continue
        try:
            data = zf.read(name)
            dest.write_bytes(data)
            total_bytes += len(data)
            written += 1
        except Exception as exc:
            errors.append(f"{name}: {exc}")
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(names)} ({total_bytes / 1e6:.0f} MB)", flush=True)
    return {"matched": len(names), "written": written,
            "megabytes": round(total_bytes / 1e6, 1), "errors": errors[:10]}
