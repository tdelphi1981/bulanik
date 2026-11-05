"""Download a file once into benchmarks/.cache and verify its SHA-256."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache"


def fetch(url: str, name: str, sha256: str) -> Path:
    """Return the cached file, downloading it first if needed; raise if the digest differs."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "bulanik-benchmarks"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        path.unlink()
        raise RuntimeError(f"{name}: SHA-256 {digest} differs from the recorded {sha256}")
    return path
