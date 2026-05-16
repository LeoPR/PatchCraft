"""Content-addressed disk cache.

Bytes in, bytes out. Optional ``zstandard`` compression (transparent fallback
when not installed). Retry on transient ``PermissionError`` from OneDrive,
antivirus, or the Windows Search indexer. Atomic write via ``*.tmp`` plus
``os.replace``. Sidecar JSON carries the full key, version, content checksum.

Contract: docs/THEORY.md §4 and §9.5.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["Cache"]

_WRITE_BACKOFF: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
_READ_BACKOFF: tuple[float, ...] = (0.25,)


def _retry[T](
    op: Callable[[], T],
    backoff: tuple[float, ...],
) -> T:
    """Retry ``op`` on ``PermissionError`` with the given backoff schedule.

    First attempt is immediate; subsequent attempts sleep for
    ``backoff[i-1]`` seconds. After exhausting the schedule the last
    ``PermissionError`` is re-raised. All other exceptions propagate
    immediately.
    """
    last: PermissionError | None = None
    for attempt in range(len(backoff) + 1):
        try:
            return op()
        except PermissionError as exc:
            last = exc
            if attempt < len(backoff):
                time.sleep(backoff[attempt])
    assert last is not None  # invariant from loop above
    raise last


def _try_zstandard() -> Any | None:
    try:
        import zstandard
    except ImportError:
        return None
    return zstandard


def _normalize_part(part: Any) -> Any:
    """Coerce a key-part into something JSON-stable.

    Lists and tuples both serialize as JSON arrays — order matters. Dicts
    are sorted by key. Bytes are hashed (avoids embedding binary blobs in
    the key). Anything else falls back to ``repr`` — explicit but uglier."""
    if isinstance(part, (str, int, float, bool)) or part is None:
        return part
    if isinstance(part, (list, tuple)):
        return [_normalize_part(p) for p in part]
    if isinstance(part, dict):
        return {str(k): _normalize_part(v) for k, v in sorted(part.items())}
    if isinstance(part, bytes):
        return {"__bytes_sha256__": hashlib.sha256(part).hexdigest()}
    return {"__repr__": repr(part)}


class Cache:
    """Single-namespace content-addressed cache on disk.

    ``root`` is created on construction if missing. ``namespace`` is used
    as a subdirectory and as part of the key (so two namespaces never
    collide even if a caller produces identical key parts). ``version``
    is the invalidation lever: bump it, and old entries become
    unreadable by construction without any delete.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        namespace: str,
        version: int = 1,
    ) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(
                f"namespace must be a non-empty str, got {namespace!r}"
            )
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise ValueError(f"version must be a positive int, got {version!r}")

        self._namespace = namespace
        self._version = version
        self._root = Path(root) / namespace
        self._root.mkdir(parents=True, exist_ok=True)
        self._zstd = _try_zstandard()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def version(self) -> int:
        return self._version

    def key_for(self, *parts: Any) -> str:
        """SHA-256 over a canonical JSON of ``parts``, namespace, version."""
        canonical = {
            "namespace": self._namespace,
            "version": self._version,
            "parts": [_normalize_part(p) for p in parts],
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        short = key[:16]
        return self._root / f"{short}.bin", self._root / f"{short}.json"

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``. Atomic; retries on transient races."""
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        data_bytes = bytes(data)

        bin_path, sidecar_path = self._paths(key)

        if self._zstd is not None:
            cctx = self._zstd.ZstdCompressor(level=3)
            payload = cctx.compress(data_bytes)
            compressed = True
        else:
            payload = data_bytes
            compressed = False

        checksum = hashlib.sha256(data_bytes).hexdigest()
        sidecar: dict[str, Any] = {
            "key": key,
            "namespace": self._namespace,
            "version": self._version,
            "checksum": checksum,
            "compressed": compressed,
            "size": len(data_bytes),
        }

        tmp_bin = bin_path.with_suffix(bin_path.suffix + ".tmp")
        tmp_side = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")

        def _write_payload() -> None:
            with open(tmp_bin, "wb") as f:
                f.write(payload)
            os.replace(tmp_bin, bin_path)

        def _write_sidecar() -> None:
            with open(tmp_side, "w", encoding="utf-8") as f:
                json.dump(sidecar, f)
            os.replace(tmp_side, sidecar_path)

        _retry(_write_payload, _WRITE_BACKOFF)
        _retry(_write_sidecar, _WRITE_BACKOFF)

    def get(self, key: str) -> bytes | None:
        """Return cached bytes, or ``None`` if absent / version-mismatched.

        Raises ``IOError`` on sidecar/payload mismatch (corrupt entry).
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")

        bin_path, sidecar_path = self._paths(key)
        if not (bin_path.exists() and sidecar_path.exists()):
            return None

        def _read_sidecar() -> dict[str, Any]:
            with open(sidecar_path, "rb") as f:
                parsed = json.loads(f.read())
            if not isinstance(parsed, dict):
                raise OSError(
                    f"sidecar at {sidecar_path} is not a JSON object"
                )
            return parsed

        try:
            sidecar = _retry(_read_sidecar, _READ_BACKOFF)
        except FileNotFoundError:
            return None

        # Different key collided into the same 16-hex prefix? Treat as miss.
        if sidecar.get("key") != key:
            return None
        # Stale version (cache invalidation lever). Transparent miss.
        if sidecar.get("version") != self._version:
            return None
        if sidecar.get("namespace") != self._namespace:
            return None

        def _read_payload() -> bytes:
            with open(bin_path, "rb") as f:
                return f.read()

        try:
            payload = _retry(_read_payload, _READ_BACKOFF)
        except FileNotFoundError:
            return None

        if sidecar.get("compressed"):
            if self._zstd is None:
                raise OSError(
                    f"cache entry {key[:16]!r} is zstd-compressed but "
                    "zstandard is not installed in this environment"
                )
            dctx = self._zstd.ZstdDecompressor()
            try:
                data: bytes = dctx.decompress(payload)
            except self._zstd.ZstdError as exc:
                # Compressed-payload corruption manifests here before checksum.
                raise OSError(
                    f"cache entry {key[:16]!r} payload corrupt (zstd decode "
                    f"failed); remove it from {self._root} to invalidate"
                ) from exc
        else:
            data = payload

        if hashlib.sha256(data).hexdigest() != sidecar.get("checksum"):
            raise OSError(
                f"cache entry {key[:16]!r} checksum mismatch — corrupt; "
                f"remove {bin_path.name} and {sidecar_path.name} from "
                f"{self._root} to invalidate"
            )
        return data

    def __repr__(self) -> str:
        return (
            f"Cache(root={str(self._root)!r}, namespace={self._namespace!r}, "
            f"version={self._version})"
        )
