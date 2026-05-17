"""Tests for `patchcraft.Cache` — contract from docs/THEORY.md §9.5."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from patchcraft import Cache

# ---------------------------------------------------------------- Roundtrip --

class TestRoundtrip:
    def test_put_then_get(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("hello", 42)
        c.put(key, b"some bytes")
        assert c.get(key) == b"some bytes"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        assert c.get(c.key_for("absent")) is None

    def test_accepts_bytearray_and_memoryview(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        k1, k2 = c.key_for(1), c.key_for(2)
        c.put(k1, bytearray(b"x" * 100))
        c.put(k2, memoryview(b"y" * 100))
        assert c.get(k1) == b"x" * 100
        assert c.get(k2) == b"y" * 100

    def test_large_payload(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        data = b"A" * (1024 * 1024)  # 1 MiB
        key = c.key_for("big")
        c.put(key, data)
        assert c.get(key) == data

    def test_empty_payload(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("empty")
        c.put(key, b"")
        assert c.get(key) == b""


# ------------------------------------------------------------------ Keying --

class TestKeyFor:
    def test_same_parts_same_key(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        assert c.key_for("a", 1, [2, 3]) == c.key_for("a", 1, [2, 3])

    def test_different_parts_different_keys(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        assert c.key_for("a") != c.key_for("b")
        assert c.key_for(1, 2) != c.key_for(2, 1)

    def test_namespace_changes_key(self, tmp_path: Path) -> None:
        a = Cache(tmp_path, namespace="ns_a").key_for("x")
        b = Cache(tmp_path, namespace="ns_b").key_for("x")
        assert a != b

    def test_version_changes_key(self, tmp_path: Path) -> None:
        a = Cache(tmp_path, namespace="t", version=1).key_for("x")
        b = Cache(tmp_path, namespace="t", version=2).key_for("x")
        assert a != b

    def test_dict_order_does_not_change_key(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        assert c.key_for({"a": 1, "b": 2}) == c.key_for({"b": 2, "a": 1})

    def test_bytes_parts_hashed_in_key(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        k1 = c.key_for(b"image-bytes-1")
        k2 = c.key_for(b"image-bytes-2")
        assert k1 != k2

    def test_keys_are_hex_digests(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("x")
        assert len(key) == 64
        assert all(ch in "0123456789abcdef" for ch in key)


# ---------------------------------------------------------- Version & namespace --

class TestVersioning:
    def test_version_bump_is_cache_miss(self, tmp_path: Path) -> None:
        c1 = Cache(tmp_path, namespace="t", version=1)
        key1 = c1.key_for("x")
        c1.put(key1, b"v1 data")

        c2 = Cache(tmp_path, namespace="t", version=2)
        # Different key by construction; old data is unreachable.
        key2 = c2.key_for("x")
        assert key1 != key2
        assert c2.get(key2) is None

    def test_namespace_isolation(self, tmp_path: Path) -> None:
        a = Cache(tmp_path, namespace="resize")
        b = Cache(tmp_path, namespace="extract")
        key_a = a.key_for("x")
        key_b = b.key_for("x")
        a.put(key_a, b"resize data")
        assert b.get(key_b) is None
        # Different physical subdirs
        assert a.root != b.root

    def test_root_auto_created(self, tmp_path: Path) -> None:
        sub = tmp_path / "nested" / "deep" / "cache"
        c = Cache(sub, namespace="t")
        assert c.root.exists()


# ---------------------------------------------------------------- Corruption --

class TestCorruption:
    def test_corruption_raises_oserror(self, tmp_path: Path) -> None:
        """Tampered .bin file is detected — either via zstd decode failure
        (when compressed) or via checksum mismatch (when uncompressed).
        Both surface as `OSError` with a clear message."""
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("x")
        c.put(key, b"original payload")

        # Tamper with the .bin file directly.
        bin_path = next(c.root.glob("*.bin"))
        bin_path.write_bytes(b"tampered different bytes content!")

        with pytest.raises(OSError, match=r"(checksum mismatch|payload corrupt)"):
            c.get(key)

    def test_missing_sidecar_returns_none(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("x")
        c.put(key, b"data")
        next(c.root.glob("*.json")).unlink()
        assert c.get(key) is None

    def test_missing_payload_returns_none(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("x")
        c.put(key, b"data")
        next(c.root.glob("*.bin")).unlink()
        assert c.get(key) is None


# ------------------------------------------------------------- OneDrive race --

class TestRetry:
    """Simulated PermissionError on os.replace must be retried transparently."""

    def test_put_retries_on_permission_error(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("x")

        real_replace = __import__("os").replace
        calls = {"n": 0}

        def flaky_replace(src: str, dst: str) -> None:
            calls["n"] += 1
            if calls["n"] in {1, 3}:  # fail first attempt of each file
                raise PermissionError(13, "Access is denied (simulated)")
            real_replace(src, dst)

        with (
            patch("patchcraft.cache.os.replace", side_effect=flaky_replace),
            patch("patchcraft.cache.time.sleep", return_value=None),
        ):
            c.put(key, b"resilient")
        assert c.get(key) == b"resilient"
        assert calls["n"] >= 4  # 2 retries + 2 successes (bin + sidecar)

    def test_put_gives_up_after_max_attempts(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("x")

        def always_fail(src: str, dst: str) -> None:
            raise PermissionError(13, "Access denied (persistent)")

        with (
            patch("patchcraft.cache.os.replace", side_effect=always_fail),
            patch("patchcraft.cache.time.sleep", return_value=None),
            pytest.raises(PermissionError),
        ):
            c.put(key, b"will fail")


# ----------------------------------------------------------------- Non-ASCII --

class TestNonASCIIPath:
    def test_works_under_path_with_accented_chars(self, tmp_path: Path) -> None:
        """OneDrive `Acadêmicos/` already triggers Unicode-on-Windows quirks."""
        weird = tmp_path / "Acadêmicos" / "patchcraft-cache"
        c = Cache(weird, namespace="ñámespace")
        key = c.key_for("héllo")
        c.put(key, b"unicode path payload")
        assert c.get(key) == b"unicode path payload"


# ---------------------------------------------------------------- Validation --

class TestRejects:
    def test_namespace_must_be_nonempty(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="namespace"):
            Cache(tmp_path, namespace="")

    def test_namespace_must_be_str(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="namespace"):
            Cache(tmp_path, namespace=42)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "1"])
    def test_version_must_be_positive_int(self, tmp_path: Path, bad: object) -> None:
        with pytest.raises(ValueError, match="version"):
            Cache(tmp_path, namespace="t", version=bad)  # type: ignore[arg-type]

    def test_version_bool_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="version"):
            Cache(tmp_path, namespace="t", version=True)  # type: ignore[arg-type]

    def test_put_rejects_non_str_key(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        with pytest.raises(TypeError, match="key must be str"):
            c.put(123, b"data")  # type: ignore[arg-type]

    def test_put_rejects_non_bytes_data(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        with pytest.raises(TypeError, match="bytes-like"):
            c.put(c.key_for("x"), "not bytes")  # type: ignore[arg-type]

    def test_get_rejects_non_str_key(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="t")
        with pytest.raises(TypeError, match="key must be str"):
            c.get(123)  # type: ignore[arg-type]


# -------------------------------------------------------------- Repr / props --

class TestProperties:
    def test_root_property(self, tmp_path: Path) -> None:
        c = Cache(tmp_path, namespace="foo")
        assert c.root == tmp_path / "foo"

    def test_namespace_property(self, tmp_path: Path) -> None:
        assert Cache(tmp_path, namespace="foo").namespace == "foo"

    def test_version_property(self, tmp_path: Path) -> None:
        assert Cache(tmp_path, namespace="foo", version=3).version == 3

    def test_repr_includes_config(self, tmp_path: Path) -> None:
        r = repr(Cache(tmp_path, namespace="ns", version=7))
        assert "Cache(" in r
        assert "'ns'" in r
        assert "version=7" in r
