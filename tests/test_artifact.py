"""Tests for artifact storage — path validation, filesystem backend, and store."""

import asyncio
import base64
from unittest.mock import MagicMock

import pytest

from mnemory.config import ArtifactConfig
from mnemory.storage.artifact import (
    ArtifactMetadata,
    ArtifactStore,
    FilesystemBackend,
    S3Backend,
    _validate_path_component,
)


def test_artifact_partial_delete_error_propagates():
    backend = S3Backend.__new__(S3Backend)
    backend._bucket = "synthetic-bucket"
    backend._client = MagicMock()
    backend._client.list_objects_v2.return_value = {
        "Contents": [{"Key": "synthetic-user/artifact/data.txt"}]
    }
    backend._client.delete_objects.return_value = {"Errors": [{"Code": "AccessDenied"}]}
    with pytest.raises(RuntimeError, match="cleanup was rejected"):
        backend.delete_artifact("synthetic-user", "artifact")


# ── _validate_path_component ──────────────────────────────────────────


class TestValidatePathComponent:
    def test_valid_simple(self):
        _validate_path_component("user123", "user_id")  # should not raise

    def test_valid_with_hyphens_dots(self):
        _validate_path_component("my-file.txt", "filename")

    def test_valid_with_colons(self):
        _validate_path_component("project:myapp", "category")

    def test_valid_with_slashes(self):
        _validate_path_component("project:demo/deployments", "category")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_path_component("", "user_id")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            _validate_path_component("a" * 257, "user_id")

    def test_dotdot_raises(self):
        with pytest.raises(ValueError, match="must not contain"):
            _validate_path_component("../etc/passwd", "filename")

    def test_dotdot_in_middle_raises(self):
        with pytest.raises(ValueError, match="must not contain"):
            _validate_path_component("foo/../bar", "filename")

    def test_null_byte_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_path_component("file\x00name", "filename")

    def test_space_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_path_component("file name", "filename")

    def test_backslash_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_path_component("file\\name", "filename")


# ── ArtifactMetadata ──────────────────────────────────────────────────


class TestArtifactMetadata:
    def test_to_dict(self):
        meta = ArtifactMetadata(
            artifact_id="abc123",
            filename="note.md",
            content_type="text/markdown",
            size=1024,
            created_at="2026-01-01T00:00:00+00:00",
        )
        d = meta.to_dict()
        assert d["id"] == "abc123"
        assert d["filename"] == "note.md"
        assert d["content_type"] == "text/markdown"
        assert d["size"] == 1024

    def test_from_dict_roundtrip(self):
        original = ArtifactMetadata(
            artifact_id="abc123",
            filename="note.md",
            content_type="text/markdown",
            size=1024,
            created_at="2026-01-01T00:00:00+00:00",
        )
        restored = ArtifactMetadata.from_dict(original.to_dict())
        assert restored.artifact_id == original.artifact_id
        assert restored.filename == original.filename
        assert restored.size == original.size


# ── FilesystemBackend path traversal ──────────────────────────────────


class TestFilesystemBackendSecurity:
    def test_path_traversal_in_filename(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        backend = FilesystemBackend(config)

        with pytest.raises(ValueError):
            backend._path("user1", "art1", "../../etc/passwd")

    def test_path_traversal_in_user_id(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        backend = FilesystemBackend(config)

        with pytest.raises(ValueError):
            backend._path("../../etc", "art1", "file.txt")

    def test_valid_path_works(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        backend = FilesystemBackend(config)

        path = backend._path("user1", "art1", "note.md")
        assert str(tmp_path) in str(path)

    def test_path_format_no_memory_id(self, tmp_path):
        """Path should be {base}/{user_id}/{artifact_id}/{filename}."""
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        backend = FilesystemBackend(config)

        path = backend._path("user1", "art1", "note.md")
        # Path should be: tmp_path/user1/art1/note.md
        expected = tmp_path / "user1" / "art1" / "note.md"
        assert path == expected


# ── ArtifactStore ─────────────────────────────────────────────────────


class TestArtifactStore:
    def test_max_size_enforced(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=100)

        with pytest.raises(ValueError, match="Artifact too large"):
            store.save(
                user_id="user1",
                content="x" * 200,
                filename="big.txt",
                content_type="text/plain",
            )

    def test_save_and_load_text(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        meta = store.save(
            user_id="user1",
            content="Hello, world!",
            filename="note.txt",
            content_type="text/plain",
        )

        assert meta.filename == "note.txt"
        assert meta.size == len("Hello, world!".encode("utf-8"))

        result = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )
        assert result["content"] == "Hello, world!"
        assert result["is_text"] is True
        assert result["has_more"] is False

    def test_load_with_pagination(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        content = "A" * 100
        meta = store.save(
            user_id="user1",
            content=content,
            filename="big.txt",
            content_type="text/plain",
        )

        result = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
            offset=0,
            limit=50,
        )
        assert len(result["content"]) == 50
        assert result["has_more"] is True

    def test_delete_artifact(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        meta = store.save(
            user_id="user1",
            content="test",
            filename="note.txt",
            content_type="text/plain",
        )

        store.delete(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )

        with pytest.raises(FileNotFoundError):
            store.load(
                user_id="user1",
                artifact_id=meta.artifact_id,
                artifacts_meta=[meta.to_dict()],
            )

    def test_delete_nonexistent_raises(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        with pytest.raises(ValueError, match="not found"):
            store.delete(
                user_id="user1",
                artifact_id="nonexistent",
                artifacts_meta=[],
            )

    def test_invalid_backend_raises(self):
        config = ArtifactConfig()
        config.backend = "invalid"
        with pytest.raises(ValueError, match="Unsupported artifact backend"):
            ArtifactStore(config)

    def test_save_with_explicit_artifact_id(self, tmp_path):
        """When artifact_id is provided, it should be used instead of generating one."""
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        meta = store.save(
            user_id="user1",
            content="test content",
            filename="doc.md",
            content_type="text/markdown",
            artifact_id="custom-id-123",
        )

        assert meta.artifact_id == "custom-id-123"
        assert meta.filename == "doc.md"

        # Verify it can be loaded
        result = store.load(
            user_id="user1",
            artifact_id="custom-id-123",
            artifacts_meta=[meta.to_dict()],
        )
        assert result["content"] == "test content"

    def test_delete_by_id(self, tmp_path):
        """delete_by_id should remove artifact without needing metadata."""
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        meta = store.save(
            user_id="user1",
            content="test",
            filename="note.txt",
            content_type="text/plain",
        )

        # Delete using delete_by_id (no metadata needed)
        store.delete_by_id(user_id="user1", artifact_id=meta.artifact_id)

        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            store.load(
                user_id="user1",
                artifact_id=meta.artifact_id,
                artifacts_meta=[meta.to_dict()],
            )

    def test_delete_all_for_user(self, tmp_path):
        """delete_all_for_user should remove all artifacts for a user."""
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        # Save two artifacts
        meta1 = store.save(
            user_id="user1",
            content="first",
            filename="a.txt",
            content_type="text/plain",
        )
        meta2 = store.save(
            user_id="user1",
            content="second",
            filename="b.txt",
            content_type="text/plain",
        )

        store.delete_all_for_user(user_id="user1")

        # Both should be gone
        with pytest.raises(FileNotFoundError):
            store.load(
                user_id="user1",
                artifact_id=meta1.artifact_id,
                artifacts_meta=[meta1.to_dict()],
            )
        with pytest.raises(FileNotFoundError):
            store.load(
                user_id="user1",
                artifact_id=meta2.artifact_id,
                artifacts_meta=[meta2.to_dict()],
            )

    def test_delete_all_for_user_preserves_other_users(self, tmp_path):
        """delete_all_for_user should not affect other users."""
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        meta1 = store.save(
            user_id="user1",
            content="user1 data",
            filename="a.txt",
            content_type="text/plain",
        )
        meta2 = store.save(
            user_id="user2",
            content="user2 data",
            filename="b.txt",
            content_type="text/plain",
        )

        store.delete_all_for_user(user_id="user1")

        # user1's artifact should be gone
        with pytest.raises(FileNotFoundError):
            store.load(
                user_id="user1",
                artifact_id=meta1.artifact_id,
                artifacts_meta=[meta1.to_dict()],
            )

        # user2's artifact should still exist
        result = store.load(
            user_id="user2",
            artifact_id=meta2.artifact_id,
            artifacts_meta=[meta2.to_dict()],
        )
        assert result["content"] == "user2 data"


# ── Binary artifact tests ────────────────────────────────────────────


# Minimal valid 1x1 red PNG (67 bytes)
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestBinaryArtifacts:
    """Tests for binary (non-text) artifact save, load, and round-trip."""

    def _make_store(self, tmp_path, max_size=10_485_760):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        return ArtifactStore(config, max_artifact_size=max_size)

    def test_save_and_load_binary_via_base64(self, tmp_path):
        """Binary content passed as base64 string should round-trip correctly."""
        store = self._make_store(tmp_path)
        b64_content = base64.b64encode(_TINY_PNG).decode("ascii")

        meta = store.save(
            user_id="user1",
            content=b64_content,
            filename="image.png",
            content_type="image/png",
        )

        assert meta.filename == "image.png"
        assert meta.content_type == "image/png"
        assert meta.size == len(_TINY_PNG)

        result = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )

        assert result["is_text"] is False
        assert result["has_more"] is False
        assert result["offset"] == 0
        assert result["total_size"] == len(_TINY_PNG)
        assert result["content_type"] == "image/png"
        assert result["filename"] == "image.png"

        # Decode the returned base64 and verify it matches the original
        decoded = base64.b64decode(result["content"])
        assert decoded == _TINY_PNG

    def test_save_binary_as_raw_bytes(self, tmp_path):
        """Binary content passed as raw bytes should be stored directly."""
        store = self._make_store(tmp_path)

        meta = store.save(
            user_id="user1",
            content=_TINY_PNG,
            filename="photo.png",
            content_type="image/png",
        )

        assert meta.size == len(_TINY_PNG)

        result = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )

        decoded = base64.b64decode(result["content"])
        assert decoded == _TINY_PNG

    def test_binary_load_returns_full_content_ignoring_limit(self, tmp_path):
        """Binary load should return full content regardless of limit parameter."""
        store = self._make_store(tmp_path)
        # Create a larger binary blob (10KB)
        large_binary = bytes(range(256)) * 40  # 10240 bytes

        meta = store.save(
            user_id="user1",
            content=large_binary,
            filename="data.bin",
            content_type="application/octet-stream",
        )

        # Load with a very small limit — should still get full content
        result = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
            offset=50,
            limit=10,
        )

        assert result["is_text"] is False
        assert result["has_more"] is False
        assert result["offset"] == 0  # offset ignored for binary
        assert result["total_size"] == len(large_binary)

        decoded = base64.b64decode(result["content"])
        assert decoded == large_binary
        assert len(decoded) == 10240

    def test_text_pagination_still_works(self, tmp_path):
        """Text artifacts should still support pagination (not affected by binary fix)."""
        store = self._make_store(tmp_path)
        content = "A" * 200

        meta = store.save(
            user_id="user1",
            content=content,
            filename="notes.txt",
            content_type="text/plain",
        )

        # First page
        result1 = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
            offset=0,
            limit=50,
        )
        assert result1["is_text"] is True
        assert result1["has_more"] is True
        assert len(result1["content"]) == 50
        assert result1["offset"] == 0

        # Second page
        result2 = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
            offset=50,
            limit=50,
        )
        assert result2["is_text"] is True
        assert result2["has_more"] is True
        assert len(result2["content"]) == 50
        assert result2["offset"] == 50

    def test_load_raw_binary(self, tmp_path):
        """load_raw should return raw bytes, content_type, and filename."""
        store = self._make_store(tmp_path)
        b64_content = base64.b64encode(_TINY_PNG).decode("ascii")

        meta = store.save(
            user_id="user1",
            content=b64_content,
            filename="photo.png",
            content_type="image/png",
        )

        raw_bytes, content_type, filename = store.load_raw(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )

        assert raw_bytes == _TINY_PNG
        assert content_type == "image/png"
        assert filename == "photo.png"

    def test_load_raw_text(self, tmp_path):
        """load_raw should also work for text artifacts (returns UTF-8 bytes)."""
        store = self._make_store(tmp_path)

        meta = store.save(
            user_id="user1",
            content="Hello, world!",
            filename="note.txt",
            content_type="text/plain",
        )

        raw_bytes, content_type, filename = store.load_raw(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )

        assert raw_bytes == b"Hello, world!"
        assert content_type == "text/plain"
        assert filename == "note.txt"

    def test_binary_pdf_round_trip(self, tmp_path):
        """PDF-like binary content should round-trip correctly."""
        store = self._make_store(tmp_path)
        # Fake PDF header + random bytes
        pdf_bytes = b"%PDF-1.4\n" + bytes(range(256)) * 10

        b64_content = base64.b64encode(pdf_bytes).decode("ascii")
        meta = store.save(
            user_id="user1",
            content=b64_content,
            filename="report.pdf",
            content_type="application/pdf",
        )

        result = store.load(
            user_id="user1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )

        assert result["is_text"] is False
        assert result["content_type"] == "application/pdf"
        decoded = base64.b64decode(result["content"])
        assert decoded == pdf_bytes

    def test_max_size_enforced_for_binary(self, tmp_path):
        """Binary artifacts exceeding max size should be rejected."""
        store = self._make_store(tmp_path, max_size=100)
        large_binary = b"\x00" * 200
        b64_content = base64.b64encode(large_binary).decode("ascii")

        with pytest.raises(ValueError, match="Artifact too large"):
            store.save(
                user_id="user1",
                content=b64_content,
                filename="big.bin",
                content_type="application/octet-stream",
            )


# ── MCP binary inline cap ────────────────────────────────────────────


class TestMCPBinaryInlineCap:
    """Test that the MCP get_artifact tool caps binary artifacts at 1 MB."""

    def test_large_binary_returns_guidance(self):
        """Binary artifact >1 MB should return guidance to use get_artifact_url."""
        import json
        from unittest.mock import patch

        # Mock the service to return a large binary result
        large_result = {
            "content": "base64data...",
            "total_size": 2_000_000,  # 2 MB
            "is_text": False,
            "has_more": False,
            "content_type": "image/png",
        }

        with (
            patch("mnemory.server._resolve_user_id", return_value="user1"),
            patch("mnemory.server._get_service") as mock_svc,
            patch("mnemory.server.get_collector", return_value=None),
        ):
            mock_svc.return_value.get_artifact.return_value = large_result

            from mnemory.server import get_artifact

            result_str = asyncio.run(get_artifact("mem-1", "art-1", user_id="user1"))
            result = json.loads(result_str)

        assert result["error"] is True
        assert "too large" in result["message"]
        assert result["use_tool"] == "get_artifact_url"
        assert result["total_size"] == 2_000_000

    def test_small_binary_returns_content(self):
        """Binary artifact <=1 MB should return content normally."""
        import json
        from unittest.mock import patch

        small_result = {
            "content": "base64data...",
            "total_size": 500_000,  # 500 KB
            "is_text": False,
            "has_more": False,
            "content_type": "image/png",
        }

        with (
            patch("mnemory.server._resolve_user_id", return_value="user1"),
            patch("mnemory.server._get_service") as mock_svc,
            patch("mnemory.server.get_collector", return_value=None),
        ):
            mock_svc.return_value.get_artifact.return_value = small_result

            from mnemory.server import get_artifact

            result_str = asyncio.run(get_artifact("mem-1", "art-1", user_id="user1"))
            result = json.loads(result_str)

        assert "error" not in result
        assert result["content"] == "base64data..."
        assert result["total_size"] == 500_000

    def test_text_artifact_not_capped(self):
        """Text artifacts should never be capped regardless of size."""
        import json
        from unittest.mock import patch

        text_result = {
            "content": "x" * 5000,
            "total_size": 5_000_000,  # 5 MB of text
            "is_text": True,
            "has_more": True,
        }

        with (
            patch("mnemory.server._resolve_user_id", return_value="user1"),
            patch("mnemory.server._get_service") as mock_svc,
            patch("mnemory.server.get_collector", return_value=None),
        ):
            mock_svc.return_value.get_artifact.return_value = text_result

            from mnemory.server import get_artifact

            result_str = asyncio.run(get_artifact("mem-1", "art-1", user_id="user1"))
            result = json.loads(result_str)

        assert "error" not in result
        assert result["is_text"] is True
        assert result["has_more"] is True
