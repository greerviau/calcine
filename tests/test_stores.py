"""Tests for built-in FeatureStore implementations."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from calcine import ExtractionResult
from calcine.features.base import Feature
from calcine.serializers import JSONSerializer, NumpySerializer
from calcine.stores import FileStore, MemoryStore
from calcine.stores.base import FeatureStore

# ---------------------------------------------------------------------------
# Shared fixture feature
# ---------------------------------------------------------------------------


class DummyFeature(Feature):
    async def extract(self, raw, context, entity_id=None):
        return ExtractionResult.of(entity_id, raw)


class AnotherFeature(Feature):
    async def extract(self, raw, context, entity_id=None):
        return ExtractionResult.of(entity_id, raw)


@pytest.fixture
def feature() -> DummyFeature:
    return DummyFeature()


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_write_and_read(self, feature):
        store = MemoryStore()
        await store.awrite(feature, "e1", ExtractionResult.of("e1", {"score": 0.9}))
        assert await store.aread(feature, "e1") == {"score": 0.9}

    @pytest.mark.asyncio
    async def test_exists_false_before_write(self, feature):
        store = MemoryStore()
        assert not await store.aexists(feature, "e1")

    @pytest.mark.asyncio
    async def test_exists_true_after_write(self, feature):
        store = MemoryStore()
        await store.awrite(feature, "e1", ExtractionResult.of("e1", 42))
        assert await store.aexists(feature, "e1")

    @pytest.mark.asyncio
    async def test_delete(self, feature):
        store = MemoryStore()
        await store.awrite(feature, "e1", ExtractionResult.of("e1", "data"))
        await store.adelete(feature, "e1")
        assert not await store.aexists(feature, "e1")

    @pytest.mark.asyncio
    async def test_read_missing_raises_key_error(self, feature):
        store = MemoryStore()
        with pytest.raises(KeyError):
            await store.aread(feature, "missing")

    @pytest.mark.asyncio
    async def test_delete_missing_raises_key_error(self, feature):
        store = MemoryStore()
        with pytest.raises(KeyError):
            await store.adelete(feature, "missing")

    @pytest.mark.asyncio
    async def test_overwrite(self, feature):
        store = MemoryStore()
        await store.awrite(feature, "e1", ExtractionResult.of("e1", "first"))
        await store.awrite(feature, "e1", ExtractionResult.of("e1", "second"))
        assert await store.aread(feature, "e1") == "second"

    @pytest.mark.asyncio
    async def test_feature_isolation(self):
        """Different feature classes should not share namespace."""
        fa = DummyFeature()
        fb = AnotherFeature()
        store = MemoryStore()

        await store.awrite(fa, "e1", ExtractionResult.of("e1", "value_a"))
        await store.awrite(fb, "e1", ExtractionResult.of("e1", "value_b"))

        assert await store.aread(fa, "e1") == "value_a"
        assert await store.aread(fb, "e1") == "value_b"

    @pytest.mark.asyncio
    async def test_stores_arbitrary_types(self, feature):
        store = MemoryStore()
        arr = np.zeros((3, 4), dtype=np.float32)
        await store.awrite(feature, "e1", ExtractionResult.of("e1", arr))
        result = await store.aread(feature, "e1")
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# FileStore
# ---------------------------------------------------------------------------


class TestFileStore:
    @pytest.mark.asyncio
    async def test_write_and_read_pickle(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            await store.awrite(feature, "e1", ExtractionResult.of("e1", {"key": "val", "num": 7}))
            result = await store.aread(feature, "e1")
            assert result == {"key": "val", "num": 7}

    @pytest.mark.asyncio
    async def test_exists(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            assert not await store.aexists(feature, "e1")
            await store.awrite(feature, "e1", ExtractionResult.of("e1", "hello"))
            assert await store.aexists(feature, "e1")

    @pytest.mark.asyncio
    async def test_delete(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            await store.awrite(feature, "e1", ExtractionResult.of("e1", "data"))
            await store.adelete(feature, "e1")
            assert not await store.aexists(feature, "e1")

    @pytest.mark.asyncio
    async def test_read_missing_raises_key_error(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            with pytest.raises(KeyError):
                await store.aread(feature, "missing")

    @pytest.mark.asyncio
    async def test_json_serializer(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir, serializer=JSONSerializer())
            payload = {"name": "alice", "count": 42}
            await store.awrite(feature, "e1", ExtractionResult.of("e1", payload))
            assert await store.aread(feature, "e1") == payload

    @pytest.mark.asyncio
    async def test_numpy_serializer(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir, serializer=NumpySerializer())
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            await store.awrite(feature, "e1", ExtractionResult.of("e1", arr))
            result = await store.aread(feature, "e1")
            np.testing.assert_array_equal(result, arr)

    @pytest.mark.asyncio
    async def test_creates_directories_automatically(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = f"{tmpdir}/a/b/c"
            store = FileStore(nested)
            await store.awrite(feature, "e1", ExtractionResult.of("e1", "nested"))
            assert await store.aread(feature, "e1") == "nested"

    @pytest.mark.asyncio
    async def test_overwrite(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            await store.awrite(feature, "e1", ExtractionResult.of("e1", "v1"))
            await store.awrite(feature, "e1", ExtractionResult.of("e1", "v2"))
            assert await store.aread(feature, "e1") == "v2"


# ---------------------------------------------------------------------------
# FeatureStore base class — default method behaviour
# ---------------------------------------------------------------------------


class ReadOnlyStore(FeatureStore):
    """Minimal store that only implements read — no write/exists/delete."""

    def __init__(self, data: dict):
        self._data = data

    def read(self, feature, entity_id):
        key = (type(feature).__name__, entity_id)
        if key not in self._data:
            raise KeyError(entity_id)
        return self._data[key]


@pytest.mark.asyncio
async def test_read_only_store_can_be_instantiated():
    """A store that only overrides read() should be constructable."""
    store = ReadOnlyStore({("DummyFeature", "e1"): {"v": 42}})
    feature = DummyFeature()
    result = await store.aread(feature, "e1")
    assert result == {"v": 42}


@pytest.mark.asyncio
async def test_read_only_store_write_raises_not_implemented():
    store = ReadOnlyStore({})
    feature = DummyFeature()
    with pytest.raises(NotImplementedError, match="write"):
        await store.awrite(feature, "e1", ExtractionResult.of("e1", {"v": 1}))


@pytest.mark.asyncio
async def test_read_only_store_exists_raises_not_implemented():
    store = ReadOnlyStore({})
    feature = DummyFeature()
    with pytest.raises(NotImplementedError, match="exists"):
        await store.aexists(feature, "e1")


@pytest.mark.asyncio
async def test_read_only_store_delete_raises_not_implemented():
    store = ReadOnlyStore({})
    feature = DummyFeature()
    with pytest.raises(NotImplementedError, match="delete"):
        await store.adelete(feature, "e1")
