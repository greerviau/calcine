"""Tests for fan-out extraction (extract_many / FanOutResult)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from calcine import FanOutResult, Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources.base import DataSource
from calcine.stores.memory import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class SimpleSource(DataSource):
    """Returns a dict with `segments` list keyed by entity_id."""

    def __init__(self, data: dict):
        self._data = data

    async def read(self, entity_id: str, context: dict) -> dict:
        if entity_id not in self._data:
            raise KeyError(entity_id)
        return self._data[entity_id]


class SegmentFeature(Feature):
    """Fan-out: one recording → N segment records with parent metadata."""

    parent_schema = FeatureSchema(
        {
            "sample_rate": types.Int64(nullable=False),
            "speaker_id": types.String(nullable=True),
        }
    )
    schema = FeatureSchema(
        {
            "rms": types.Float64(nullable=False),
            "duration_ms": types.Float64(nullable=False),
        }
    )

    async def extract_many(self, raw: dict, context: dict, entity_id: str) -> FanOutResult:
        return FanOutResult(
            metadata={"sample_rate": raw["sample_rate"], "speaker_id": raw.get("speaker_id")},
            records={
                f"{entity_id}/{i}": {"rms": seg["rms"], "duration_ms": seg["duration_ms"]}
                for i, seg in enumerate(raw["segments"])
            },
        )

    async def extract(self, raw, context, entity_id=None):
        raise NotImplementedError  # pipeline always uses extract_many


class NoMetaSegmentFeature(Feature):
    """Fan-out without parent metadata."""

    schema = FeatureSchema({"value": types.Float64(nullable=False)})

    async def extract_many(self, raw: dict, context: dict, entity_id: str) -> FanOutResult:
        return FanOutResult(
            records={f"{entity_id}/{i}": {"value": v} for i, v in enumerate(raw["values"])}
        )

    async def extract(self, raw, context, entity_id=None):
        raise NotImplementedError


class BadMetaFeature(Feature):
    """Fan-out that returns invalid parent metadata."""

    parent_schema = FeatureSchema({"count": types.Int64(nullable=False)})
    schema = FeatureSchema({"v": types.Float64(nullable=False)})

    async def extract_many(self, raw: dict, context: dict, entity_id: str) -> FanOutResult:
        return FanOutResult(
            metadata={"count": "not-an-int"},  # should fail validation
            records={f"{entity_id}/0": {"v": 1.0}},
        )

    async def extract(self, raw, context, entity_id=None):
        raise NotImplementedError


class BadRecordFeature(Feature):
    """Fan-out that returns invalid sub-entity records."""

    schema = FeatureSchema({"v": types.Float64(nullable=False)})

    async def extract_many(self, raw: dict, context: dict, entity_id: str) -> FanOutResult:
        return FanOutResult(
            records={
                f"{entity_id}/0": {"v": 1.0},
                f"{entity_id}/1": {"v": "not-a-float"},  # invalid
            }
        )

    async def extract(self, raw, context, entity_id=None):
        raise NotImplementedError


class RaisingFeature(Feature):
    """Fan-out that raises during extract_many."""

    async def extract_many(self, raw: dict, context: dict, entity_id: str) -> FanOutResult:
        raise ValueError("boom")

    async def extract(self, raw, context, entity_id=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FanOutResult dataclass
# ---------------------------------------------------------------------------


def test_fanout_result_with_metadata():
    r = FanOutResult(records={"a/0": {"x": 1}}, metadata={"key": "val"})
    assert r.records == {"a/0": {"x": 1}}
    assert r.metadata == {"key": "val"}


def test_fanout_result_without_metadata():
    r = FanOutResult(records={"a/0": {"x": 1}})
    assert r.metadata is None


# ---------------------------------------------------------------------------
# Feature.extract_many detection
# ---------------------------------------------------------------------------


def test_extract_many_detection_on_subclass():
    """Subclass that overrides extract_many is detectable via MRO check."""
    feat = SegmentFeature()
    assert type(feat).extract_many is not Feature.extract_many


def test_extract_many_detection_on_base():
    """Base Feature.extract_many points to the base implementation."""

    class PlainFeature(Feature):
        async def extract(self, raw, context, entity_id=None):
            return raw

    feat = PlainFeature()
    assert type(feat).extract_many is Feature.extract_many


# ---------------------------------------------------------------------------
# Pipeline: fan-out generate
# ---------------------------------------------------------------------------


SOURCE_DATA = {
    "rec1": {
        "sample_rate": 16000,
        "speaker_id": "alice",
        "segments": [
            {"rms": 0.1, "duration_ms": 100.0},
            {"rms": 0.2, "duration_ms": 200.0},
        ],
    },
    "rec2": {
        "sample_rate": 44100,
        "speaker_id": None,
        "segments": [
            {"rms": 0.5, "duration_ms": 50.0},
        ],
    },
}


@pytest.mark.asyncio
async def test_fanout_basic_generate():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1", "rec2"])

    assert report.success_count == 2
    assert report.failure_count == 0

    # Parent metadata stored under the source entity_id
    meta1 = await store.aread(SegmentFeature(), "rec1")
    assert meta1 == {"sample_rate": 16000, "speaker_id": "alice"}

    meta2 = await store.aread(SegmentFeature(), "rec2")
    assert meta2 == {"sample_rate": 44100, "speaker_id": None}

    # Sub-entity records
    seg0 = await store.aread(SegmentFeature(), "rec1/0")
    assert seg0 == {"rms": 0.1, "duration_ms": 100.0}

    seg1 = await store.aread(SegmentFeature(), "rec1/1")
    assert seg1 == {"rms": 0.2, "duration_ms": 200.0}

    seg_r2 = await store.aread(SegmentFeature(), "rec2/0")
    assert seg_r2 == {"rms": 0.5, "duration_ms": 50.0}


@pytest.mark.asyncio
async def test_fanout_report_stores_fanout_result():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1"])
    assert isinstance(report.succeeded["rec1"], FanOutResult)
    assert "rec1/0" in report.succeeded["rec1"].records


@pytest.mark.asyncio
async def test_fanout_store_results_false():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1"], store_results=False)
    assert report.succeeded["rec1"] is None
    # Data is still written to the store
    assert await store.aexists(SegmentFeature(), "rec1")


@pytest.mark.asyncio
async def test_fanout_no_metadata():
    data = {"e1": {"values": [1.0, 2.0, 3.0]}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=NoMetaSegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.success_count == 1

    # Tombstone written under parent key
    parent_val = await store.aread(NoMetaSegmentFeature(), "e1")
    assert parent_val == {}  # sentinel for overwrite=False

    sub0 = await store.aread(NoMetaSegmentFeature(), "e1/0")
    assert sub0 == {"value": 1.0}


# ---------------------------------------------------------------------------
# Pipeline: overwrite=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_overwrite_false_skips_existing():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)

    # First run
    report1 = await pipeline.agenerate(entity_ids=["rec1"], overwrite=False)
    assert report1.success_count == 1
    assert report1.skip_count == 0

    # Second run — parent entity exists; should skip
    report2 = await pipeline.agenerate(entity_ids=["rec1"], overwrite=False)
    assert report2.skip_count == 1
    assert report2.success_count == 0


@pytest.mark.asyncio
async def test_fanout_overwrite_true_rewrites():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)

    await pipeline.agenerate(entity_ids=["rec1"])
    report2 = await pipeline.agenerate(entity_ids=["rec1"], overwrite=True)
    assert report2.success_count == 1
    assert report2.skip_count == 0


# ---------------------------------------------------------------------------
# Pipeline: validation failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_bad_parent_metadata_fails_entity():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=BadMetaFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.failure_count == 1
    assert report.success_count == 0
    assert "e1" in report.failed


@pytest.mark.asyncio
async def test_fanout_bad_record_fails_entity():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=BadRecordFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.failure_count == 1
    errors = report.failed["e1"]
    # Error messages should reference the bad sub-entity
    assert any("e1/1" in e for e in errors)


@pytest.mark.asyncio
async def test_fanout_extract_many_raises_is_caught():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=RaisingFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.failure_count == 1
    assert "boom" in report.failed["e1"][0]


# ---------------------------------------------------------------------------
# MemoryStore: alist_entities / list_entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alist_entities_all():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)
    await pipeline.agenerate(entity_ids=["rec1", "rec2"])

    entities = await store.alist_entities(feature)
    assert set(entities) == {"rec1", "rec1/0", "rec1/1", "rec2", "rec2/0"}


@pytest.mark.asyncio
async def test_alist_entities_with_prefix():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)
    await pipeline.agenerate(entity_ids=["rec1", "rec2"])

    sub_ids = await store.alist_entities(feature, prefix="rec1/")
    assert set(sub_ids) == {"rec1/0", "rec1/1"}


@pytest.mark.asyncio
async def test_alist_entities_empty_store():
    store = MemoryStore()
    feature = SegmentFeature()
    assert await store.alist_entities(feature) == []


def test_list_entities_sync():
    store = MemoryStore()
    feature = NoMetaSegmentFeature()
    store.write(feature, "e1/0", {"value": 1.0})
    store.write(feature, "e1/1", {"value": 2.0})

    ids = store.list_entities(feature, prefix="e1/")
    assert set(ids) == {"e1/0", "e1/1"}


# ---------------------------------------------------------------------------
# FeatureStore.awrite_fanout default behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_fanout_writes_metadata_and_records():
    store = MemoryStore()
    feature = SegmentFeature()
    result = FanOutResult(
        metadata={"sample_rate": 8000, "speaker_id": "bob"},
        records={
            "r/0": {"rms": 0.3, "duration_ms": 10.0},
            "r/1": {"rms": 0.4, "duration_ms": 20.0},
        },
    )
    await store.awrite_fanout(feature, "r", result)

    assert await store.aread(feature, "r") == {"sample_rate": 8000, "speaker_id": "bob"}
    assert await store.aread(feature, "r/0") == {"rms": 0.3, "duration_ms": 10.0}
    assert await store.aread(feature, "r/1") == {"rms": 0.4, "duration_ms": 20.0}


@pytest.mark.asyncio
async def test_write_fanout_none_metadata_writes_tombstone():
    store = MemoryStore()
    feature = NoMetaSegmentFeature()
    result = FanOutResult(records={"e/0": {"value": 5.0}})
    await store.awrite_fanout(feature, "e", result)

    assert await store.aread(feature, "e") == {}
    assert await store.aexists(feature, "e")


# ---------------------------------------------------------------------------
# Concurrency and batch_size: fan-out routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_concurrent_generate():
    data = {
        f"r{i}": {
            "sample_rate": 100,
            "speaker_id": None,
            "segments": [{"rms": float(i), "duration_ms": 1.0}],
        }
        for i in range(10)
    }
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=list(data.keys()), concurrency=5)
    assert report.success_count == 10
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_fanout_batch_size_falls_back_to_per_entity():
    """Fan-out features ignore batch_size and still use extract_many per entity."""
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1", "rec2"], batch_size=4)
    assert report.success_count == 2
    assert report.failure_count == 0

    # Sub-entities must still be written correctly
    assert await store.aexists(SegmentFeature(), "rec1/0")
    assert await store.aexists(SegmentFeature(), "rec2/0")


@pytest.mark.asyncio
async def test_fanout_batch_size_with_concurrency():
    """batch_size + concurrency together still route fan-out correctly."""
    data = {
        f"r{i}": {
            "sample_rate": 100,
            "speaker_id": None,
            "segments": [{"rms": float(i), "duration_ms": 1.0}],
        }
        for i in range(8)
    }
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=list(data.keys()), batch_size=4, concurrency=2)
    assert report.success_count == 8
    assert report.failure_count == 0


# ---------------------------------------------------------------------------
# Executor: fan-out with ThreadPoolExecutor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_executor_success():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    with ThreadPoolExecutor(max_workers=2) as ex:
        report = await pipeline.agenerate(entity_ids=["rec1", "rec2"], executor=ex, concurrency=2)

    assert report.success_count == 2
    assert report.failure_count == 0
    assert await store.aexists(SegmentFeature(), "rec1/0")
    assert await store.aexists(SegmentFeature(), "rec2/0")


@pytest.mark.asyncio
async def test_fanout_executor_validation_failure():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=BadRecordFeature(),
        store=store,
    )
    with ThreadPoolExecutor(max_workers=1) as ex:
        report = await pipeline.agenerate(entity_ids=["e1"], executor=ex)

    assert report.failure_count == 1
    assert any("e1/1" in e for e in report.failed["e1"])


@pytest.mark.asyncio
async def test_fanout_executor_exception_caught():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=RaisingFeature(),
        store=store,
    )
    with ThreadPoolExecutor(max_workers=1) as ex:
        report = await pipeline.agenerate(entity_ids=["e1"], executor=ex)

    assert report.failure_count == 1
    assert "boom" in report.failed["e1"][0]
