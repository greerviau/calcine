"""Tests for Pipeline and GenerationReport."""

from __future__ import annotations

import pandas as pd
import pytest
from stratum import GenerationReport, Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.sources import DataFrameSource
from stratum.stores import MemoryStore

# ---------------------------------------------------------------------------
# Helper feature implementations
# ---------------------------------------------------------------------------


class MeanFeature(Feature):
    schema = FeatureSchema({"mean_value": types.Float64(nullable=False)})

    async def extract(self, raw: pd.DataFrame, context: dict) -> dict:
        if raw.empty:
            raise ValueError("No rows for entity")
        return {"mean_value": float(raw["amount"].mean())}


class FailingFeature(Feature):
    async def extract(self, raw, context):
        raise RuntimeError("Intentional extraction failure")


class HookTrackingFeature(Feature):
    def __init__(self) -> None:
        self.pre_called = False
        self.post_called = False

    async def pre_extract(self, raw):
        self.pre_called = True
        return raw

    async def extract(self, raw, context):
        return {"value": 42.0}

    async def post_extract(self, result):
        self.post_called = True
        return result


@pytest.fixture
def df():
    return pd.DataFrame({"entity_id": ["u1", "u1", "u2"], "amount": [10.0, 20.0, 15.0]})


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_report(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1", "u2"])

    assert isinstance(report, GenerationReport)
    assert report.success_count == 2
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_generate_correct_values(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1", "u2"])

    # u1 has rows [10, 20] → mean 15; u2 has [15] → mean 15
    assert report.succeeded["u1"]["mean_value"] == 15.0
    assert report.succeeded["u2"]["mean_value"] == 15.0


@pytest.mark.asyncio
async def test_generate_never_raises_on_entity_failure(df):
    """A failing feature should not crash generate() — it collects errors."""
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=FailingFeature(),
        store=MemoryStore(),
    )
    # Must not raise
    report = await pipeline.generate(entity_ids=["u1", "u2"])

    assert report.failure_count == 2
    assert report.success_count == 0
    assert "u1" in report.failed
    assert "u2" in report.failed


@pytest.mark.asyncio
async def test_generate_missing_entity_does_not_crash(df):
    """An entity not in the source is captured as a failure, not a crash."""
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1", "u_missing"])

    assert "u1" in report.succeeded
    assert "u_missing" in report.failed


@pytest.mark.asyncio
async def test_generate_schema_violation_captured_as_failure(df):
    """Schema violations should be recorded in failed, not raise."""

    class BadFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw, context):
            return {"score": "not_a_float"}  # wrong type

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=BadFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1"])

    assert "u1" in report.failed
    assert len(report.failed["u1"]) > 0


@pytest.mark.asyncio
async def test_generate_default_context_is_empty(df):
    """generate() should work without an explicit context argument."""
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1"])
    assert "u1" in report.succeeded


@pytest.mark.asyncio
async def test_generate_context_forwarded_to_extract(df):
    received: dict = {}

    class ContextCapture(Feature):
        async def extract(self, raw, context):
            received.update(context)
            return {"v": 1.0}

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=ContextCapture(),
        store=MemoryStore(),
    )
    ctx = {"version": "v2", "ts": 999}
    await pipeline.generate(entity_ids=["u1"], context=ctx)

    assert received == ctx


@pytest.mark.asyncio
async def test_generate_hooks_called(df):
    feature = HookTrackingFeature()
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=feature,
        store=MemoryStore(),
    )
    await pipeline.generate(entity_ids=["u1"])

    assert feature.pre_called
    assert feature.post_called


# ---------------------------------------------------------------------------
# retrieve() / retrieve_batch() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_after_generate(df):
    store = MemoryStore()
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=store,
    )
    await pipeline.generate(entity_ids=["u1"])

    result = await pipeline.retrieve("u1")
    assert result["mean_value"] == 15.0


@pytest.mark.asyncio
async def test_retrieve_missing_raises_key_error(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    with pytest.raises(KeyError):
        await pipeline.retrieve("nonexistent")


@pytest.mark.asyncio
async def test_retrieve_batch(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    await pipeline.generate(entity_ids=["u1", "u2"])

    results = await pipeline.retrieve_batch(["u1", "u2", "u_none"])
    assert "u1" in results
    assert "u2" in results
    assert "u_none" not in results  # missing silently omitted


# ---------------------------------------------------------------------------
# GenerationReport tests
# ---------------------------------------------------------------------------


def test_report_counts():
    report = GenerationReport(
        succeeded={"a": 1, "b": 2},
        failed={"c": ["err"]},
    )
    assert report.success_count == 2
    assert report.failure_count == 1


def test_report_repr():
    report = GenerationReport(succeeded={"a": 1}, failed={})
    assert "succeeded=1" in repr(report)
    assert "failed=0" in repr(report)
