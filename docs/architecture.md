# Architecture

This document explains the core design decisions in calcine and the
reasoning behind them.

---

## The three-part abstraction

```
DataSource ──► Feature ──► FeatureStore
                 │
             GenerationReport
```

Each component has a single, narrow responsibility:

| Component | Responsibility |
|-----------|---------------|
| `DataSource` | Fetching raw data for a given entity |
| `Feature` | Transforming raw data into a typed, validated value |
| `FeatureStore` | Persisting and retrieving feature values |

`Pipeline` is a thin coordinator — it calls these three components in sequence
and collects results. It contains no domain logic of its own.

The clearest way to think about which concern belongs where:

| Layer | Owns |
|-------|------|
| `Feature` + `FeatureSchema` | **What** the data is — shape, types, valid values |
| `Pipeline` | **Which** entities are processed — the entity_id contract |
| `FeatureStore` | **How** data is physically stored — internal representation |

---

## Async convention

calcine uses sync as the default public interface. The overwhelming majority of
calcine usage — batch jobs, training scripts, Airflow DAGs, DataLoader workers —
is synchronous. Async is the right internal mechanism for concurrency and
fan-out I/O, but should not be imposed on callers who don't need it.

**Convention:** sync methods are the default; async variants are opt-in using
the `a`-prefix:

```python
# Sync — default, works everywhere
report = pipeline.generate(entity_ids=ids, concurrency=32)
value  = store.read(MyFeature, entity_id)
values = store.read_many(MyFeature, entity_ids)

# Async — opt-in for callers already in an async context
report = await pipeline.agenerate(entity_ids=ids, concurrency=32)
value  = await store.aread(MyFeature, entity_id)
```

Async remains first-class where it genuinely matters: `SourceBundle` uses
`asyncio.gather` to fan out concurrent I/O across sources, and async serving
contexts (FastAPI, async task workers) use the `a`-prefixed methods naturally.
The internal implementation is async throughout; this is purely a public API
ergonomics decision.

---

## Why `raw` is `Any`

`Feature.extract(raw, context)` takes `raw: Any`. This is intentional.

The type of `raw` is entirely determined by the `DataSource` — a
`DataFrameSource` produces a `pd.DataFrame`, a `FileSource` produces `bytes`,
and a `SourceBundle` produces a `dict`. There is no single type that covers all
of these, and constraining it would either force every Feature to accept `Any`
anyway or require awkward generic parameterisation.

The schema system (`FeatureSchema`) handles output typing rigorously. Input
typing is left to the Feature author, who always knows what source they pair
with.

---

## The entity_id contract

The `entity_id` is an **external contract that is stable across the full
lifecycle** — generation, storage, retrieval, and reporting. Whatever string
you pass to `pipeline.generate(entity_ids=[...])` is the same string you use
to retrieve from the store.

This contract is what makes the system predictable for multiple consumers.
A training job and an inference handler both call
`store.read(MyFeature, "user_42")` and get the same thing. The
`GenerationReport` reports success and failure by entity_id. `overwrite=False`
checks existence by entity_id.

**What stores can do internally:** A store can use integer primary keys, shard
across files, compress data, partition into multiple tables, or represent the
value in any format — as long as `store.read(feature, entity_id)` returns the
same logical value that was validated and written. Internal representation is
an implementation detail; the entity_id interface is not.

**What stores must not do:** Invent or remap entity_ids silently. If the store
changes the keys under which data is addressable, the generation report becomes
meaningless, `overwrite=False` breaks, and consumers disagree on what ID refers
to what data.

**Fan-out and sub-entity IDs:** When `Feature.extract_many()` is used, the
feature itself defines the sub-entity ID scheme (e.g. `f"{entity_id}/{i}"`).
These IDs are part of the feature's documented contract, not an internal store
detail. See [Fan-out extraction](#fan-out-extraction) below.

---

## SourceBundle design

`SourceBundle` is itself a `DataSource` — it composes at the source layer
rather than at the pipeline layer. This keeps `Pipeline.__init__` unchanged
(still `source: DataSource`) and means bundles can be nested or mixed freely.

Sources in a bundle are read **concurrently** via `asyncio.gather`. This
matters for sources that do real I/O (HTTP, database) where waiting for each
source serially would be wasteful. This is one of the cases where async is
load-bearing — the concurrency is the point.

The downside is that a failure in any sub-source fails the whole bundle for
that entity. See [Weak Point C in the examples](../examples/05_weak_points.py)
for the mitigation pattern, or use `fault_tolerant=True` once that option is
implemented.

---

## Schema design

The schema system is deliberately simple:

- `FeatureSchema` holds a `dict[str, FeatureType]`
- Validation returns `list[str]` — never raises
- Schema failures produce entries in `report.failed`, not exceptions

This means the pipeline never aborts due to a schema violation. Partial output
(valid entities) is always captured even when some entities fail validation.

**Schema as a two-way contract.** The schema governs both directions:
- **On write** — `extract()` output is validated before reaching the store;
  invalid values never enter the store
- **On read** — retrieved values are validated against the same schema; data
  that doesn't conform (written by an old version, corrupted, migrated
  incorrectly) fails loudly at the access point

This is what makes the schema more than just input validation — it is the
shared contract between everyone who touches the feature.

**Single-field schema for non-dict features:**
When a `Feature` returns a raw value (e.g. a numpy array), the schema can
validate it directly using a single-field schema. The field name is arbitrary;
only its type validator matters:

```python
schema = FeatureSchema({"_vec": types.NDArray(shape=(128,), dtype="float32")})
errors = schema.validate(arr)   # validates arr directly, not a dict
```

**Fan-out features** use two schemas: `schema` validates each sub-entity
record; `parent_schema` validates the optional shared parent metadata. See
[Fan-out extraction](#fan-out-extraction).

---

## Fan-out extraction

Standard extraction is 1:1: one entity_id produces one stored value.
Fan-out extraction is 1:many: one source entity produces multiple
independently-stored sub-entity records.

Implement `extract_many()` instead of `extract()` to opt in:

```python
@dataclass
class FanOutResult:
    records:  dict[str, Any]              # sub_id → value; validated against schema
    metadata: dict[str, Any] | None = None  # parent-level; validated against parent_schema
```

```python
class AudioSegmentFeature(Feature):
    parent_schema = FeatureSchema({       # optional; for shared source-level metadata
        "sample_rate": types.Int64(nullable=False),
        "speaker_id":  types.String(nullable=True),
    })
    schema = FeatureSchema({              # validates each record
        "mfcc": types.NDArray(shape=(13,), dtype="float32"),
        "rms":  types.Float64(nullable=False),
    })

    async def extract_many(
        self, raw: bytes, context: dict, entity_id: str
    ) -> FanOutResult:
        header   = parse_header(raw)
        segments = segment_audio(raw, window_ms=100)
        return FanOutResult(
            metadata={
                "sample_rate": header.sample_rate,
                "speaker_id":  header.speaker_id,
            },
            records={
                f"{entity_id}/{i}": {"mfcc": compute_mfcc(s), "rms": compute_rms(s)}
                for i, s in enumerate(segments)
            },
        )
```

**Why fan-out belongs in the Feature, not the Store:**
The sub-entity ID scheme (`recording_001/0`, `recording_001/1`, ...) is part
of the feature's contract. Any consumer — training job, inference handler,
debugging script — needs to know these IDs to retrieve data. If the store
invented the IDs internally, the ID scheme would be an undocumented
implementation detail and would differ between store implementations, breaking
portability and consumer consistency.

**Store key convention:**
```
("AudioSegmentFeature", "recording_001")    ← parent metadata
("AudioSegmentFeature", "recording_001/0")  ← sub-entity records
("AudioSegmentFeature", "recording_001/1")
```

Stores may override `write_fanout(feature, entity_id, result)` to write
parent metadata and sub-entity records atomically (e.g. in a single SQL
transaction). The default implementation calls `write()` for each entry
individually.

---

## Error handling philosophy

calcine distinguishes these categories of failure:

| Error type | Cause | Pipeline behaviour |
|-----------|-------|--------------------|
| `SourceError` | I/O or missing data in the source | Entity goes to `report.failed` |
| Schema violation | `Feature.validate` returns non-empty list | Entity goes to `report.failed` |
| `StoreError` | Persistent storage failure | Entity goes to `report.failed` |
| Unhandled exception | Bug in Feature or unexpected error | Entity goes to `report.failed` |

`Pipeline.generate()` **never raises**. The caller can inspect
`report.failed` to decide whether to retry, alert, or discard.

For fan-out features, the source entity is the unit of reporting. Sub-entity
failures are nested under the source entity in `report.failures` so the caller
can see exactly which segments or chunks failed without losing the parent
context.

---

## Store key design

By default, stores use `type(feature).__name__` as the namespace key. This is
simple and works well for most use cases.

For multi-team or multi-module codebases where class name collisions are
possible, override `_feature_key` in a store subclass (see
[Weak Point D in the examples](../examples/05_weak_points.py)).

For fan-out sub-entities, the feature itself is responsible for defining the
sub-entity ID format. The convention `{parent_id}/{index_or_label}` is
recommended because it makes the hierarchy self-evident in the store and
enables prefix-based listing:

```python
sub_ids  = store.list_entities(AudioSegmentFeature, prefix="recording_001/")
segments = store.read_many(AudioSegmentFeature, sub_ids)
```

---

## Serializers

Serializers are an implementation detail of `FileStore` — they are not part of
the `FeatureStore` interface. This means:

- `MemoryStore` and `ParquetStore` have no serializer concept
- `FileStore` can accept any `Serializer` without changing the `FeatureStore` API
- Users can add custom serializers (e.g. MessagePack, Arrow IPC) without
  touching any framework code

---

## Known limitations

See [`examples/05_weak_points.py`](../examples/05_weak_points.py) for
executable demonstrations and workarounds for:

- Empty `DataFrameSource` results silently reaching `extract()`
- `SourceBundle` all-or-nothing failure semantics (mitigated by `fault_tolerant=True` once implemented)
- Feature class-name collisions in stores
