# calcine

[![CI](https://github.com/greerviau/calcine/actions/workflows/ci.yml/badge.svg)](https://github.com/greerviau/calcine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A source-agnostic, type-agnostic featurization pipeline framework for Python.

```
DataSource  ──►  Feature  ──►  FeatureStore
```

calcine gives you a clean three-part abstraction for building reproducible,
validated feature extraction pipelines — over any data source and any storage
backend, with no lock-in on format or framework.

---

## Highlights

- **Sync-first interface** — `pipeline.generate()` and `store.read()` are plain synchronous calls by default; async variants (`agenerate`, `aread`, …) are opt-in for callers already in an async context
- **Type-safe schemas** — validate floats, ints, strings, booleans, categoricals,
  ndarrays, and bytes before anything hits the store
- **Composable sources** — `SourceBundle` fans out to multiple sources concurrently;
  combine DataFrames, files, HTTP, or anything custom
- **Batteries included** — file, DataFrame, and HTTP sources out of the box;
  memory, file, and Parquet stores; pluggable serializers
- **Fan-out extraction** — `extract()` returns an `ExtractionResult` with one or
  more records; parent metadata and sub-entities are stored independently
  and discovered via `store.list_entities(feature, prefix="…")`
- **Never crashes on partial failure** — `generate()` captures every per-entity
  error in a `GenerationReport`; valid entities are always written
- **No required pandas** — the core only needs `numpy`; pandas is optional

---

## Installation

```bash
pip install calcine                  # core (numpy only)
pip install "calcine[http]"          # + async HTTP source
pip install "calcine[parquet]"       # + Parquet store
pip install "calcine[dev]"           # + test/lint tools
```

---

## Quick start

```python
from calcine import ExtractionResult, Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources.base import DataSource
from calcine.stores import MemoryStore


# --- 1. Define your data source (any async I/O: DB, API, S3, …) ---

class UserDBSource(DataSource):
    async def read(self, entity_id: str, **kwargs) -> dict:
        return await db.fetch_user(entity_id)  # your async call here


# --- 2. Define a schema-validated feature ---

class UserEngagementFeature(Feature):
    schema = FeatureSchema({
        "spend_tier":  types.Category(categories=["low", "mid", "high", "whale"]),
        "event_rate":  types.Float64(nullable=False),
        "total_spend": types.Float64(nullable=False),
    })

    async def extract(self, raw: dict, context: dict, entity_id=None) -> ExtractionResult:
        spend = raw["total_spend"]
        if spend < 100:    tier = "low"
        elif spend < 1000: tier = "mid"
        elif spend < 3000: tier = "high"
        else:              tier = "whale"
        return ExtractionResult.of(entity_id, {
            "spend_tier":  tier,
            "event_rate":  raw["event_count"] / raw["days_active"],
            "total_spend": spend,
        })


# --- 3. Build and run ---

pipeline = Pipeline(
    source=UserDBSource(),
    feature=UserEngagementFeature(),
    store=MemoryStore(),
)

# All 1 000 reads fire concurrently; failures are isolated per entity
report = pipeline.generate(entity_ids=user_ids, concurrency=32)
print(report)   # GenerationReport(succeeded=997, failed=3, skipped=0)

# Re-run later — already-stored entities are skipped automatically
pipeline.generate(entity_ids=new_user_ids, overwrite=False)

value = pipeline.retrieve("u42")
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a fully runnable version with a simulated async source, bad-data handling, and incremental generation.

---

## Multiple sources with SourceBundle

When your feature needs data from more than one place, compose sources with
`SourceBundle`. All sources are read concurrently; `Feature.extract` receives
a plain `dict` keyed by whatever names you choose:

```python
from calcine.sources import SourceBundle

pipeline = Pipeline(
    source=SourceBundle(
        transactions=TransactionSource(),
        profile=ProfileSource(),
        embeddings=EmbeddingSource(),
    ),
    feature=MyFeature(),
    store=MemoryStore(),
)


class MyFeature(Feature):
    async def extract(self, raw: dict, context: dict, entity_id=None) -> ExtractionResult:
        txns = raw["transactions"]
        prof = raw["profile"]
        embs = raw["embeddings"]
        ...
```

No assumptions are made about what the sources represent or how they relate.

---

## Fan-out extraction

When one source entity produces multiple independently-stored sub-entity records
(audio → segments, document → chunks, session → events), return an
`ExtractionResult` with multiple records from `extract`:

```python
from calcine import ExtractionResult

class AudioSegmentFeature(Feature):
    parent_schema = FeatureSchema({
        "sample_rate": types.Int64(nullable=False),
        "speaker_id":  types.String(nullable=True),
    })
    schema = FeatureSchema({
        "rms": types.Float64(nullable=False),
    })

    async def extract(self, raw: bytes, context: dict, entity_id: str | None = None) -> ExtractionResult:
        segments = split_audio(raw)
        return ExtractionResult(
            metadata={"sample_rate": 16000, "speaker_id": "alice"},
            records={f"{entity_id}/{i}": {"rms": rms(s)} for i, s in enumerate(segments)},
        )


report = pipeline.generate(entity_ids=recording_ids)

# Retrieve parent metadata and sub-entity records
meta     = store.read(feature, "recording_001")
sub_ids  = store.list_entities(feature, prefix="recording_001/")
segments = [store.read(feature, sid) for sid in sub_ids]
```

`ExtractionResult.of(entity_id, value)` is a convenience constructor for
single-record features. For fan-out, pass `records` directly with sub-entity IDs.
Parent metadata and sub-entity records are stored under separate keys;
`overwrite=False` skips the source entity if its parent key already exists.

---

## Schema system

```python
from calcine.schema import FeatureSchema, types

schema = FeatureSchema({
    "score":     types.Float64(nullable=False, default=0.0),
    "category":  types.Category(categories=["low", "mid", "high"]),
    "embedding": types.NDArray(shape=(None, 128), dtype="float32"),
    "label":     types.String(nullable=True),
    "active":    types.Boolean(),
    "count":     types.Int64(nullable=False),
    "payload":   types.Bytes(),
    "anything":  types.Any(),
})
```

For non-dict features (e.g. raw arrays), use a single-field schema:

```python
schema = FeatureSchema({"_vec": types.NDArray(shape=(128,), dtype="float32")})
errors = schema.validate(arr)   # validates the array directly
```

See [`docs/schema.md`](docs/schema.md) for the full reference.

---

## Available components

### Sources

| Class | Description |
|-------|-------------|
| `FileSource(path)` | Read bytes from a single file |
| `DirectorySource(path, pattern)` | Stream bytes from files matching a glob |
| `DataFrameSource(df, entity_col)` | Filter a pandas DataFrame by entity |
| `HTTPSource(url_template, ...)` | Async HTTP GET (requires `[http]`) |
| `SourceBundle(**sources)` | Read multiple sources concurrently |

### Stores

| Class | Description |
|-------|-------------|
| `MemoryStore` | Dict-backed, no I/O — for tests and prototyping |
| `FileStore(path, serializer)` | One file per entity per feature |
| `ParquetStore(path)` | Parquet files partitioned by feature (requires `[parquet]`) |

### Serializers (for FileStore)

| Class | Best for |
|-------|----------|
| `PickleSerializer` | Any Python object (default) |
| `JSONSerializer` | Dict / list / primitive features |
| `NumpySerializer` | `numpy` arrays |

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/architecture.md`](docs/architecture.md) | Design rationale and core decisions |
| [`docs/extending.md`](docs/extending.md) | Adding custom sources, stores, and schema types |
| [`docs/schema.md`](docs/schema.md) | Full schema type reference |
| [`examples/README.md`](examples/README.md) | Guide to the runnable examples |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development setup and PR process |

---

## Running the tests

```bash
uv pip install -e ".[dev]"
pytest
```

189 tests, covering the pipeline, schema, fan-out extraction, all built-in sources and stores.

---

## Project layout

```
calcine/
├── pipeline.py        Pipeline + GenerationReport
├── extraction.py      ExtractionResult
├── schema.py          FeatureSchema + type system
├── serializers.py     Serializer ABC + Pickle / JSON / Numpy
├── exceptions.py      SourceError, StoreError, SchemaViolationError
├── sources/           DataSource ABC + FileSource, DataFrameSource, HTTPSource, SourceBundle
├── features/          Feature ABC (extract)
└── stores/            FeatureStore ABC + MemoryStore, FileStore, ParquetStore

tests/                 189 tests mirroring the calcine structure
examples/              5 runnable end-to-end scripts + generated datasets
docs/                  Architecture, extension guide, schema reference
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All contributions are welcome —
new sources, stores, schema types, bug fixes, and documentation improvements.

## License

[MIT](LICENSE)
