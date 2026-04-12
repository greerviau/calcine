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

**Key features:**
- **Pipeline orchestration:** Concurrent entity processing with per-entity error isolation, incremental generation, and partition-by support
- **Type-safe schemas:** Validate scalars, strings, categoricals, ndarrays, bytes, lists, and dicts on write and read
- **Detailed reporting:** `GenerationReport` tracks successes, failures, and skips; `timing_summary()` gives p50/p95/max per phase (read / extract / write); exports to a pandas DataFrame
- **Fan-out extraction:** `extract()` returns one or many records; each sub-entity is stored, validated, and retrievable independently
- **Composable sources:** `SourceBundle` reads from multiple sources concurrently, delivering a single `dict` to `extract`
- **Executor support:** Offload CPU-bound extraction to thread or process pools via `executor=`

## Installation

```bash
pip install calcine
```

## Quick start

```python
import asyncio
import io
from pathlib import Path

import librosa
import numpy as np
import zarr

from calcine import ExtractionResult, Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources.base import DataSource
from calcine.stores.base import FeatureStore


# --- 1. Source: read raw audio from disk, one file per recording ---

class AudioFileSource(DataSource):
    def __init__(self, root: str):
        self._root = Path(root)

    def read(self, entity_id: str, **kwargs) -> bytes:
        path = self._root / f"{entity_id}.wav"
        return path.read_bytes()


# --- 2. Feature: extract log-mel spectrogram + metadata ---

class SpectrogramFeature(Feature):
    schema = FeatureSchema({
        "spectrogram": types.NDArray(shape=(None, 128), dtype="float32"),
        "duration_s":  types.Float64(nullable=False),
        "sample_rate": types.Int64(nullable=False),
    })

    def extract(self, raw: bytes, context: dict, entity_id=None) -> ExtractionResult:
        audio, sr = librosa.load(io.BytesIO(raw), sr=None)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel).T.astype("float32")  # (T, 128)
        return ExtractionResult.of(entity_id, {
            "spectrogram": log_mel,
            "duration_s":  float(len(audio) / sr),
            "sample_rate": int(sr),
        })


# --- 3. Store: write arrays into a Zarr group, ready for training ---

class ZarrStore(FeatureStore):
    def __init__(self, path: str):
        self._root = zarr.open_group(path, mode="a")

    def write(self, feature, entity_id, result, context=None):
        for sub_id, record in result.records.items():
            grp = self._root.require_group(sub_id)
            grp["spectrogram"] = record["spectrogram"]
            grp.attrs.update({k: v for k, v in record.items() if k != "spectrogram"})

    def read(self, feature, entity_id):
        grp = self._root[entity_id]
        return {"spectrogram": grp["spectrogram"][:], **dict(grp.attrs)}

    def exists(self, feature, entity_id) -> bool:
        return entity_id in self._root

    def delete(self, feature, entity_id):
        del self._root[entity_id]


# --- 4. Build and run ---

pipeline = Pipeline(
    source=AudioFileSource("/data/recordings/"),
    feature=SpectrogramFeature(),
    store=ZarrStore("/data/features/spectrograms.zarr"),
)

# Concurrent reads; failures are isolated per recording
report = pipeline.generate(entity_ids=recording_ids, concurrency=8)
print(report)
# GenerationReport(entities=2840, records=2840, failed=12, skipped=0, duration=43.7s)

# Identify bottlenecks across read / extract / write phases
summary = report.timing_summary()
print(f"p95 read:    {summary['read']['p95']*1000:.1f} ms")
print(f"p95 extract: {summary['extract']['p95']*1000:.1f} ms")

# Re-run on new recordings — already-processed ones are skipped automatically
pipeline.generate(entity_ids=new_recording_ids, overwrite=False)
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a fully runnable version with a simulated async source, bad-data handling, and incremental generation.

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
    def extract(self, raw: dict, context: dict, entity_id=None) -> ExtractionResult:
        txns = raw["transactions"]
        prof = raw["profile"]
        embs = raw["embeddings"]
        ...
```

No assumptions are made about what the sources represent or how they relate.

## Fan-out extraction

When one source entity produces multiple independently-stored sub-entity records
(audio → segments, document → chunks, session → events), return an
`ExtractionResult` with multiple records from `extract`:

```python
from calcine import ExtractionResult

class AudioSegmentFeature(Feature):
    metadata_schema = FeatureSchema({
        "sample_rate": types.Int64(nullable=False),
        "speaker_id":  types.String(nullable=True),
    })
    schema = FeatureSchema({
        "rms": types.Float64(nullable=False),
    })

    def extract(self, raw: bytes, context: dict, entity_id: str | None = None) -> ExtractionResult:
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
    "tags":      types.List(item_type=types.String()),
    "scores":    types.Dict(key_type=types.String(), value_type=types.Float64()),
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

## Built-in components

calcine ships with reference sources, stores, and serializers for common patterns.
See [`docs/sources.md`](docs/sources.md) and [`docs/stores.md`](docs/stores.md).

---

## Documentation

See [`docs/`](docs/README.md) for the full documentation index.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
