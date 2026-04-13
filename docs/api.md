# API reference

This document covers the full public API for calcine's core classes.
For architectural rationale, see [`architecture.md`](architecture.md).
For extending calcine with custom implementations, see [`extending.md`](extending.md).

---

## Pipeline

```python
from calcine import Pipeline
```

`Pipeline` ties a `DataSource`, `Feature`, and `FeatureStore` together into a
generate / retrieve interface.

### Constructor

```python
Pipeline(source: DataSource, feature: Feature, store: FeatureStore)
```

All three arguments are required. The pipeline holds references to each
component but does not validate them at construction time.

---

### generate() / agenerate()

```python
report = pipeline.generate(
    entity_ids=["u1", "u2", "u3"],
    context={"model_version": "v2"},
    concurrency=16,
    overwrite=False,
)

# Async variant — use inside FastAPI handlers or async task workers
report = await pipeline.agenerate(entity_ids=["u1", "u2"], concurrency=8)
```

`generate` is the synchronous default (calls `asyncio.run` internally).
`agenerate` is the async variant for use inside an existing event loop.
All keyword arguments are identical between the two.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_ids` | `list[str]` | — | Flat list of entity IDs to process. Mutually exclusive with `partitions`. |
| `context` | `dict` | `{}` | Shared dict forwarded to every `Feature.extract` call. |
| `context_fn` | `(entity_id) -> dict` | `None` | Per-entity context additions. Merged on top of `context` (entity values shadow shared ones). |
| `partitions` | `dict[str, list[str]]` | `None` | Pre-built partition map. Mutually exclusive with `entity_ids`. |
| `partition_by` | `(entity_id) -> Hashable` | `None` | Groups `entity_ids` into partitions by return value. Requires `entity_ids`; cannot combine with `partitions`. |
| `partition_context_fn` | `(partition_key) -> dict` | `None` | Context additions for all entities in a partition. Merged between `context` and `context_fn`. |
| `concurrency` | `int \| None` | `None` | Max concurrent partitions (or batches in flat mode). When `None` and `executor` is provided, inferred from the executor's worker count. When `None` and no executor, defaults to `1`. |
| `batch_size` | `int` | `1` | Entities per `extract_batch` call. `1` uses the per-entity `extract` path. |
| `overwrite` | `bool` | `True` | When `False`, already-stored entities are skipped without re-extracting. |
| `store_results` | `bool` | `True` | When `False`, `report.succeeded` is empty (saves memory on large runs). |
| `on_progress` | `(completed, total, report) -> None` | `None` | Sync callback invoked after each entity resolves. |
| `executor` | `concurrent.futures.Executor` | `None` | Offloads read + extract to a thread or process pool. |

Returns a [`GenerationReport`](#generationreport).

---

#### Concurrency modes

**Flat mode** (default): `concurrency` caps how many entities run at once.

```python
pipeline.generate(entity_ids=ids, concurrency=32)
```

**Partition function**: entities are grouped by `partition_by`'s return value
and processed serially within each group. `concurrency` caps how many groups
run concurrently. Use this to honour per-account rate limits or enforce
ordered processing within a user session.

```python
# Process up to 8 accounts in parallel; entities within an account are serial
pipeline.generate(
    entity_ids=ids,
    partition_by=lambda eid: eid.split("_")[0],   # group by account prefix
    concurrency=8,
)
```

**Explicit partitions**: supply the mapping directly.

```python
pipeline.generate(
    partitions={"shard_0": shard_0_ids, "shard_1": shard_1_ids},
    concurrency=2,
)
```

When using partitions (either via `partition_by` or `partitions`), the
partition key is injected into every entity's context as
`context["_partition_key"]`.

---

#### Context and per-entity context

`context` is a shared dict forwarded to every `Feature.extract` call.
Use it for things that are constant across all entities: model version,
experiment flags, a timestamp, a shared connection pool.

`context_fn` adds per-entity values on top. The dicts are merged as:

```
{**context, **context_fn(entity_id)}
```

Example: inject each entity's tier from a lookup table.

```python
tiers = {"u1": "premium", "u2": "free", "u3": "premium"}

pipeline.generate(
    entity_ids=list(tiers),
    context={"model_version": "v3"},
    context_fn=lambda eid: {"tier": tiers[eid]},
)

class MyFeature(Feature):
    def extract(self, raw, context, entity_id=None):
        tier = context["tier"]       # "premium" or "free"
        version = context["model_version"]
        ...
```

`partition_context_fn` fills the same role at the partition level, merged
between `context` and `context_fn`.

---

#### Incremental generation

Pass `overwrite=False` to skip entities that already have a stored value.
Skipped entity IDs appear in `report.skipped`.

```python
# First run: generate everything
pipeline.generate(entity_ids=all_ids)

# Second run: only process new entities
pipeline.generate(entity_ids=all_ids + new_ids, overwrite=False)
```

`overwrite=False` calls `store.exists()` for each entity before reading. For
fan-out features, the parent key is what is checked.

---

#### Batch extraction

When `batch_size > 1`, entities are grouped into sub-batches and
`Feature.extract_batch` is called once per batch. This enables vectorised
computation: ML model inference, batch embedding APIs, bulk database queries.

```python
pipeline.generate(entity_ids=ids, batch_size=64, concurrency=4)
```

Override `extract_batch` on your `Feature` to take advantage of this:

```python
class EmbeddingFeature(Feature):
    schema = FeatureSchema({"embedding": types.NDArray(shape=(768,), dtype="float32")})

    def __init__(self, model):
        self.model = model

    def extract(self, raw: str, context, entity_id=None):
        vec = self.model.encode([raw])
        return ExtractionResult.of(entity_id, {"embedding": vec[0]})

    def extract_batch(self, raws, context, entity_ids=None, entity_contexts=None):
        vecs = self.model.encode(raws)
        return [
            ExtractionResult.of(eid, {"embedding": vec})
            for eid, vec in zip(entity_ids, vecs)
        ]
```

For a model with a native async API, override `aextract_batch` instead:

```python
    async def aextract_batch(self, raws, context, entity_ids=None, entity_contexts=None):
        vecs = await self.async_model.encode(raws)
        return [
            ExtractionResult.of(eid, {"embedding": vec})
            for eid, vec in zip(entity_ids, vecs)
        ]

    def extract(self, raw, context, entity_id=None):  # satisfies abstract requirement
        raise NotImplementedError("use aextract_batch")
```

Return one element per input. Individual items can be `BaseException` instances
to signal per-entity failure without aborting the rest of the batch.

---

#### Executor support

Pass a `concurrent.futures.Executor` to offload the read + extract + validate
pipeline stages out of the asyncio event loop. When `concurrency` is not set,
it is inferred automatically from the executor's worker count.

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Thread pool — useful when your feature holds the GIL (numpy, pandas)
with ThreadPoolExecutor(max_workers=8) as pool:
    report = pipeline.generate(entity_ids=ids, executor=pool)

# Process pool — bypasses the GIL for CPU-bound extraction
with ProcessPoolExecutor(max_workers=4) as pool:
    report = pipeline.generate(entity_ids=ids, executor=pool)
```

Store writes always run in the main process, so all store backends (including
`MemoryStore`) work correctly regardless of executor type.

With `ProcessPoolExecutor`, your `DataSource` and `Feature` must be picklable.

You can still override `concurrency` explicitly if you want fewer concurrent
tasks than the pool has workers (e.g. to leave headroom for other work):

```python
with ProcessPoolExecutor(max_workers=8) as pool:
    report = pipeline.generate(entity_ids=ids, executor=pool, concurrency=4)
```

---

#### Progress callbacks

`on_progress` is called after each entity resolves (success, failure, or skip).

```python
from tqdm import tqdm

def make_progress_bar(total):
    bar = tqdm(total=total)
    def on_progress(completed, total, report):
        bar.update(1)
        bar.set_postfix(failed=report.failure_count)
    return on_progress

report = pipeline.generate(
    entity_ids=ids,
    on_progress=make_progress_bar(len(ids)),
)
```

---

### retrieve() / aretrieve()

```python
value = pipeline.retrieve("user_42")
value = await pipeline.aretrieve("user_42")
```

Reads the stored feature value for one entity. Raises `KeyError` if no value
exists, `StoreError` if the read fails.

---

### retrieve_batch() / aretrieve_batch()

```python
values = pipeline.retrieve_batch(["u1", "u2", "u99"])
# {"u1": ..., "u2": ...}  — u99 silently omitted if not stored

values = await pipeline.aretrieve_batch(["u1", "u2"])
```

Reads stored values for multiple entities concurrently. Entities with no stored
value are silently omitted from the result dict.

---

## Feature

```python
from calcine.features.base import Feature
```

### Class attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `schema` | `FeatureSchema \| None` | Validates each record in `ExtractionResult.records`. |
| `metadata_schema` | `FeatureSchema \| None` | Validates `ExtractionResult.metadata` (fan-out features only). |

### extract()

```python
def extract(self, raw: Any, context: dict, entity_id: str | None = None) -> ExtractionResult
```

The method every `Feature` subclass must implement. `raw` is whatever the
`DataSource` returned; `context` is the merged dict from `generate()`; `entity_id`
is the current entity being processed.

Return `ExtractionResult.of(entity_id, value)` for single-record features, or
`ExtractionResult(records={...}, metadata={...})` for fan-out.

The framework calls `aextract` internally, which runs `extract` in a thread
executor by default.

### aextract()

```python
async def aextract(self, raw: Any, context: dict, entity_id: str | None = None) -> ExtractionResult
```

The async variant called by the pipeline. By default wraps `extract` in a thread
executor. Override this instead of `extract` for natively async extraction (async
model clients, async HTTP):

```python
class EmbeddingFeature(Feature):
    async def aextract(self, raw, context, entity_id=None):
        vec = await self.async_client.embed(raw)
        return ExtractionResult.of(entity_id, {"embedding": vec})

    def extract(self, raw, context, entity_id=None):  # satisfies abstract requirement
        raise NotImplementedError("use aextract")
```

### extract_batch()

```python
def extract_batch(
    self,
    raws: list[Any],
    context: dict,
    entity_ids: list[str] | None = None,
    entity_contexts: list[dict] | None = None,
) -> list[ExtractionResult | BaseException]
```

Override for vectorised computation. The default implementation calls `extract`
for each item individually. Return one element per input, in the same order.
Individual failures can be returned as `BaseException` instances.

The framework calls `aextract_batch` internally. Override `aextract_batch`
instead when your batch API is natively async.

### validate()

```python
def validate(self, result: Any) -> list[str]
```

Called automatically by the pipeline after extraction. Uses `schema.validate`
if `schema` is set; otherwise returns an empty list. Override for custom
validation logic beyond field-level schema checks.

---

## ExtractionResult

```python
from calcine import ExtractionResult
```

The universal return type from `Feature.extract`.

### ExtractionResult.of()

```python
ExtractionResult.of(entity_id, value)
```

Convenience constructor for single-record features. Equivalent to
`ExtractionResult(records={entity_id: value})`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `records` | `dict[str, Any]` | Entity ID → extracted value. One entry for single-record features; many for fan-out. |
| `metadata` | `dict[str, Any] \| None` | Parent-level metadata for fan-out features. `None` for single-record features. |

---

## FeatureStore

```python
from calcine.stores.base import FeatureStore
```

### Sync methods (implement these)

| Method | Signature | Description |
|--------|-----------|-------------|
| `read` | `(feature, entity_id) -> Any` | Return stored value. Raise `KeyError` if missing, `StoreError` on I/O failure. |
| `write` | `(feature, entity_id, result, context=None) -> None` | Persist an `ExtractionResult`. Default raises `NotImplementedError`. |
| `exists` | `(feature, entity_id) -> bool` | Return `True` if a value is stored. Default raises `NotImplementedError`. |
| `delete` | `(feature, entity_id) -> None` | Delete a stored value. Raise `KeyError` if missing. Default raises `NotImplementedError`. |
| `list_entities` | `(feature, prefix=None) -> list[str]` | List stored entity IDs, optionally filtered by prefix. Default raises `NotImplementedError`. |

### Async methods (override for native async backends)

The framework calls the `a`-prefixed variants internally. By default they
wrap the sync methods in a thread executor. Override them directly when your
backend has a native async client:

| Method | Default behaviour |
|--------|-------------------|
| `aread` | Runs `read` in executor |
| `awrite` | Runs `write` in executor |
| `aexists` | Runs `exists` in executor |
| `adelete` | Runs `delete` in executor |
| `alist_entities` | Runs `list_entities` in executor |

### _feature_key()

```python
def _feature_key(self, feature: Feature) -> str
```

Returns `type(feature).__name__` by default. Override in a store subclass to
customise namespacing, for example to include a module path or version tag when
class name collisions are possible across teams.

### write() and fan-out

`write` always receives an `ExtractionResult`. For single-record features,
`result.records` has one entry keyed by `entity_id`. For fan-out features,
`result.records` contains sub-entity entries and `result.metadata` holds
optional parent-level data.

To support `overwrite=False`, the store must write something under the parent
`entity_id` even for fan-out features. The convention is to write `result.metadata`
(or `{}` when `metadata` is `None`) when `entity_id` is not already a key in
`result.records`:

```python
def write(self, feature, entity_id, result, context=None):
    if entity_id not in result.records:
        parent = result.metadata if result.metadata is not None else {}
        self._backend.set(self._key(feature, entity_id), parent)
    for sub_id, record in result.records.items():
        self._backend.set(self._key(feature, sub_id), record)
```

---

## DataSource

```python
from calcine.sources.base import DataSource
```

### read()

```python
def read(self, entity_id: str, **kwargs) -> Any
```

Implement this for synchronous sources. The framework runs it in a thread
executor via `aread`, passing `entity_id` as a keyword argument.

The base class signature is `def read(self, **kwargs)` — `entity_id` arrives
as a keyword argument from the pipeline. Declare it explicitly in your
implementation for clarity:

```python
class MySource(DataSource):
    def read(self, entity_id: str, **kwargs) -> Any:
        return self.db.get(entity_id)
```

### aread()

```python
async def aread(self, entity_id: str, **kwargs) -> Any
```

Override this instead of `read` for natively async sources (async HTTP
clients, async database drivers). The default implementation runs `read`
in a thread executor.

---

## SourceBundle

```python
from calcine.sources import SourceBundle

source = SourceBundle(
    transactions=TransactionSource(),
    profile=ProfileSource(),
)
```

A `DataSource` that reads from multiple sub-sources concurrently via
`asyncio.gather` and delivers a single `dict` keyed by the names you choose.
`Feature.extract` receives that dict as `raw`:

```python
async def extract(self, raw: dict, context, entity_id=None):
    txns = raw["transactions"]
    prof = raw["profile"]
```

A failure in any sub-source fails the whole bundle for that entity.

---

## GenerationReport

See [`generation-report.md`](generation-report.md) for the full reference.

Quick summary:

```python
report = pipeline.generate(entity_ids=ids)

# Counts
report.success_count     # int
report.failure_count     # int (len(report.failed))
report.skip_count        # int (len(report.skipped))
report.record_count      # int — total records written (> success_count for fan-out)
report.throughput        # float — entities/second

# Outcomes
report.succeeded         # dict[str, ExtractionResult] — only when store_results=True
report.failed            # dict[str, list[str]] — entity_id → error strings
report.exceptions        # dict[str, BaseException] — raw exceptions for unhandled failures
report.skipped           # set[str]

# Analysis
report.error_summary()   # dict[str, list[str]] — grouped by error message, sorted by frequency
report.timing_summary()  # per-phase p50/p95/max/mean/total seconds
report.to_dataframe()    # pandas DataFrame, one row per entity

# Retry failures
retry = pipeline.generate(entity_ids=list(report.failed))
```
