# calcine — TODO

Priority tiers: **P0** = blocking real use, **P1** = significant gap, **P2** = quality of life, **P3** = stretch.

---

## Core value proposition

calcine's differentiation is in **pipeline orchestration** (concurrency, per-entity error isolation, incremental generation, reporting) and the **schema validation system**. Together these form a **typed contract between feature producers and consumers**: feature engineers generate into the store with schema validation on write; ML engineers and inference engineers read from the store with schema validation on read. Everyone goes through the same typed interface, catching data problems at the access point rather than inside a model or serving system.

The built-in sources and file-based stores are reference implementations, not the product. The `DataSource` and `FeatureStore` ABCs — and the schema that enforces correctness across both directions — are the product.

---

## P0 — Core functional gaps

- [x] **Partitioned concurrency in `generate()`** — `generate()` currently iterates entities serially. Need a flexible model that supports three modes without changing the rest of the API:

  1. **Flat concurrency** — `concurrency: int` acts as a semaphore cap; up to N entities processed concurrently from a shared pool.
  2. **Partition function** — `partition_by: Callable[[str], Hashable]` groups entity IDs by the return value. Each group is a serial queue; up to `concurrency` groups run concurrently. Handles rate-limit-per-account, shared-shard writes, ordered-per-user processing, etc.
  3. **Explicit partitions** — `partitions: dict[str, list[str]]` lets the caller supply pre-built groups directly (useful when partition structure comes from external metadata).

  Target API:
  ```python
  # flat (sync default)
  report = pipeline.generate(entity_ids=ids, concurrency=16)
  # partitioned by function
  report = pipeline.generate(entity_ids=ids, partition_by=lambda eid: eid.split("_")[0], concurrency=8)
  # partitioned explicitly
  report = pipeline.generate(partitions={"shard_0": [...], "shard_1": [...]}, concurrency=4)
  # async opt-in
  report = await pipeline.agenerate(entity_ids=ids, concurrency=16)
  ```
  Default (`concurrency=1`, no partition args) stays fully serial so existing behaviour is unchanged.

- [x] **Incremental generation (`only_missing` mode)** — Add an `overwrite: bool = True` parameter to `generate()`. When `False`, skip entities where `store.exists()` returns True. Makes re-running pipelines cheap.

- [x] **Progress reporting** — Add a `on_progress` callback parameter to `generate()` that receives `(completed: int, total: int, report_so_far: GenerationReport)` after each entity. Lets callers wire in tqdm, logging, or custom telemetry without coupling calcine to any specific library.

- [x] **Sync-first public API** — Sync is the default public interface; async variants are opt-in using the `a`-prefix convention (following Django ORM). Pipeline: `generate`/`retrieve`/`retrieve_batch` are sync; `agenerate`/`aretrieve`/`aretrieve_batch` are async. Store: `read`/`write`/`exists`/`delete` are sync wrappers; `aread`/`awrite`/`aexists`/`adelete` are async. (`read_many` / `to_dataframe` are tracked separately under P1.)

- [ ] **Multi-feature pipeline** — Support running multiple `Feature` instances against a single `DataSource` in one `Pipeline`. The source is read once; each feature gets the same raw data. Avoids re-reading the source N times for N features from the same origin. This is the most common real-world pattern and a significant gap in the current API.

---

## P1 — Significant feature gaps

- [x] **Batch extract** — Add an optional `async extract_batch(raws: list[Any], context: dict) -> list[Any]` method to `Feature`. When defined, `generate()` should collect a batch of raw reads and call it once instead of N times. Critical for ML model inference and bulk DB queries.

- [ ] **Feature versioning** — Allow a `version: str` class attribute on `Feature` that is included in the store key, so `MeanPurchaseValue_v2` coexists with `MeanPurchaseValue_v1` without collision. This is what separates a pipeline runner from an actual feature store; without it, re-extracting a feature silently overwrites prior results.

- [ ] **Schema statistical constraints** — Extend the schema type system with value-level validation: range bounds (`min`/`max` for numeric types), `allow_inf: bool` on Float types, and finite-only enforcement. The NaN fix was a correctness patch; this makes the schema genuinely useful as a data quality gate rather than just a shape/type checker.

- [x] **Fix NaN in Float32/Float64 validation** — `float("nan")` currently passes. Add `allow_nan: bool = False` parameter (default False for new code, keep True as opt-in). This is a correctness bug, not just a limitation.

- [ ] **SQLite store** — A `SQLiteStore(path)` that persists features to a single SQLite file keyed by `(feature_name, entity_id)`. Persistent, zero-config, no directory explosion, portable. The right default persistent store; `FileStore` should be de-emphasized in docs in favour of this once it exists.

- [x] **`GenerationReport` per-phase timing** — Track wall time for source read, extract, and store write separately per entity. Surface aggregates (p50/p95/max per phase) in `GenerationReport`. This turns calcine from a pipeline runner into a bottleneck analysis tool — a real reason to adopt it over a hand-rolled loop.

- [ ] **Store bulk read + DataFrame export** — Add `read_many(feature, entity_ids) -> list[Any]` for validated bulk retrieval, and `to_dataframe(feature, entity_ids) -> pd.DataFrame` for ML-ready export with correct dtypes derived from the schema. Both sync by default with `aread_many` / `ato_dataframe` async variants. Without this, the typed contract story only holds for single-entity lookups; training workflows have no clean path through the store. `read_many` also supersedes the existing `retrieve_batch` on `Pipeline`.

- [x] **Fan-out extraction (`ExtractionResult`)** — `ExtractionResult(records, metadata)` is the universal return type from `Feature.extract`; single-record features use `ExtractionResult.of(entity_id, value)`; `parent_schema` validates metadata, `schema` validates each record; `MemoryStore` implements `alist_entities(feature, prefix)`; `overwrite=False` checks parent entity_id.

- [ ] **Fault-tolerant SourceBundle** — Add `SourceBundle(..., fault_tolerant: bool = False)`. When enabled, a failing sub-source returns `None` for its key rather than propagating the exception. Lets features degrade gracefully when optional sources are unavailable.

- [ ] **ParquetStore append optimization** — Current implementation loads the entire Parquet file, modifies a row, and rewrites. Replace with an append-and-deduplicate strategy (or partition by entity) to make writes O(1) rather than O(n).

---

## P2 — Quality of life

- [ ] **Cross-field schema validation** — Add an optional `validate_record(self, record: dict) -> list[str]` hook to `FeatureSchema` (or `FeatureType`) for constraints that span multiple fields (e.g., `end_time > start_time`).

- [x] **Simplify Feature API — removed `pre_extract` / `post_extract`** — Lifecycle hooks removed; `extract` is the single transformation point.

- [ ] **Demote built-in sources to examples/extras** — `FileSource`, `DirectorySource`, `HTTPSource`, and `DataFrameSource` are thin wrappers that create a false impression of completeness. Move them to a `calcine.contrib` subpackage or clearly document them as reference implementations, not production components. The `DataSource` ABC is the product; the built-ins are scaffolding.

- [ ] **`Pipeline` async context manager** — Support `async with Pipeline(...) as p:` so stores that need setup/teardown (e.g., connection pools) can manage their lifecycle cleanly.

- [ ] **Store inspection and prefix-based entity listing** — Add `FeatureStore.list_features() -> list[str]` and `FeatureStore.list_entities(feature, prefix: str | None = None) -> list[str]`. Prefix filtering is a first-class requirement, not an afterthought: it is the primary mechanism for discovering sub-entities produced by fan-out features (e.g. `store.list_entities(AudioSegmentFeature, prefix="recording_001/")`).

- [ ] **HTTPSource retry/backoff** — Add `retries: int = 0` and `backoff: float = 0.5` parameters to `HTTPSource`. Retry on transient HTTP errors (5xx, timeouts) with exponential backoff.

---

## P3 — Stretch / future

- [ ] **S3 / object storage source and store** — `S3Source` and `S3Store` via `boto3`/`aiobotocore` as an optional extra `calcine[s3]`.

- [ ] **Redis store** — `RedisStore` for low-latency feature serving as an optional extra `calcine[redis]`.

- [ ] **Feature registry** — A lightweight `FeatureRegistry` that maps feature names to classes, enabling pipeline construction from config files or CLI invocations.

- [ ] **CLI** — `calcine generate`, `calcine inspect`, `calcine delete` commands for operating on stores from the terminal without writing Python.

- [ ] **Stream-mode generate** — Instead of collecting all `entity_ids` upfront, accept an `AsyncIterator[str]` so pipelines can process infinite or externally-driven entity streams (Kafka, webhooks, etc.).
