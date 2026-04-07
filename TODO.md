# calcine — TODO

Priority tiers: **P0** = blocking real use, **P1** = significant gap, **P2** = quality of life, **P3** = stretch.

---

## P0 — Core functional gaps

- [ ] **Multi-feature pipeline** — Support running multiple `Feature` instances against a single `DataSource` in one `Pipeline`. The source is read once; each feature receives the same raw data. Avoids re-reading the source N times for N features from the same origin. Most common real-world pattern and a significant gap in the current API.

---

## P1 — Significant feature gaps

- [ ] **Feature versioning** — Allow a `version: str` class attribute on `Feature` that is included in the store key, so `MeanPurchaseValue_v2` coexists with `MeanPurchaseValue_v1` without collision. Without it, re-extracting a feature silently overwrites prior results.

- [ ] **Schema statistical constraints** — Extend the schema type system with value-level validation: range bounds (`min`/`max` for numeric types), `allow_inf: bool` on Float types, and finite-only enforcement. Makes the schema a genuine data quality gate rather than just a shape and type checker.

- [ ] **SQLite store** — A `SQLiteStore(path)` that persists features to a single SQLite file keyed by `(feature_name, entity_id)`. Persistent, zero-config, no directory explosion, portable. The right default persistent store; `FileStore` should be de-emphasised in docs once this exists.

- [ ] **Store bulk read** — Add `read_many(feature, entity_ids) -> list[Any]` for validated bulk retrieval, with an `aread_many` async variant. Without this, the typed contract story only holds for single-entity lookups. `read_many` would also supersede `retrieve_batch` on `Pipeline`.

- [ ] **Fault-tolerant SourceBundle** — Add `SourceBundle(..., fault_tolerant: bool = False)`. When enabled, a failing sub-source returns `None` for its key rather than propagating the exception. Lets features degrade gracefully when optional sources are unavailable.

---

## P2 — Quality of life

- [ ] **`FileStore` entity listing** — `FileStore` does not implement `list_entities`. `MemoryStore` does. `FileStore` should scan its directory structure to support prefix-based sub-entity discovery for fan-out features.

- [ ] **Cross-field schema validation** — Add an optional `validate_record(self, record: dict) -> list[str]` hook to `FeatureSchema` for constraints that span multiple fields (e.g. `end_time > start_time`).

- [ ] **Demote built-in sources** — `FileSource`, `DirectorySource`, and `DataFrameSource` are thin wrappers that create a false impression of completeness. Move them to a `calcine.contrib` subpackage or document them clearly as reference implementations, not production components.

- [ ] **`Pipeline` async context manager** — Support `async with Pipeline(...) as p:` so stores that need setup/teardown (e.g. connection pools) can manage their lifecycle cleanly.

- [ ] **`list_features()` on store** — Add `FeatureStore.list_features() -> list[str]` to return all feature namespaces present in the store. Useful for inspection and tooling.

---

## P3 — Stretch / future

- [ ] **S3 source and store** — `S3Source` and `S3Store` via `boto3`/`aiobotocore`.

- [ ] **Redis store** — `RedisStore` for low-latency feature serving.

- [ ] **Feature registry** — A lightweight `FeatureRegistry` that maps feature names to classes, enabling pipeline construction from config files or CLI invocations.

- [ ] **CLI** — `calcine generate`, `calcine inspect`, `calcine delete` commands for operating on stores from the terminal without writing Python.

- [ ] **Stream-mode generate** — Accept an `AsyncIterator[str]` as entity source so pipelines can process infinite or externally-driven entity streams (Kafka, webhooks, etc.).
