# Team workflow and value proposition

This document explains how calcine fits into an ML engineering team's workflow,
what problems it solves, and what it is explicitly not trying to do.

---

## The problem

ML teams that build features for training and serving typically run into the same
set of problems as their systems grow:

**Reliability.** Batch feature jobs fail silently on a subset of entities.
A data issue in one user's record crashes the entire pipeline, or worse, writes
bad data without error. The team has no systematic view of which entities
succeeded, which failed, and why.

**Correctness.** A feature is recomputed and its output shape changes — an
embedding that was `(128,)` is now `(256,)`. Or a category value is added. The
training job doesn't notice until it throws a shape mismatch error. The serving
system doesn't notice until a model prediction is silently wrong.

**Consistency.** Feature logic lives in a notebook, a training script, and an
inference handler — three copies that have quietly drifted apart. Training and
serving are computing different things.

**Reproducibility.** The model was trained on features as they existed last
Tuesday. Nobody knows exactly what values were used, because the pipeline
overwrote them on Wednesday.

calcine addresses all four through a single mechanism: a **typed contract between
the code that produces features and the code that consumes them**.

---

## The typed contract

calcine's core abstraction is a `Feature` class that carries a `FeatureSchema`.
The schema is the contract: it declares exactly what fields a feature produces,
what types they are, what values are valid. This schema is enforced in both
directions:

- **On write** — when `Pipeline.generate()` runs, every extracted value is
  validated against the schema before it reaches the store. Entities that fail
  validation are recorded in the `GenerationReport` rather than written with bad
  data. Bad data never enters the store.

- **On read** — when the store is queried for a feature value, the returned data
  is validated against the same schema. If the store contains data that doesn't
  conform — because it was written by an old version of the feature, or was
  corrupted, or came from a different source — the read fails loudly rather than
  silently returning garbage.

The contract is expressed once, in the feature class, and enforced everywhere.

```python
class UserEngagementFeature(Feature):
    schema = FeatureSchema({
        "spend_tier": types.Category(categories=["low", "mid", "high", "whale"]),
        "event_rate": types.Float64(nullable=False, allow_nan=False),
        "total_spend": types.Float64(nullable=False, allow_nan=False),
    })

    async def extract(self, raw: dict, context: dict, entity_id: str) -> dict:
        spend = raw["total_spend"]
        return {
            "spend_tier": "low" if spend < 100 else "mid" if spend < 1000 else "high" if spend < 3000 else "whale",
            "event_rate": raw["event_count"] / raw["days_active"],
            "total_spend": spend,
        }
```

Every consumer of `UserEngagementFeature` — whether a training job, a serving
API, or a debugging script — reads through this same class and gets the same
guarantees.

---

## Async convention

calcine uses sync as the default public interface. Most ML and data engineering
code — batch jobs, training scripts, Airflow tasks, DataLoader workers — is
synchronous. Async is used internally for concurrency but should not be imposed
on callers who don't need it.

The convention throughout calcine:

```python
# Sync — default, works everywhere
report = pipeline.generate(entity_ids=ids)
value  = store.read(MyFeature, entity_id)

# Async — opt-in, for callers already in an async context
report = await pipeline.agenerate(entity_ids=ids)
value  = await store.aread(MyFeature, entity_id)
```

Async is the right choice in `SourceBundle` (concurrent reads across multiple
sources) and in async serving contexts (FastAPI handlers, async task workers).
Everywhere else, use the sync interface.

---

## How a team uses calcine

Three roles interact with calcine in different ways. In smaller teams these roles
overlap; the distinction is about the kind of work, not the number of people.

### Feature engineer

The feature engineer owns the feature definition and the generation pipeline.
Their primary tool is `Pipeline.generate()`.

```python
# Runs nightly as a batch job — plain Python script, no async boilerplate
pipeline = Pipeline(
    source=UserDBSource(),              # custom: reads from your database
    feature=UserEngagementFeature(),
    store=SQLiteStore("features/engagement.db"),
)

report = pipeline.generate(
    entity_ids=fetch_active_user_ids(),
    overwrite=False,        # skip users already computed since last run
    concurrency=32,
    on_progress=lambda done, total, _: print(f"{done}/{total}"),
)

# Full visibility into what happened
print(f"succeeded: {report.succeeded}")
print(f"failed:    {report.failed}")    # inspect report.failures for root cause
print(f"skipped:   {report.skipped}")   # already in store from a previous run
```

Key concerns for this role:
- **Reliability** — the pipeline never crashes on a bad entity; every failure is
  isolated and recorded
- **Incremental runs** — `overwrite=False` makes re-runs cheap; only new entities
  are processed
- **Schema correctness** — the schema in the feature class is the source of truth;
  anything that doesn't conform to it never reaches the store

The feature engineer does not need to know who will consume the features or how.
The schema is the handoff.

---

### ML engineer

The ML engineer consumes features for training experiments. They did not write
the feature pipeline and may not know its internals. They access features through
the store.

The store's read interface is synchronous by default, which means it works
naturally in training scripts, notebooks, and DataLoaders without any async
machinery.

#### Small to medium datasets — bulk read

For datasets that fit in memory, the simplest path is `read_many` or
`to_dataframe`:

```python
store = SQLiteStore("features/engagement.db")

# Bulk read — returns a list of validated dicts
features = store.read_many(UserEngagementFeature, entity_ids=training_ids)

# Or export directly to a DataFrame with correct dtypes derived from schema
df = store.to_dataframe(UserEngagementFeature, entity_ids=training_ids)

# Convert to tensors and train
X = torch.from_numpy(df[["event_rate", "total_spend"]].values.astype("float32"))
y = torch.tensor(labels)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
```

#### Large datasets — PyTorch DataLoader

For datasets too large to load at once, wrap the store in a `Dataset`. Because
store reads are synchronous, `Dataset.__getitem__` works cleanly with
`num_workers > 0`:

```python
class FeatureDataset(Dataset):
    def __init__(self, store, feature, entity_ids, labels):
        self.store = store
        self.feature = feature
        self.entity_ids = entity_ids
        self.labels = labels

    def __len__(self):
        return len(self.entity_ids)

    def __getitem__(self, idx):
        # store.read() is sync — works naturally in DataLoader worker processes
        features = self.store.read(self.feature, self.entity_ids[idx])
        return features_to_tensor(features), self.labels[idx]

dataset = FeatureDataset(store, UserEngagementFeature(), training_ids, labels)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

model = MyModel()
for epoch in range(num_epochs):
    for X_batch, y_batch in loader:
        ...
```

#### Why go through the store rather than computing features inline?

Two reasons:

1. **Cost.** Feature computation may involve DB queries, API calls, or expensive
   model inference. Computing features inline means paying that cost on every
   training run, every epoch. Storing them means paying it once.

2. **Contract.** Reading through the store validates the schema on the way out.
   If a feature was written with a bug — wrong shape, NaN in a non-nullable
   field, invalid category — `read_many` tells you immediately rather than
   letting bad data corrupt a training run silently. The same schema that
   governed the write governs the read.

Key concerns for this role:
- **Type safety** — the schema tells them exactly what fields exist and what type
  each one is; no guessing at the shape of the data
- **Confidence** — data problems surface at read time, not inside the model
- **Reproducibility** — features are stored, not recomputed; the same entity ID
  returns the same value across training runs until the feature engineer
  explicitly regenerates with a new version

---

### Inference engineer

The inference engineer serves model predictions at request time. They access
features through the same store interface, typically for single-entity lookups.

In a serving context (FastAPI, etc.) the caller is already async, so the async
store interface is the natural fit:

```python
# FastAPI handler
store = RedisStore(...)   # fast store for low-latency serving

@app.get("/predict/{user_id}")
async def predict(user_id: str) -> float:
    features = await store.aread(UserEngagementFeature, user_id)
    # features conforms to the schema — same guarantee the training job had
    return model.predict(features_to_tensor(features))
```

Key concerns for this role:
- **Latency** — features are pre-computed; the store read is a fast key-value
  lookup, not a live computation
- **Training-serving consistency** — the same feature class and schema is used
  for both training and serving; the contract is identical
- **Correctness at serving time** — if a feature value in the store is stale or
  malformed, the read fails loudly rather than feeding bad data to the model

---

## The full lifecycle

```
                  FEATURE ENGINEER
                        │
              pipeline.generate()          ← sync, reliable batch computation
                        │                    per-entity error isolation
              ┌─────────▼──────────┐         incremental runs (overwrite=False)
              │    FeatureStore    │  ← schema enforced on write
              │  (SQLite / Redis)  │
              └────────┬───────────┘
                       │                  schema enforced on read
          ┌────────────┼─────────────────────┐
          │            │                     │
    ML ENGINEER   ML ENGINEER          INFERENCE ENGINEER
    (small data)  (large data)
    read_many()   Dataset + DataLoader   aread()
    to_dataframe()   store.read()        async serving API
    direct tensor    per-item, sync
    conversion
```

The store is the shared artifact. The schema is the shared contract.
The feature class owns both.

---

## What calcine is not

**Not a training framework.** calcine does not replace PyTorch, Lightning,
or any training loop. It feeds data into them through standard interfaces
(DataLoader, DataFrame).

**Not a serving framework.** calcine does not handle model loading, request
routing, or latency budgets. It provides pre-computed, validated feature values
to whatever serving system you already have.

**Not a data access layer.** calcine does not provide production-grade database
clients, API clients, or data connectors. The built-in sources (`FileSource`,
`HTTPSource`, `DataFrameSource`) are reference implementations. In practice,
feature engineers write a `DataSource` subclass that wraps their own data
infrastructure.

**Not a feature platform.** calcine does not manage feature discovery across
teams, handle ACLs, or provide a UI. It is a library, not a managed service.

---

## What calcine is

A library for **defining, computing, storing, and accessing features with a
consistent typed contract**. It is most useful when:

- Features are computed in batch and consumed at training or serving time
- Multiple consumers (training jobs, inference services, experiments) need the
  same feature with the same semantics
- Feature computation is expensive enough that you want to compute once and
  reuse, not recompute on every training run or every request
- You want failures in feature computation to be visible and isolated, not silent
  or pipeline-aborting
