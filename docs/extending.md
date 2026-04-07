# Extending calcine

Every component in calcine is designed to be subclassed.  This guide shows
the minimum required implementation for each extension point.

## Custom DataSource

Implement `read` for synchronous sources.  The framework runs it in a thread
executor so it never blocks the event loop:

```python
from typing import Any
from calcine.sources.base import DataSource
from calcine.exceptions import SourceError


class PostgresSource(DataSource):
    """Read rows from a Postgres table filtered by entity_id."""

    def __init__(self, conn, table: str, entity_col: str = "entity_id"):
        self.conn = conn
        self.table = table
        self.entity_col = entity_col

    def read(self, entity_id: str | None = None, **kwargs: Any):
        if entity_id is None:
            raise ValueError("entity_id is required")
        try:
            return self.conn.execute(
                f"SELECT * FROM {self.table} WHERE {self.entity_col} = %s",
                (entity_id,),
            ).fetchall()
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=entity_id,
                cause=exc,
            ) from exc
```

For sources with a **native async client** (async database drivers, async HTTP),
override `aread` instead:

```python
class BigQuerySource(DataSource):
    def __init__(self, client, table: str):
        self.client = client
        self.table = table

    async def aread(self, entity_id: str | None = None, **kwargs: Any):
        if entity_id is None:
            raise ValueError("entity_id is required")
        try:
            rows = await self.client.query(
                f"SELECT * FROM `{self.table}` WHERE entity_id = @id",
                parameters={"id": entity_id},
            )
            return rows
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=entity_id,
                cause=exc,
            ) from exc
```

**Rules:**

- Accept `entity_id` as a keyword argument; scope results to it
- Wrap all I/O failures in `SourceError` with the three required fields
- Return whatever type your `Feature` expects — there's no constraint on format
- Implement `read` for synchronous sources (the default); override `aread` for
  natively async sources

---

## Custom Feature

```python
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types


class SentimentFeature(Feature):
    schema = FeatureSchema({
        "score":     types.Float64(nullable=False),
        "label":     types.Category(categories=["positive", "neutral", "negative"]),
        "confident": types.Boolean(nullable=False),
    })

    def __init__(self, model):
        self.model = model

    async def extract(self, raw: str, context: dict) -> dict:
        score = await self.model.predict(raw)
        label = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
        return {
            "score":     float(score),
            "label":     label,
            "confident": abs(score - 0.5) > 0.3,
        }
```

**Lifecycle:**

```
source.aread() → feature.extract(raw, context) → validate(result) → store.awrite()
```

---

## Custom FeatureStore

Implement the synchronous methods.  The framework runs them in a thread
executor automatically:

```python
import pickle
from calcine.stores.base import FeatureStore
from calcine.exceptions import StoreError


class RedisStore(FeatureStore):
    def __init__(self, redis):
        self.redis = redis

    def _key(self, feature, entity_id: str) -> str:
        return f"calcine:{self._feature_key(feature)}:{entity_id}"

    def write(self, feature, entity_id, result, context=None):
        # Write metadata/tombstone under parent key for fan-out support
        if entity_id not in result.records:
            parent = result.metadata if result.metadata is not None else {}
            self.redis.set(self._key(feature, entity_id), pickle.dumps(parent))
        for sub_id, record in result.records.items():
            try:
                self.redis.set(self._key(feature, sub_id), pickle.dumps(record))
            except Exception as exc:
                raise StoreError(
                    store_name=type(self).__name__,
                    feature_name=self._feature_key(feature),
                    entity_id=sub_id,
                    cause=exc,
                ) from exc

    def read(self, feature, entity_id):
        try:
            raw = self.redis.get(self._key(feature, entity_id))
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc
        if raw is None:
            raise KeyError(f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'")
        return pickle.loads(raw)

    def exists(self, feature, entity_id) -> bool:
        return bool(self.redis.exists(self._key(feature, entity_id)))

    def delete(self, feature, entity_id):
        deleted = self.redis.delete(self._key(feature, entity_id))
        if not deleted:
            raise KeyError(f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'")
```

For stores with a **native async client**, override the async methods instead:

```python
class AsyncRedisStore(FeatureStore):
    def __init__(self, redis):
        self.redis = redis

    def _key(self, feature, entity_id):
        return f"calcine:{self._feature_key(feature)}:{entity_id}"

    async def awrite(self, feature, entity_id, result, context=None):
        if entity_id not in result.records:
            parent = result.metadata if result.metadata is not None else {}
            await self.redis.set(self._key(feature, entity_id), pickle.dumps(parent))
        for sub_id, record in result.records.items():
            await self.redis.set(self._key(feature, sub_id), pickle.dumps(record))

    async def aread(self, feature, entity_id):
        raw = await self.redis.get(self._key(feature, entity_id))
        if raw is None:
            raise KeyError(entity_id)
        return pickle.loads(raw)

    async def aexists(self, feature, entity_id) -> bool:
        return bool(await self.redis.exists(self._key(feature, entity_id)))

    async def adelete(self, feature, entity_id):
        deleted = await self.redis.delete(self._key(feature, entity_id))
        if not deleted:
            raise KeyError(entity_id)
```

**Rules:**

- `read` and `delete` must raise `KeyError` (not `StoreError`) when the entity
  simply doesn't exist — this is how `retrieve_batch` knows to silently skip it
- Wrap all other I/O failures in `StoreError` with the four required fields
- Use `_feature_key(feature)` for namespacing, not `id(feature)`
- Implement sync methods (`read`, `write`, etc.) for the default path; override
  async methods (`aread`, `awrite`, etc.) for natively async clients

---

## Custom Serializer (for FileStore)

```python
import msgpack
from calcine.serializers import Serializer


class MsgPackSerializer(Serializer):
    def serialize(self, data) -> bytes:
        return msgpack.packb(data, use_bin_type=True)

    def deserialize(self, raw: bytes):
        return msgpack.unpackb(raw, raw=False)
```

```python
store = FileStore("/data/features", serializer=MsgPackSerializer())
```

---

## Custom schema type

```python
from calcine.schema import FeatureType, types
from typing import Any


class PositiveFloat(FeatureType):
    """A float that must be strictly greater than zero."""

    def _validate_value(self, value: Any) -> list[str]:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return [f"Expected numeric value, got {type(value).__name__}"]
        if f <= 0:
            return [f"Expected positive float, got {f}"]
        return []


# Use it directly
schema = FeatureSchema({"price": PositiveFloat(nullable=False)})
```

---

## Combining extensions

Extensions compose naturally:

```python
Pipeline(
    source=SourceBundle(
        events=BigQuerySource(bq_client, "project.dataset.events"),
        profile=PostgresSource(pg_pool, "users"),
    ),
    feature=SentimentFeature(model=my_model),
    store=RedisStore(redis_client),
)
```
