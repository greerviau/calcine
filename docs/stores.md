# Built-in stores

calcine ships with several reference `FeatureStore` implementations covering
common persistence patterns. These are scaffolding for prototyping and simple
use cases — the `FeatureStore` ABC is the product. Build your own when the
built-ins don't fit.

For custom store implementations, see [`docs/extending.md`](extending.md).

---

## MemoryStore

Dict-backed, no I/O. Intended for tests and interactive prototyping.

```python
from calcine.stores import MemoryStore

store = MemoryStore()
```

Not persistent — data is lost when the process exits. All operations are
synchronous dict lookups under the hood.

---

## FileStore

Stores one file per entity per feature. Serialization is pluggable; the
default is `PickleSerializer`.

```python
from calcine.stores import FileStore
from calcine.serializers import JSONSerializer

store = FileStore("/data/features/", serializer=JSONSerializer())
```

Directory structure: `<path>/<FeatureName>/<entity_id>.<ext>`

---

## ParquetStore

Stores features in Parquet files, one file per feature class. Requires
`pip install "calcine[parquet]"`.

```python
from calcine.stores import ParquetStore

store = ParquetStore("/data/features/")
```

Well-suited for tabular dict features. Each write currently reloads and
rewrites the full Parquet file — see the TODO for the planned append
optimization.

---

## Serializers (for FileStore)

Serializers control how feature values are converted to and from bytes in
`FileStore`. Pass one to `FileStore(path, serializer=...)`.

| Class | File ext | Best for |
|-------|----------|----------|
| `PickleSerializer` | `.pkl` | Any Python object (default) |
| `JSONSerializer` | `.json` | Dict / list / primitive features |
| `NumpySerializer` | `.npy` | `numpy` arrays |

```python
from calcine.serializers import NumpySerializer
from calcine.stores import FileStore

store = FileStore("/data/embeddings/", serializer=NumpySerializer())
```

To add a custom serializer, subclass `Serializer` and implement `serialize`
and `deserialize`. See [`docs/extending.md`](extending.md).
