# Built-in sources

calcine ships with several reference `DataSource` implementations covering common
patterns. These are scaffolding for prototyping and simple use cases — the
`DataSource` ABC is the product. Build your own when the built-ins don't fit.

For custom source implementations, see [`docs/extending.md`](extending.md).

---

## DataFrameSource

Filters a pandas DataFrame by entity ID. Useful for offline experiments and tests.

```python
from calcine.sources import DataFrameSource
import pandas as pd

df = pd.DataFrame({"entity_id": ["u1", "u2"], "amount": [100.0, 200.0]})
source = DataFrameSource(df, entity_col="entity_id")
```

`entity_col` defaults to `"entity_id"`. `read()` returns the filtered
sub-DataFrame for the given entity. Returns an empty DataFrame (not an error)
if no rows match.

---

## FileSource

Reads the entire contents of a single file as `bytes`.

```python
from calcine.sources import FileSource

source = FileSource("/data/features/input.bin")
```

Returns the same bytes regardless of `entity_id`. Useful when the file itself
encodes the full dataset and the feature slices it internally.

---

## DirectorySource

Reads all files matching a glob pattern from a directory and returns their
contents as a `list[bytes]`, sorted by filename.

```python
from calcine.sources import DirectorySource

source = DirectorySource("/data/audio/", pattern="*.wav")
```

Also supports streaming via `async for chunk in source.stream()` — each
file is yielded individually, which avoids loading all files into memory at
once for large directories.

---

## SourceBundle

Reads from multiple sources concurrently and delivers a single `dict` to
`Feature.extract`. All sources run simultaneously via `asyncio.gather`.

```python
from calcine.sources import SourceBundle

source = SourceBundle(
    transactions=TransactionSource(),
    profile=ProfileSource(),
)
```

`Feature.extract` receives `raw = {"transactions": ..., "profile": ...}`.

A failure in any sub-source fails the bundle for that entity. See
[`docs/architecture.md`](architecture.md) for the fault-tolerance discussion.
