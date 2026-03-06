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
sub-DataFrame for the given entity. Raises `SourceError` if no rows match.

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

Reads one file per entity from a directory, matching filenames with a pattern.

```python
from calcine.sources import DirectorySource

source = DirectorySource("/data/audio/", pattern="{entity_id}.wav")
```

Returns `bytes` for the matched file. Raises `SourceError` if no file matches
the pattern for a given entity.

---

## HTTPSource

Makes an async HTTP GET request per entity. Requires `pip install "calcine[http]"`.

```python
from calcine.sources import HTTPSource

source = HTTPSource(url_template="https://api.example.com/users/{entity_id}")
```

The URL template is formatted with `entity_id`. Returns the response body as
`bytes`. Raises `SourceError` on non-2xx status codes.

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
