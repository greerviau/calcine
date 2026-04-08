# calcine

Async feature engineering pipeline library. Core abstraction:

```
DataSource → Feature → FeatureStore
```

orchestrated by `Pipeline`.

## Commands

```bash
uv run pytest tests/              # run tests
uv run ruff check .               # lint
uv run ruff check --fix .         # lint + autofix
```

## Architecture

**Pipeline** is a generation tool only — `generate()` is its purpose. It orchestrates concurrency, per-entity error isolation, incremental generation (skip already-stored), and reporting. It is not a data access layer.

**DataSource** fetches raw data per entity. **Feature** transforms raw data into a typed, validated value. **FeatureStore** persists and retrieves feature values.

## Key conventions

**Users override sync methods; the framework handles async.**

- `DataSource`: override `def read(self, entity_id, **kwargs)`. The base class `aread` runs it in a thread executor. For native async sources (async DB drivers, async HTTP), override `aread` instead.
- `FeatureStore`: override `def read`, `def write`, `def exists`, `def delete`. The base class `aread/awrite/aexists/adelete` wrap these in a thread executor. For native async backends, override the async methods instead.
- `Feature`: override `def extract(self, raw, context, entity_id=None) -> ExtractionResult`. The base class `aextract` runs it in a thread executor. For native async extraction (async model clients, async HTTP), override `aextract` instead.

**`ExtractionResult`** is the universal return type from `Feature.extract`:
- Single record: `ExtractionResult.of(entity_id, value)`
- Fan-out: `ExtractionResult(records={sub_id: value, ...}, metadata={...})`

**Schema validation** runs on write (before the store) and on read. Schema violations go to `report.failed`, never raise.

## What not to do

- Don't add `retrieve()` to `Pipeline` — it doesn't belong there; use `store.read()` directly.
- Don't add `pre_extract`/`post_extract` hooks — put that logic in `extract`.
- Don't make built-in sources/stores more production-grade — they are reference implementations.
- Don't break the sync/async layering: the pipeline always calls `aread`/`awrite` internally; sync convenience is at the store level.

## Documentation style

- Avoid overusing em dashes. Use them sparingly — only when a comma or period would be genuinely weaker. Prefer shorter, direct sentences over long ones connected by dashes.
- No emojis.
- Code examples should be minimal and runnable.

## Claude agents and skills

**Skills** (slash commands):

- `/test` — run the test suite and diagnose any failures
- `/lint` — run ruff and fix all lint errors

**Subagents** (invoke via the Agent tool or by asking Claude to use them):

- `extension-reviewer` — reviews a custom `DataSource`, `Feature`, or `FeatureStore` implementation for correctness against calcine's conventions
- `architect` — evaluates proposed features or API changes against calcine's architectural principles; returns a recommendation, rationale, trade-offs, and API sketch

## Testing

Tests are in `tests/`, mirroring the `calcine/` structure. All store tests use `await store.aread/awrite/aexists/adelete`. All source tests use `await source.aread`. Custom sources in tests use `def read` (sync). Custom stores in tests use `def read` (sync, abstract must be satisfied). Custom features in tests use `def extract` (sync); override `aextract` only when the test requires native async behaviour.
