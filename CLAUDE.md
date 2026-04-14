# CLAUDE.md

## Project Overview

Async feature engineering pipeline library. Core abstraction: `DataSource → Feature → FeatureStore`, orchestrated by `Pipeline`.

## Commands

```bash
uv run pytest tests/    # run tests
uv run ruff check .     # lint
uv run ruff check --fix .  # lint + autofix
```

## Architecture

**Pipeline** = generation tool only. `generate()` is its purpose: concurrency, per-entity error isolation, incremental generation, reporting. Not a data access layer.

**DataSource** fetches raw data. **Feature** transforms raw → typed, validated value. **FeatureStore** persists and retrieves.

## Conventions

**Override sync; framework handles async.**

- `DataSource`: override `def read(self, entity_id, **kwargs)`. Base `aread` runs it in thread executor. For native async sources, override `aread`.
- `FeatureStore`: override `def read/write/exists/delete`. Base `aread/awrite/aexists/adelete` wrap in thread executor. For native async backends, override async methods.
- `Feature`: override `def extract(self, raw, context, entity_id=None) -> ExtractionResult`. Base `aextract` runs in thread executor. For native async, override `aextract`.

**`ExtractionResult`** — universal return from `Feature.extract`:

- Single: `ExtractionResult.of(entity_id, value)`
- Fan-out: `ExtractionResult(records={sub_id: value, ...}, metadata={...})`

**Schema validation** on write (before store) and on read. Violations → `report.failed`, never raise.

## What not to do

- No `retrieve()` on `Pipeline` — use `store.read()` directly.
- No `pre_extract`/`post_extract` hooks — put logic in `extract`.
- Don't make built-in sources/stores production-grade — reference implementations only.
- Don't break sync/async layering: pipeline always calls `aread`/`awrite` internally.

## Docs style

- Em dashes sparingly. Prefer shorter, direct sentences.
- No emojis.
- Code examples: minimal and runnable.

## Skills & agents

**Skills** (slash commands):
- `/test` — run the test suite and diagnose failures
- `/lint` — run ruff and fix all lint errors

**Agents** (invoke via Agent tool):
- `extension-reviewer` — reviews custom DataSource/Feature/FeatureStore implementations
- `architect` — evaluates proposed features/API changes; returns recommendation, rationale, trade-offs, API sketch

## Testing

Tests in `tests/`, mirroring `calcine/` structure.

- Store tests: `await store.aread/awrite/aexists/adelete`
- Source tests: `await source.aread`
- Custom sources in tests: `def read` (sync)
- Custom stores in tests: `def read` (sync, satisfies abstract)
- Custom features in tests: `def extract` (sync); override `aextract` only when test requires native async
