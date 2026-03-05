"""Parquet-based feature store for tabular / dict features."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import StoreError
from ..extraction import ExtractionResult
from .base import FeatureStore

if TYPE_CHECKING:
    from ..features.base import Feature


class ParquetStore(FeatureStore):
    """Persist dict-valued features as Parquet files partitioned by feature name.

    Each feature class gets its own ``.parquet`` file containing all entity
    rows.  A special ``entity_id`` column records the entity key.  Non-dict
    scalar values are stored under a ``"value"`` column.

    Directory layout::

        {base_path}/
            {FeatureClassName}.parquet

    Args:
        path: Base directory where Parquet files are written.

    Requires the ``[parquet]`` extra::

        pip install calcine[parquet]

    Example::

        store = ParquetStore("/data/feature_store")
        store.write(feature, "u1", {"mean_value": 15.0, "count": 2})
        record = store.read(feature, "u1")
        # {"mean_value": 15.0, "count": 2}
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def _feature_path(self, feature: Feature) -> Path:
        return self.path / f"{self._feature_key(feature)}.parquet"

    @staticmethod
    def _check_deps() -> None:
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pandas and pyarrow are required for ParquetStore. "
                "Install with: pip install calcine[parquet]"
            ) from exc

    async def awrite(
        self,
        feature: Feature,
        entity_id: str,
        result: ExtractionResult,
        context: dict | None = None,
    ) -> None:
        self._check_deps()
        import pandas as pd

        path = self._feature_path(feature)
        loop = asyncio.get_running_loop()

        def _write() -> None:
            # Collect all (sub_id, record) pairs from the ExtractionResult.
            # For single-record features, result.records == {entity_id: value}.
            # For fan-out features, result.records contains sub-entity entries;
            # in that case also write a metadata/tombstone row under entity_id.
            rows: list[dict[str, Any]] = []
            if entity_id not in result.records:
                parent = result.metadata if result.metadata is not None else {}
                row: dict[str, Any] = {"entity_id": entity_id}
                if isinstance(parent, dict):
                    row.update(parent)
                else:
                    row["value"] = parent
                rows.append(row)
            for sub_id, record in result.records.items():
                row = {"entity_id": sub_id}
                if isinstance(record, dict):
                    row.update(record)
                else:
                    row["value"] = record
                rows.append(row)

            new_df = pd.DataFrame(rows)
            written_ids = {r["entity_id"] for r in rows}

            if path.exists():
                existing = pd.read_parquet(path)
                existing = existing[~existing["entity_id"].isin(written_ids)]
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(path, index=False)

        try:
            await loop.run_in_executor(None, _write)
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc

    async def aread(self, feature: Feature, entity_id: str) -> Any:
        self._check_deps()

        path = self._feature_path(feature)
        if not path.exists():
            raise KeyError(
                f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
            )

        loop = asyncio.get_running_loop()

        def _read() -> dict[str, Any]:
            import pandas as pd

            df = pd.read_parquet(path)
            rows = df[df["entity_id"] == entity_id]
            if rows.empty:
                raise KeyError(
                    f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
                )
            return rows.iloc[0].drop("entity_id").to_dict()

        try:
            return await loop.run_in_executor(None, _read)
        except KeyError:
            raise
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc

    async def aexists(self, feature: Feature, entity_id: str) -> bool:
        self._check_deps()

        path = self._feature_path(feature)
        if not path.exists():
            return False

        loop = asyncio.get_running_loop()

        def _check() -> bool:
            import pandas as pd

            df = pd.read_parquet(path)
            return entity_id in df["entity_id"].values

        return await loop.run_in_executor(None, _check)

    async def adelete(self, feature: Feature, entity_id: str) -> None:
        self._check_deps()

        path = self._feature_path(feature)
        if not path.exists():
            raise KeyError(
                f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
            )

        loop = asyncio.get_running_loop()

        def _delete() -> None:
            import pandas as pd

            df = pd.read_parquet(path)
            if entity_id not in df["entity_id"].values:
                raise KeyError(
                    f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
                )
            df = df[df["entity_id"] != entity_id]
            df.to_parquet(path, index=False)

        try:
            await loop.run_in_executor(None, _delete)
        except KeyError:
            raise
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc
