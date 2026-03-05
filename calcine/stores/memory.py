"""In-memory feature store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..extraction import ExtractionResult
from .base import FeatureStore

if TYPE_CHECKING:
    from ..features.base import Feature


class MemoryStore(FeatureStore):
    """In-memory feature store backed by a nested dict.

    Fully functional with no I/O dependencies — ideal for tests, notebooks,
    and rapid prototyping.  Data does not persist across ``MemoryStore``
    instances or process restarts.

    Example::

        store = MemoryStore()
        pipeline = Pipeline(source, feature, store)
        report = pipeline.generate(["u1", "u2"])

        value = store.read(feature, "u1")
    """

    def __init__(self) -> None:
        # Structure: {feature_name: {entity_id: data}}
        self._data: dict[str, dict[str, Any]] = {}

    async def awrite(
        self,
        feature: Feature,
        entity_id: str,
        result: ExtractionResult,
        context: dict | None = None,
    ) -> None:
        key = self._feature_key(feature)
        if key not in self._data:
            self._data[key] = {}
        # Write metadata (or tombstone) under entity_id when it isn't already
        # a record key — ensures aexists(entity_id) is True after any write.
        if entity_id not in result.records:
            self._data[key][entity_id] = result.metadata if result.metadata is not None else {}
        for sub_id, record in result.records.items():
            self._data[key][sub_id] = record

    async def aread(self, feature: Feature, entity_id: str) -> Any:
        key = self._feature_key(feature)
        try:
            return self._data[key][entity_id]
        except KeyError:
            raise KeyError(f"No data for feature '{key}', entity '{entity_id}'") from None

    async def aexists(self, feature: Feature, entity_id: str) -> bool:
        key = self._feature_key(feature)
        return key in self._data and entity_id in self._data[key]

    async def adelete(self, feature: Feature, entity_id: str) -> None:
        key = self._feature_key(feature)
        try:
            del self._data[key][entity_id]
        except KeyError:
            raise KeyError(f"No data for feature '{key}', entity '{entity_id}'") from None

    async def alist_entities(self, feature: Feature, prefix: str | None = None) -> list[str]:
        key = self._feature_key(feature)
        if key not in self._data:
            return []
        entities = list(self._data[key].keys())
        if prefix is not None:
            entities = [e for e in entities if e.startswith(prefix)]
        return entities

    def __repr__(self) -> str:  # pragma: no cover
        sizes = {k: len(v) for k, v in self._data.items()}
        return f"MemoryStore({sizes})"
