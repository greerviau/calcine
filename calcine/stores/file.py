"""File-based feature store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import StoreError
from ..extraction import ExtractionResult
from ..serializers import PickleSerializer, Serializer
from .base import FeatureStore

if TYPE_CHECKING:
    from ..features.base import Feature


class FileStore(FeatureStore):
    """Persist feature values as individual files on the local filesystem.

    Directory layout::

        {base_path}/
            {FeatureClassName}/
                {entity_id}.bin

    Fan-out sub-entities with ``/`` in their IDs (e.g. ``"recording/0"``)
    are stored in subdirectories automatically (``recording/0.bin``).

    Args:
        path: Base directory where feature files are stored.  Created
            automatically on first write.
        serializer: Serializer to use for encoding values.  Defaults to
            ``PickleSerializer``.

    Example::

        store = FileStore("/tmp/features", serializer=JSONSerializer())
        store.write(feature, "u1", ExtractionResult.of("u1", {"score": 0.92}))
    """

    def __init__(self, path: str, serializer: Serializer | None = None) -> None:
        self.path = Path(path)
        self.serializer: Serializer = serializer if serializer is not None else PickleSerializer()

    def _entity_path(self, feature: Feature, entity_id: str) -> Path:
        return self.path / self._feature_key(feature) / f"{entity_id}.bin"

    def _write_single(self, feature: Feature, entity_id: str, data: Any) -> None:
        path = self._entity_path(feature, entity_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.serializer.serialize(data))
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc

    def write(
        self,
        feature: Feature,
        entity_id: str,
        result: ExtractionResult,
        context: dict | None = None,
    ) -> None:
        # Write metadata (or tombstone) under entity_id when it isn't already
        # a record key — ensures exists(entity_id) is True after any write.
        if entity_id not in result.records:
            parent = result.metadata if result.metadata is not None else {}
            self._write_single(feature, entity_id, parent)
        for sub_id, record in result.records.items():
            self._write_single(feature, sub_id, record)

    def read(self, feature: Feature, entity_id: str) -> Any:
        path = self._entity_path(feature, entity_id)
        if not path.exists():
            raise KeyError(
                f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
            )
        try:
            return self.serializer.deserialize(path.read_bytes())
        except KeyError:
            raise
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc

    def exists(self, feature: Feature, entity_id: str) -> bool:
        return self._entity_path(feature, entity_id).exists()

    def delete(self, feature: Feature, entity_id: str) -> None:
        path = self._entity_path(feature, entity_id)
        if not path.exists():
            raise KeyError(
                f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
            )
        try:
            path.unlink()
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc
