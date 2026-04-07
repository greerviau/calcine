"""Abstract base class for feature stores."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..extraction import ExtractionResult

if TYPE_CHECKING:
    from ..features.base import Feature


class FeatureStore(ABC):
    """Contract for all feature stores in the calcine pipeline.

    A ``FeatureStore`` persists and retrieves extracted feature values.
    All operations are keyed by ``(feature, entity_id)`` where the feature
    type name is used as the namespace.

    Implement the synchronous methods (``read``, ``write``, ``exists``,
    ``delete``).  The framework calls the async variants (``aread``,
    ``awrite``, ``aexists``, ``adelete``) internally; by default these run
    your sync implementations in a thread executor.

    For natively async backends (e.g. async database clients), override the
    async methods directly instead.

    ``write`` always receives an :class:`~calcine.ExtractionResult`.  For
    single-record features, ``result.records`` contains exactly one entry keyed
    by ``entity_id``.  For fan-out features, ``result.records`` contains the
    sub-entity entries and ``result.metadata`` (if set) is stored under
    ``entity_id``.  Implementations should write all records and the metadata
    (or a ``{}`` tombstone when metadata is ``None`` and ``entity_id`` is not
    already in ``result.records``) to support ``overwrite=False`` existence
    checks.

    Minimal read-only example::

        class RemoteFeatureStore(FeatureStore):
            def read(self, feature, entity_id):
                return fetch_from_remote(feature, entity_id)

    Full read-write example::

        class RedisStore(FeatureStore):
            def __init__(self, redis_client):
                self.redis = redis_client

            def write(self, feature, entity_id, result, context=None):
                if entity_id not in result.records:
                    parent = result.metadata if result.metadata is not None else {}
                    key = f"{self._feature_key(feature)}:{entity_id}"
                    self.redis.set(key, pickle.dumps(parent))
                for sub_id, record in result.records.items():
                    key = f"{self._feature_key(feature)}:{sub_id}"
                    self.redis.set(key, pickle.dumps(record))

            def read(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                raw = self.redis.get(key)
                if raw is None:
                    raise KeyError(key)
                return pickle.loads(raw)

            def exists(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                return bool(self.redis.exists(key))

            def delete(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                if not self.redis.delete(key):
                    raise KeyError(key)

    For a natively async backend, override the async methods instead::

        class AsyncRedisStore(FeatureStore):
            async def awrite(self, feature, entity_id, result, context=None):
                ...

            async def aread(self, feature, entity_id):
                ...

            async def aexists(self, feature, entity_id):
                ...

            async def adelete(self, feature, entity_id):
                ...
    """

    # ------------------------------------------------------------------
    # Sync implementation interface — override these
    # ------------------------------------------------------------------

    @abstractmethod
    def read(self, feature: Feature, entity_id: str) -> Any:
        """Retrieve a stored feature value for an entity.

        Args:
            feature: The ``Feature`` instance.
            entity_id: Unique entity identifier.

        Returns:
            Previously stored feature value.

        Raises:
            KeyError: If no value exists for ``(feature, entity_id)``.
            StoreError: If the read operation fails.
        """
        ...

    def write(
        self,
        feature: Feature,
        entity_id: str,
        result: ExtractionResult,
        context: dict | None = None,
    ) -> None:
        """Persist an extraction result for a source entity.

        Args:
            feature: The ``Feature`` instance (class name used as namespace).
            entity_id: Source entity identifier (parent write key).
            result: :class:`~calcine.ExtractionResult` from ``Feature.extract``.
            context: The pipeline context dict at write time.

        Raises:
            NotImplementedError: If this store is read-only.
            StoreError: If the write operation fails.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support write operations.")

    def exists(self, feature: Feature, entity_id: str) -> bool:
        """Check whether a feature value exists for an entity.

        Args:
            feature: The ``Feature`` instance.
            entity_id: Unique entity identifier.

        Returns:
            ``True`` if a value is stored, ``False`` otherwise.

        Raises:
            NotImplementedError: If this store does not support existence checks.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support exists operations.")

    def delete(self, feature: Feature, entity_id: str) -> None:
        """Delete the stored feature value for an entity.

        Args:
            feature: The ``Feature`` instance.
            entity_id: Unique entity identifier.

        Raises:
            NotImplementedError: If this store does not support deletion.
            KeyError: If no value exists for ``(feature, entity_id)``.
            StoreError: If the delete operation fails.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support delete operations.")

    def list_entities(self, feature: Feature, prefix: str | None = None) -> list[str]:
        """Return entity IDs stored for a feature, optionally filtered by prefix.

        Args:
            feature: The ``Feature`` instance.
            prefix: When given, only entity IDs starting with this string are
                returned.  Primary use-case is discovering sub-entities produced
                by fan-out extraction (e.g. ``prefix="recording_001/"``).

        Returns:
            List of matching entity IDs in arbitrary order.

        Raises:
            NotImplementedError: If this store does not support entity listing.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support list_entities.")

    # ------------------------------------------------------------------
    # Async interface — wraps sync by default; override for native async
    # ------------------------------------------------------------------

    async def aread(self, feature: Feature, entity_id: str) -> Any:
        """Async version of ``read``.  Runs ``read()`` in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.read, feature, entity_id)

    async def awrite(
        self,
        feature: Feature,
        entity_id: str,
        result: ExtractionResult,
        context: dict | None = None,
    ) -> None:
        """Async version of ``write``.  Runs ``write()`` in a thread executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.write, feature, entity_id, result, context)

    async def aexists(self, feature: Feature, entity_id: str) -> bool:
        """Async version of ``exists``.  Runs ``exists()`` in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.exists, feature, entity_id)

    async def adelete(self, feature: Feature, entity_id: str) -> None:
        """Async version of ``delete``.  Runs ``delete()`` in a thread executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.delete, feature, entity_id)

    async def alist_entities(self, feature: Feature, prefix: str | None = None) -> list[str]:
        """Async version of ``list_entities``.  Runs ``list_entities()`` in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.list_entities(feature, prefix))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _feature_key(self, feature: Feature) -> str:
        """Return a stable string namespace key for a feature instance.

        Uses the class name.  Override to customise namespacing.
        """
        return type(feature).__name__
