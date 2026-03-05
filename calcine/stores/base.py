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

    Only ``aread`` is abstract — subclasses must implement it.  The remaining
    operations (``awrite``, ``aexists``, ``adelete``) have default implementations
    that raise ``NotImplementedError``, so read-only stores (e.g. a production
    serving layer that you can query but not write to) only need to override
    ``aread``.  Full read-write stores should override all four.

    Async methods (``aread``, ``awrite``, ``aexists``, ``adelete``) are the
    implementation interface.  Sync wrappers (``read``, ``write``, ``exists``,
    ``delete``) are provided for use outside an async context.

    ``awrite`` always receives an :class:`~calcine.ExtractionResult`.  For
    single-record features, ``result.records`` contains exactly one entry keyed
    by ``entity_id``.  For fan-out features, ``result.records`` contains the
    sub-entity entries and ``result.metadata`` (if set) is stored under
    ``entity_id``.  Implementations should write all records and the metadata
    (or a ``{}`` tombstone when metadata is ``None`` and ``entity_id`` is not
    already in ``result.records``) to support ``overwrite=False`` existence
    checks.

    Minimal read-only example::

        class RemoteFeatureStore(FeatureStore):
            async def aread(self, feature, entity_id):
                return await fetch_from_remote(feature, entity_id)

    Full read-write example::

        class RedisStore(FeatureStore):
            def __init__(self, redis_client):
                self.redis = redis_client

            async def awrite(self, feature, entity_id, result, context=None):
                # Write metadata / tombstone under parent key
                if entity_id not in result.records:
                    parent = result.metadata if result.metadata is not None else {}
                    key = f"{self._feature_key(feature)}:{entity_id}"
                    await self.redis.set(key, pickle.dumps(parent))
                # Write each record
                for sub_id, record in result.records.items():
                    key = f"{self._feature_key(feature)}:{sub_id}"
                    await self.redis.set(key, pickle.dumps(record))

            async def aread(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                raw = await self.redis.get(key)
                if raw is None:
                    raise KeyError(key)
                return pickle.loads(raw)

            async def aexists(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                return bool(await self.redis.exists(key))

            async def adelete(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                if not await self.redis.delete(key):
                    raise KeyError(key)
    """

    async def awrite(
        self,
        feature: Feature,
        entity_id: str,
        result: ExtractionResult,
        context: dict | None = None,
    ) -> None:
        """Persist an extraction result for a source entity.

        Writes metadata (or a ``{}`` tombstone) under ``entity_id`` when
        ``entity_id`` is not already a key in ``result.records``, then writes
        each record in ``result.records`` under its own key.  This ensures
        ``aexists(feature, entity_id)`` returns ``True`` after any write,
        enabling ``overwrite=False`` to work correctly for both single-record
        and fan-out features.

        Args:
            feature: The ``Feature`` instance (class name used as namespace).
            entity_id: Source entity identifier (parent write key).
            result: :class:`~calcine.ExtractionResult` from ``Feature.extract``.
            context: The pipeline context dict at write time.  Implementations
                may use this for routing (e.g. sharding writes by region) while
                keeping ``aread`` context-free.  Ignored by default.

        Raises:
            NotImplementedError: If this store is read-only.
            StoreError: If the write operation fails.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support write operations.")

    @abstractmethod
    async def aread(self, feature: Feature, entity_id: str) -> Any:
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

    async def aexists(self, feature: Feature, entity_id: str) -> bool:
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

    async def adelete(self, feature: Feature, entity_id: str) -> None:
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

    async def alist_entities(self, feature: Feature, prefix: str | None = None) -> list[str]:
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
    # Synchronous convenience wrappers
    # ------------------------------------------------------------------

    def write(
        self,
        feature: Feature,
        entity_id: str,
        result: ExtractionResult,
        context: dict | None = None,
    ) -> None:
        """Blocking version of :meth:`awrite` for use outside an async context."""
        return asyncio.run(self.awrite(feature, entity_id, result, context))

    def read(self, feature: Feature, entity_id: str) -> Any:
        """Blocking version of :meth:`aread` for use outside an async context."""
        return asyncio.run(self.aread(feature, entity_id))

    def exists(self, feature: Feature, entity_id: str) -> bool:
        """Blocking version of :meth:`aexists` for use outside an async context."""
        return asyncio.run(self.aexists(feature, entity_id))

    def delete(self, feature: Feature, entity_id: str) -> None:
        """Blocking version of :meth:`adelete` for use outside an async context."""
        return asyncio.run(self.adelete(feature, entity_id))

    def list_entities(self, feature: Feature, prefix: str | None = None) -> list[str]:
        """Blocking version of :meth:`alist_entities` for use outside an async context."""
        return asyncio.run(self.alist_entities(feature, prefix))

    def _feature_key(self, feature: Feature) -> str:
        """Return a stable string namespace key for a feature instance.

        Uses the class name.  Override to customise namespacing.
        """
        return type(feature).__name__
