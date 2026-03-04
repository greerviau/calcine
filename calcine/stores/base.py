"""Abstract base class for feature stores."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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

    Minimal read-only example::

        class RemoteFeatureStore(FeatureStore):
            async def aread(self, feature, entity_id):
                return await fetch_from_remote(feature, entity_id)

    Full read-write example::

        class RedisStore(FeatureStore):
            def __init__(self, redis_client):
                self.redis = redis_client

            async def awrite(self, feature, entity_id, data, context=None):
                key = f"{self._feature_key(feature)}:{entity_id}"
                await self.redis.set(key, pickle.dumps(data))

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
        data: Any,
        context: dict | None = None,
    ) -> None:
        """Persist a feature value for an entity.

        Args:
            feature: The ``Feature`` instance (class name used as namespace).
            entity_id: Unique entity identifier.
            data: Extracted feature value to store.
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

    # ------------------------------------------------------------------
    # Synchronous convenience wrappers
    # ------------------------------------------------------------------

    def write(
        self,
        feature: Feature,
        entity_id: str,
        data: Any,
        context: dict | None = None,
    ) -> None:
        """Blocking version of :meth:`awrite` for use outside an async context."""
        return asyncio.run(self.awrite(feature, entity_id, data, context))

    def read(self, feature: Feature, entity_id: str) -> Any:
        """Blocking version of :meth:`aread` for use outside an async context."""
        return asyncio.run(self.aread(feature, entity_id))

    def exists(self, feature: Feature, entity_id: str) -> bool:
        """Blocking version of :meth:`aexists` for use outside an async context."""
        return asyncio.run(self.aexists(feature, entity_id))

    def delete(self, feature: Feature, entity_id: str) -> None:
        """Blocking version of :meth:`adelete` for use outside an async context."""
        return asyncio.run(self.adelete(feature, entity_id))

    def _feature_key(self, feature: Feature) -> str:
        """Return a stable string namespace key for a feature instance.

        Uses the class name.  Override to customise namespacing.
        """
        return type(feature).__name__
