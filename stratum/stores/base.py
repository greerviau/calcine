"""Abstract base class for feature stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..features.base import Feature


class FeatureStore(ABC):
    """Contract for all feature stores in the stratum pipeline.

    A ``FeatureStore`` persists and retrieves extracted feature values.
    All operations are keyed by ``(feature, entity_id)`` where the feature
    type name is used as the namespace.

    Implementations must provide ``write``, ``read``, ``exists``, and
    ``delete``.  All are async to allow non-blocking I/O.

    Example implementation::

        class RedisStore(FeatureStore):
            def __init__(self, redis_client):
                self.redis = redis_client

            async def write(self, feature, entity_id, data):
                key = f"{self._feature_key(feature)}:{entity_id}"
                await self.redis.set(key, pickle.dumps(data))

            async def read(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                raw = await self.redis.get(key)
                if raw is None:
                    raise KeyError(key)
                return pickle.loads(raw)

            async def exists(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                return bool(await self.redis.exists(key))

            async def delete(self, feature, entity_id):
                key = f"{self._feature_key(feature)}:{entity_id}"
                if not await self.redis.delete(key):
                    raise KeyError(key)
    """

    @abstractmethod
    async def write(self, feature: "Feature", entity_id: str, data: Any) -> None:
        """Persist a feature value for an entity.

        Args:
            feature: The ``Feature`` instance (class name used as namespace).
            entity_id: Unique entity identifier.
            data: Extracted feature value to store.

        Raises:
            StoreError: If the write operation fails.
        """
        ...

    @abstractmethod
    async def read(self, feature: "Feature", entity_id: str) -> Any:
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

    @abstractmethod
    async def exists(self, feature: "Feature", entity_id: str) -> bool:
        """Check whether a feature value exists for an entity.

        Args:
            feature: The ``Feature`` instance.
            entity_id: Unique entity identifier.

        Returns:
            ``True`` if a value is stored, ``False`` otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, feature: "Feature", entity_id: str) -> None:
        """Delete the stored feature value for an entity.

        Args:
            feature: The ``Feature`` instance.
            entity_id: Unique entity identifier.

        Raises:
            KeyError: If no value exists for ``(feature, entity_id)``.
            StoreError: If the delete operation fails.
        """
        ...

    def _feature_key(self, feature: "Feature") -> str:
        """Return a stable string namespace key for a feature instance.

        Uses the class name.  Override to customise namespacing.
        """
        return type(feature).__name__
