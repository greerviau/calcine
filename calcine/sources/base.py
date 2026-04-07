"""Abstract base class for data sources."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any


class DataSource:
    """Contract for all data sources in the calcine pipeline.

    A ``DataSource`` is responsible for fetching raw data for a given entity.
    The raw data format is completely source-defined — it might be a pandas
    DataFrame, raw bytes, a JSON dict, etc.

    Implement ``read`` for simple synchronous sources.  The framework runs it
    in a thread executor so it never blocks the event loop:

    Example implementation::

        class MyDBSource(DataSource):
            def __init__(self, conn):
                self.conn = conn

            def read(self, entity_id: str, **kwargs) -> list[dict]:
                return self.conn.execute(
                    "SELECT * FROM events WHERE user_id = ?", (entity_id,)
                ).fetchall()

    For sources that are natively async (e.g. async database clients, HTTP),
    override ``aread`` instead::

        class AsyncDBSource(DataSource):
            def __init__(self, conn):
                self.conn = conn

            async def aread(self, entity_id: str, **kwargs) -> list[dict]:
                return await self.conn.fetchall(
                    "SELECT * FROM events WHERE user_id = $1", entity_id
                )
    """

    def read(self, **kwargs: Any) -> Any:
        """Read data for a single entity from the source (sync).

        Override this for simple synchronous sources.  The framework
        automatically runs this in a thread executor via ``aread``.

        Args:
            **kwargs: Source-specific parameters.  Implementations should
                accept ``entity_id: str`` at minimum.

        Returns:
            Raw data in the source's native format.

        Raises:
            SourceError: If the source cannot fulfill the request.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement read() or aread()")

    async def aread(self, **kwargs: Any) -> Any:
        """Read data for a single entity from the source (async).

        The default implementation runs ``read()`` in a thread executor so
        blocking I/O does not stall the event loop.

        Override this instead of ``read()`` for natively async sources.

        Args:
            **kwargs: Source-specific parameters, forwarded to ``read()``.

        Returns:
            Raw data in the source's native format.

        Raises:
            SourceError: If the source cannot fulfill the request.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.read(**kwargs))

    async def stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream data as an async iterator.

        The default implementation wraps ``aread()`` as a single-item stream.
        Override this for large sources where incremental reading is more
        efficient.

        Args:
            **kwargs: Source-specific parameters, forwarded to ``aread()``.

        Yields:
            Data items from the source.
        """
        result = await self.aread(**kwargs)
        yield result
