"""Fan-out extraction types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FanOutResult:
    """Result of a fan-out extraction: one source entity → many stored records.

    A ``FanOutResult`` is returned by :meth:`~calcine.Feature.extract_many` to
    represent the 1:many relationship between a source entity and the sub-entity
    records it produces.

    Attributes:
        records: Mapping from sub-entity ID to extracted value.  Each value is
            validated against ``Feature.schema`` if set.  Sub-entity IDs are
            defined by the ``Feature`` implementation and are written directly as
            store keys (e.g. ``"recording_001/0"``, ``"recording_001/1"``).
        metadata: Optional parent-level data stored under the source
            ``entity_id``.  Validated against ``Feature.parent_schema`` if set.
            Also serves as the existence sentinel for ``overwrite=False`` checks:
            if ``metadata`` is ``None`` a tombstone ``{}`` is written instead so
            that incremental generation can detect whether a parent entity has
            already been processed.
    """

    records: dict[str, Any]
    metadata: dict[str, Any] | None = None
