"""Abstract base class for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from ..schema import FeatureSchema

if TYPE_CHECKING:
    from ..fanout import FanOutResult


class Feature(ABC):
    """Contract for all feature extractors in the calcine pipeline.

    A ``Feature`` transforms raw data (from a ``DataSource``) into a
    structured, typed feature value.

    Subclasses must implement :meth:`extract`.  :meth:`validate` runs
    automatically after extraction if ``schema`` is set.

    The ``schema`` class attribute, if set to a ``FeatureSchema``, enables
    automatic validation of the extracted result.

    For 1:many fan-out extraction (one source entity → many stored sub-entity
    records), override :meth:`extract_many` instead of ``extract`` and
    optionally set ``parent_schema`` for parent-level metadata validation.

    Example::

        class EmbeddingFeature(Feature):
            schema = FeatureSchema({"vec": types.NDArray(shape=(128,), dtype="float32")})

            async def extract(self, raw: str, context: dict) -> dict:
                return {"vec": encode(raw)}

    Fan-out example::

        class SegmentFeature(Feature):
            parent_schema = FeatureSchema({"duration_s": types.Float64(nullable=False)})
            schema = FeatureSchema({"rms": types.Float64(nullable=False)})

            async def extract_many(self, raw, context, entity_id) -> FanOutResult:
                segments = split(raw)
                return FanOutResult(
                    metadata={"duration_s": len(raw) / SR},
                    records={f"{entity_id}/{i}": {"rms": rms(s)} for i, s in enumerate(segments)},
                )

            async def extract(self, raw, context, entity_id=None):
                raise NotImplementedError  # pipeline uses extract_many
    """

    schema: ClassVar[FeatureSchema | None] = None
    parent_schema: ClassVar[FeatureSchema | None] = None

    @abstractmethod
    async def extract(self, raw: Any, context: dict, entity_id: str | None = None) -> Any:
        """Extract the feature value from raw source data.

        Args:
            raw: Raw data returned by the ``DataSource``.  The type depends
                on the source implementation.
            context: Arbitrary dict supplied by the caller (e.g. timestamps,
                model version, experiment flags).
            entity_id: The identifier of the entity being processed.  Useful
                for logging, per-entity branching, or keying external lookups.
                ``None`` when called outside a ``Pipeline`` context.

        Returns:
            Extracted feature value.  Should conform to ``schema`` if set.
        """
        ...

    async def extract_many(self, raw: Any, context: dict, entity_id: str) -> FanOutResult:
        """Fan-out extraction: one source entity → many stored sub-entity records.

        Override this instead of :meth:`extract` when a single source entity
        produces multiple independently-stored sub-entity records (e.g. audio
        file → segments, document → chunks, session log → events).

        When this method is overridden, the pipeline routes through the fan-out
        path and never calls :meth:`extract`.  Fan-out features may implement
        ``extract`` as ``raise NotImplementedError``.

        Args:
            raw: Raw data from the source.
            context: Pipeline context dict.
            entity_id: The source entity identifier.  Use this to build
                sub-entity IDs (e.g. ``f"{entity_id}/{i}"``).

        Returns:
            :class:`~calcine.FanOutResult` with ``records`` (sub-entity ID →
            value, each validated against ``schema``) and optional ``metadata``
            (validated against ``parent_schema``).

        Raises:
            NotImplementedError: Base implementation — override to enable
                fan-out extraction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_many. "
            "Override extract_many to enable fan-out extraction."
        )

    async def extract_batch(
        self,
        raws: list[Any],
        context: dict[str, Any],
        entity_ids: list[str] | None = None,
        entity_contexts: list[dict[str, Any]] | None = None,
    ) -> list[Any | BaseException]:
        """Extract features for a batch of entities in a single call.

        Override this to enable vectorised or batch-API computation
        (e.g. ML model inference, bulk database queries, batch embedding
        APIs).  The default implementation calls ``extract()`` for each
        item individually and is therefore equivalent to — but no faster
        than — the per-entity path.

        Return one element per input, in the same order.  Individual items
        may be ``BaseException`` instances to signal per-entity failure
        without aborting the rest of the batch; the pipeline will record
        those entities as failed and continue.

        Args:
            raws: Raw data for each entity in the batch.
            context: Shared context dict forwarded from ``generate()``.
            entity_ids: Entity identifiers corresponding to each item in
                *raws*, in the same order.  ``None`` when called outside
                a ``Pipeline`` context.
            entity_contexts: Per-entity context dicts (already merged with
                the shared *context*), one per item in *raws*.  Present when
                ``generate()`` is called with ``context_fn``.  ``None``
                otherwise — fall back to *context* for all entities.

        Returns:
            List of results or ``BaseException`` instances, one per input.
        """
        results: list[Any | BaseException] = []
        for i, raw in enumerate(raws):
            eid = entity_ids[i] if entity_ids is not None else None
            ctx = entity_contexts[i] if entity_contexts is not None else context
            try:
                results.append(await self.extract(raw, ctx, entity_id=eid))
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results

    async def validate(self, result: Any) -> list[str]:
        """Validate the (post-processed) extraction result.

        Uses ``schema.validate`` if ``schema`` is set; otherwise returns
        an empty list (no validation).

        Args:
            result: Extracted feature value.

        Returns:
            List of validation error strings.  Empty means valid.
        """
        if self.schema is not None:
            return self.schema.validate(result)
        return []
