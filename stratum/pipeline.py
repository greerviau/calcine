"""Pipeline: ties DataSource → Feature → FeatureStore together."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .features.base import Feature
from .sources.base import DataSource
from .stores.base import FeatureStore


@dataclass
class GenerationReport:
    """Summary of a ``Pipeline.generate()`` run.

    Attributes:
        succeeded: Mapping of ``entity_id`` to extracted feature value for
            every entity that was processed without error.
        failed: Mapping of ``entity_id`` to a list of error strings for
            every entity whose processing failed (source read error,
            extraction error, or schema violation).
    """

    succeeded: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, list[str]] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        """Number of successfully processed entities."""
        return len(self.succeeded)

    @property
    def failure_count(self) -> int:
        """Number of entities that failed processing."""
        return len(self.failed)

    def __repr__(self) -> str:
        return f"GenerationReport(succeeded={self.success_count}, failed={self.failure_count})"


class Pipeline:
    """Orchestrate a featurization workflow.

    Ties together a ``DataSource``, a ``Feature``, and a ``FeatureStore``
    into a ``generate`` / ``retrieve`` interface.

    Args:
        source: Where to read raw data for each entity.
        feature: How to extract the feature from raw data.
        store: Where to persist and retrieve extracted feature values.

    Example::

        pipeline = Pipeline(
            source=DataFrameSource(df),
            feature=MeanPurchaseValue(),
            store=MemoryStore(),
        )

        report = await pipeline.generate(entity_ids=["u1", "u2"])
        print(report)  # GenerationReport(succeeded=2, failed=0)

        value = await pipeline.retrieve("u1")
    """

    def __init__(
        self,
        source: DataSource,
        feature: Feature,
        store: FeatureStore,
    ) -> None:
        self.source = source
        self.feature = feature
        self.store = store

    async def generate(
        self,
        entity_ids: list[str],
        context: dict[str, Any] | None = None,
    ) -> GenerationReport:
        """Extract and store features for a list of entities.

        For each entity the pipeline executes::

            raw = source.read(entity_id=entity_id)
            raw = feature.pre_extract(raw)
            result = feature.extract(raw, context)
            result = feature.post_extract(result)
            errors = feature.validate(result)
            if not errors:
                store.write(feature, entity_id, result)

        Individual entity failures are captured in the report — they never
        cause ``generate`` to raise.

        Args:
            entity_ids: List of entity identifiers to process.
            context: Arbitrary dict forwarded to ``Feature.extract``.
                Defaults to an empty dict.

        Returns:
            ``GenerationReport`` collecting every success and failure.
        """
        if context is None:
            context = {}

        report = GenerationReport()
        feature_name = type(self.feature).__name__

        for entity_id in entity_ids:
            try:
                # 1. Read raw data
                raw = await self.source.read(entity_id=entity_id)

                # 2. Pre-extract hook
                raw = await self.feature.pre_extract(raw)

                # 3. Extract
                result = await self.feature.extract(raw, context)

                # 4. Post-extract hook
                result = await self.feature.post_extract(result)

                # 5. Validate
                errors = await self.feature.validate(result)
                if errors:
                    report.failed[entity_id] = errors
                    continue

                # 6. Persist
                await self.store.write(self.feature, entity_id, result)
                report.succeeded[entity_id] = result

            except Exception as exc:
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                ]

        return report

    async def retrieve(self, entity_id: str) -> Any:
        """Read the stored feature value for one entity.

        Args:
            entity_id: The entity identifier.

        Returns:
            The stored feature value.

        Raises:
            KeyError: If no feature has been generated for ``entity_id``.
            StoreError: If the underlying store read fails.
        """
        return await self.store.read(self.feature, entity_id)

    async def retrieve_batch(self, entity_ids: list[str]) -> dict[str, Any]:
        """Read stored feature values for multiple entities.

        Entities with no stored value are silently omitted from the result.

        Args:
            entity_ids: List of entity identifiers.

        Returns:
            Dict mapping ``entity_id`` to feature value for every entity
            that has a stored value.
        """
        results: dict[str, Any] = {}
        for entity_id in entity_ids:
            try:
                results[entity_id] = await self.store.read(self.feature, entity_id)
            except KeyError:
                pass
        return results
