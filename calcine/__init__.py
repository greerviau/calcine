"""calcine — A source-agnostic, type-agnostic featurization pipeline framework.

Core abstraction::

    DataSource → Feature → FeatureStore

tied together by::

    Pipeline.generate(entity_ids)  →  GenerationReport   # sync default
    Pipeline.retrieve(entity_id)   →  Any                # sync default

Quick start::

    from calcine import ExtractionResult, Pipeline
    from calcine.sources import DataFrameSource
    from calcine.features.base import Feature
    from calcine.stores import MemoryStore
    from calcine.schema import FeatureSchema, types

    class MyFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw, context, entity_id=None):
            return ExtractionResult.of(entity_id, {"score": raw["value"].mean()})

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MyFeature(),
        store=MemoryStore(),
    )

    report = pipeline.generate(["e1", "e2"])
    value  = pipeline.retrieve("e1")

Implementing custom components::

    # DataSource — override read() with plain sync code
    class MySource(DataSource):
        def read(self, entity_id: str, **kwargs):
            return self.db.fetch(entity_id)

    # FeatureStore — override sync read/write/exists/delete
    class MyStore(FeatureStore):
        def read(self, feature, entity_id):
            return self.db.get(self._feature_key(feature), entity_id)

        def write(self, feature, entity_id, result, context=None):
            for sub_id, record in result.records.items():
                self.db.set(self._feature_key(feature), sub_id, record)

    # For natively async backends, override aread/awrite/aexists/adelete instead.
"""

from .exceptions import CalcineError, SchemaViolationError, SourceError, StoreError
from .extraction import ExtractionResult
from .features.base import Feature
from .pipeline import GenerationReport, Pipeline
from .schema import FeatureSchema, types
from .sources.base import DataSource
from .sources.bundle import SourceBundle
from .stores.base import FeatureStore

__version__ = "0.1.0"

__all__ = [
    # Pipeline
    "Pipeline",
    "GenerationReport",
    # Extraction
    "ExtractionResult",
    # ABCs
    "Feature",
    "DataSource",
    "SourceBundle",
    "FeatureStore",
    # Schema
    "FeatureSchema",
    "types",
    # Exceptions
    "CalcineError",
    "SchemaViolationError",
    "SourceError",
    "StoreError",
]
