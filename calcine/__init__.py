"""calcine — A source-agnostic, type-agnostic featurization pipeline framework.

Core abstraction::

    DataSource → Feature → FeatureStore

tied together by::

    Pipeline.generate(entity_ids)  →  GenerationReport   # sync default
    Pipeline.retrieve(entity_id)   →  Any                # sync default

Quick start::

    from calcine import Pipeline
    from calcine.sources import DataFrameSource
    from calcine.features.base import Feature
    from calcine.stores import MemoryStore
    from calcine.schema import FeatureSchema, types

    class MyFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw, context):
            return {"score": raw["value"].mean()}

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MyFeature(),
        store=MemoryStore(),
    )

    report = pipeline.generate(["e1", "e2"])
    value  = pipeline.retrieve("e1")
"""

from .exceptions import CalcineError, SchemaViolationError, SourceError, StoreError
from .fanout import FanOutResult
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
    # Fan-out
    "FanOutResult",
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
