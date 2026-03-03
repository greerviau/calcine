"""calcine hello world: async user-engagement feature extraction.

This example demonstrates where calcine earns its keep over a plain pandas
one-liner:

  - A real async DataSource (simulates a database or API call per entity)
  - Schema-validated, multi-field features that catch bad data before it
    reaches the store
  - Per-entity error isolation: some users fail; the rest always succeed
  - Concurrent extraction (concurrency=8 fires all reads in parallel)
  - Incremental generation: a second run with overwrite=False skips entities
    that already have stored values

Run with:
    python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from calcine import Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources.base import DataSource
from calcine.stores import MemoryStore

# ---------------------------------------------------------------------------
# Simulated async database
# ---------------------------------------------------------------------------
# In a real pipeline this would be an async DB driver, an HTTP API, S3, etc.
# The key point: each entity's data is fetched independently and asynchronously.

_USER_DB: dict[str, dict[str, Any]] = {
    "u01": {"total_spend": 1240.50, "event_count": 87,  "days_active": 120},
    "u02": {"total_spend": 89.99,   "event_count": 12,  "days_active": 14},
    "u03": {"total_spend": 4320.00, "event_count": 412, "days_active": 365},
    "u04": {"total_spend": 0.0,     "event_count": 0,   "days_active": 1},
    "u05": {"total_spend": 760.10,  "event_count": 55,  "days_active": 60},
    "u06": {"total_spend": None,    "event_count": 31,  "days_active": 30},  # bad data
    "u07": {"total_spend": 2100.00, "event_count": 190, "days_active": 200},
    "u08": {"total_spend": 330.00,  "event_count": 28,  "days_active": 45},
    "u09": {"total_spend": 980.00,  "event_count": 72,  "days_active": 90},
    "u10": {"total_spend": 15.00,   "event_count": 3,   "days_active": 5},
    # u11 and u12 have no DB record at all → source raises KeyError
}


class UserDBSource(DataSource):
    """Fetch a single user's activity record from the (simulated) database."""

    async def read(self, entity_id: str, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.05)  # simulate ~50 ms network round-trip
        if entity_id not in _USER_DB:
            raise KeyError(f"No record found for user '{entity_id}'")
        return _USER_DB[entity_id]


# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

_SPEND_TIERS = ["low", "mid", "high", "whale"]


class UserEngagementFeature(Feature):
    """Derive validated engagement metrics from raw user activity data."""

    schema = FeatureSchema({
        "spend_tier":  types.Category(categories=_SPEND_TIERS, nullable=False),
        "event_rate":  types.Float64(nullable=False),   # events per active day
        "total_spend": types.Float64(nullable=False),
    })

    async def extract(self, raw: dict, context: dict, entity_id: str | None = None) -> dict:
        spend = raw["total_spend"]
        if spend is None:
            raise ValueError("total_spend is missing — cannot compute engagement features")

        event_rate = raw["event_count"] / max(raw["days_active"], 1)

        if spend < 100:
            tier = "low"
        elif spend < 1000:
            tier = "mid"
        elif spend < 3000:
            tier = "high"
        else:
            tier = "whale"

        return {
            "spend_tier":  tier,
            "event_rate":  round(event_rate, 4),
            "total_spend": spend,
        }


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

ALL_USERS = [f"u{i:02d}" for i in range(1, 13)]  # u01 … u12

pipeline = Pipeline(
    source=UserDBSource(),
    feature=UserEngagementFeature(),
    store=MemoryStore(),
)


async def main() -> None:
    # --- Run 1: first 8 users, 8 concurrent reads ----------------------------
    print("=== Run 1: first 8 users (concurrency=8) ===")
    report = await pipeline.generate(
        entity_ids=ALL_USERS[:8],
        concurrency=8,
    )
    print(report)
    print(f"  succeeded : {sorted(report.succeeded)}")
    print(f"  failed    : {dict(report.failed)}")
    # u06 fails (total_spend is None); u01-u05, u07-u08 succeed

    # --- Run 2: all 12 users, skip already-stored ----------------------------
    print("\n=== Run 2: all 12 users, overwrite=False ===")
    report2 = await pipeline.generate(
        entity_ids=ALL_USERS,
        concurrency=8,
        overwrite=False,
    )
    print(report2)
    print(f"  skipped   : {sorted(report2.skipped)}")   # already in store
    print(f"  succeeded : {sorted(report2.succeeded)}")  # newly processed
    print(f"  failed    : {dict(report2.failed)}")
    # u01-u05, u07-u08 skipped; u09-u10 succeed; u06, u11, u12 fail

    # --- Retrieve results ----------------------------------------------------
    print("\n=== Retrieving stored features ===")
    batch = await pipeline.retrieve_batch(["u01", "u03", "u07", "u09"])
    for uid, feat in batch.items():
        print(f"  {uid}: {feat}")


if __name__ == "__main__":
    asyncio.run(main())
