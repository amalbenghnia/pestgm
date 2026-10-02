"""
Build the PV physical reference dataset (CDC continuation Phase 3-8).

    data/processed/pv_production_reference_15min.csv

from:
  1. data/raw/pv_fleet_prosol_3_snapshots.csv (REAL)
  2. PVGIS real satellite irradiance + PV simulation (src/ingestion/pvgis_provider.py)
  3. pvlib physical modelling (solar geometry, temperature correction)
  4. real fleet capacity evolution (src/ingestion/capacity_timeseries.py)

STRICT RULE (per this phase's own instructions): if PVGIS cannot be reached,
this script does NOT fabricate a "physical reference" from synthetic weather
and mislabel it as PVGIS-derived. It stops, reports the exact network
failure, and leaves `production_target.mode` at `DEMO_PROXY` in
configs/config.yaml untouched - the existing, honestly-labelled synthetic
fallback remains the only demo-mode target until this script can actually
reach PVGIS.

Usage (numbered 10 rather than 09 - 09 is already used by
scripts/09_enstab_external_validation.py in this repo):
    python scripts/10_build_pv_reference_dataset.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.fleet_loader import load_fleet_csv
from src.ingestion.pvgis_provider import PVGISProvider, PVGIS_SOURCE_LABEL
from src.ingestion.capacity_timeseries import build_capacity_timeseries

FLEET_PATH = "data/raw/pv_fleet_prosol_3_snapshots.csv"
OUT_PATH = "data/processed/pv_production_reference_15min.csv"
REPORT_PATH = "docs/PRODUCTION_REFERENCE_DATASET_REPORT.md"


def attempt_pvgis_for_all_districts(districts: pd.DataFrame, start_year: int, end_year: int) -> tuple[list[pd.DataFrame], dict]:
    """Tries PVGIS for every district. Returns (successful_frames, status_dict).
    Stops at the FIRST failure and reports it - does not silently continue
    with partial real + partial fabricated data for the remaining districts,
    since a dataset that is real for some districts and invented for others
    without a per-row flag would be worse than stopping cleanly."""
    provider = PVGISProvider()
    frames = []
    status = {"attempted": 0, "succeeded": 0, "first_failure": None}
    for _, row in districts.iterrows():
        status["attempted"] += 1
        try:
            df = provider.get_reference_series(
                row["district_id"], row["district"], row["governorate"],
                row["latitude"], row["longitude"], start_year, end_year,
                peakpower_kwc=1.0, use_cache=True,
            )
            frames.append(df)
            status["succeeded"] += 1
        except RuntimeError as e:
            status["first_failure"] = {
                "district_id": row["district_id"], "district": row["district"], "error": str(e),
            }
            break  # stop immediately per the strict rule above
    return frames, status


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=2020,
                     help="PVGIS historical coverage typically runs up to ~2020 for SARAH-2 - pick a real past "
                          "year here, NOT the fictional 2026 dates used by the synthetic demo pipeline.")
    ap.add_argument("--end_year", type=int, default=2020)
    args = ap.parse_args()

    fleet = load_fleet_csv(FLEET_PATH)
    districts = fleet.drop_duplicates("district_id")[
        ["district_id", "district", "governorate", "latitude", "longitude"]
    ]
    print(f"1. Loaded {len(districts)} districts from real Prosol fleet data.")

    print(f"2. Attempting PVGIS retrieval ({args.start_year}-{args.end_year}) for all districts "
          f"(real satellite irradiance)...")
    frames, status = attempt_pvgis_for_all_districts(districts, args.start_year, args.end_year)
    print(f"   PVGIS attempted={status['attempted']} succeeded={status['succeeded']}")

    Path("docs").mkdir(exist_ok=True)

    if status["succeeded"] < len(districts):
        # STOP per the strict rule - report and exit, do not fabricate.
        failure = status["first_failure"]
        report = f"""# PRODUCTION_REFERENCE_DATASET_REPORT.md

## Status: BLOCKED - PVGIS unreachable from this environment

Attempted PVGIS retrieval for all {len(districts)} districts using the real
Prosol fleet coordinates. Succeeded for {status['succeeded']}/{len(districts)}
before stopping at the first failure (per the strict no-partial-fabrication
rule: mixing real PVGIS rows for some districts with any kind of substitute
for others, in the same file, without the substitute being clearly
impossible to confuse with the real rows, was judged too easy to
misinterpret downstream - so this script stops cleanly instead).

**First failure:**
- District: `{failure['district_id']}` ({failure['district']})
- Error: `{failure['error']}`

This matches the network behaviour already documented in
`docs/REAL_PRODUCTION_DATA_SOURCES.md` and `docs/PRODUCTION_DATA_SEARCH_REPORT.md`:
`re.jrc.ec.europa.eu` is not reachable from this build sandbox (confirmed
again on {datetime.now(timezone.utc).date().isoformat()}, this session).

## What this means for the training target

`configs/config.yaml`'s `production_target.mode` **remains `DEMO_PROXY`** -
this script does not touch it, since no real PVGIS-derived reference was
actually obtained. The existing, honestly-labelled `SYNTHETIC_CLEARSKY` /
`DEMO_PROXY_PHYSICS_SIMULATION` pipeline (`scripts/02`, `scripts/04`) remains
the only training data source in this environment.

## What is ready to run elsewhere

`src/ingestion/pvgis_provider.py` (the client) and this orchestration script
are both complete and untested-only-because-unreachable. From any environment
where `re.jrc.ec.europa.eu` responds (a laptop, a CI runner, a cloud VM):

```bash
python scripts/10_build_pv_reference_dataset.py
```

will retrieve real hourly GHI + PVGIS's own physical PV simulation for all
50 districts, cache each response under `data/external/pvgis_cache/`, scale
by real fleet capacity, and write
`data/processed/pv_production_reference_15min.csv` with
`production_source=PVGIS_PHYSICAL_ESTIMATE_REAL_IRRADIANCE` - never
"measured". At that point `production_target.mode` should be manually
updated to `PHYSICAL_REFERENCE` in `configs/config.yaml`, and
`scripts/04_build_training_dataset.py` extended to read from this file
instead of generating the synthetic proxy (not done in this script, to keep
the "stop and report" behaviour minimal and unambiguous). Hourly-to-15-minute
disaggregation (preserving daily energy, per this phase's §5) is also left
for that follow-up step, since PVGIS's own `seriescalc` output is hourly and
disaggregating unreachable data has nothing to validate against yet.

## Real ground-truth support for using PVGIS here once reachable

See `docs/PRODUCTION_DATA_SEARCH_REPORT.md §B`: two independent, real,
Tunisia-specific studies validate PVGIS-equivalent irradiance estimates
locally within 4-14% error (Borj-Cedria, INM Bizerte/Nabeul/Djerba).
"""
        Path(REPORT_PATH).write_text(report)
        print(f"\nSTOPPED (as designed). Wrote {REPORT_PATH}")
        print(f"First failure: {json.dumps(failure, indent=2)}")
        sys.exit(1)

    # --- Only reached if PVGIS actually succeeded for every district ---
    print("3. All districts retrieved from PVGIS. Building capacity time series...")
    pvgis_df = pd.concat(frames, ignore_index=True)
    ts_min, ts_max = pvgis_df["timestamp"].min(), pvgis_df["timestamp"].max()
    # NOTE: PVGIS's historical irradiance necessarily predates the Dec-2025/
    # Mar-2026/Jul-2026 Prosol snapshots. build_capacity_timeseries's documented
    # "hold_before_first" rule therefore applies here: every PVGIS timestamp gets
    # the Dec-2025 capacity value. This is a deliberate, stated approximation -
    # "what would today's installed fleet have produced under real historical
    # weather" - not a claim that the fleet was that size in 2023.
    cap_ts = build_capacity_timeseries(fleet, freq="1h", start=ts_min, end=ts_max)
    print(f"   Capacity approximation note: PVGIS period ({ts_min.date()}..{ts_max.date()}) predates "
          f"all 3 Prosol snapshots -> held at the earliest (Dec-2025) capacity value throughout.")
    merged = pvgis_df.merge(cap_ts[["district_id", "timestamp", "capacity_mw"]],
                             on=["district_id", "timestamp"], how="inner")
    merged["pv_production_mw_reference"] = (
        merged["pv_output_estimate_w"] / 1000.0 * merged["capacity_mw"]  # per-kWc curve x real capacity_mw
    )
    merged["production_source"] = PVGIS_SOURCE_LABEL
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(merged):,} rows). Update configs/config.yaml "
          f"production_target.mode to PHYSICAL_REFERENCE manually to switch the pipeline over.")


if __name__ == "__main__":
    main()
