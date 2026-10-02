# PRODUCTION_REFERENCE_DATASET_REPORT.md

## Status: BLOCKED - PVGIS unreachable from this environment

Attempted PVGIS retrieval for all 50 districts using the real
Prosol fleet coordinates. Succeeded for 0/50
before stopping at the first failure (per the strict no-partial-fabrication
rule: mixing real PVGIS rows for some districts with any kind of substitute
for others, in the same file, without the substitute being clearly
impossible to confuse with the real rows, was judged too easy to
misinterpret downstream - so this script stops cleanly instead).

**First failure:**
- District: `D01` (TUNIS VILLE)
- Error: `PVGIS request failed after 2 attempts: 400 Client Error: BAD REQUEST for url: https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=36.80278&lon=10.17972&startyear=2020&endyear=2023&pvcalculation=1&peakpower=1.0&loss=14.0&outputformat=json&raddatabase=PVGIS-SARAH3`

This matches the network behaviour already documented in
`docs/REAL_PRODUCTION_DATA_SOURCES.md` and `docs/PRODUCTION_DATA_SEARCH_REPORT.md`:
`re.jrc.ec.europa.eu` is not reachable from this build sandbox (confirmed
again on 2026-10-02, this session).

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
