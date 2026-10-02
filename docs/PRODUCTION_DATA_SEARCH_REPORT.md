# PRODUCTION_DATA_SEARCH_REPORT.md

Systematic search conducted 2026-09-23, extending the search already
documented in `docs/REAL_PRODUCTION_DATA_SOURCES.md`. This pass adds
academic/research-repository sources (Zenodo, OpenAIRE-indexed, institutional
repositories) and re-verifies direct network reachability of PVGIS/NASA POWER/
data.gov.tn from this build environment.

**Network note for this session:** direct `curl` to `re.jrc.ec.europa.eu`,
`power.larc.nasa.gov`, and `data.gov.tn` all return HTTP 403 from this
sandbox's own egress proxy (not the sites' own protection — a different
failure mode than PVGIS's WAF rejection documented previously). No Apify (or
equivalent relay) tool was available in this session to work around it, so
this pass is search-only; the direct-fetch results from the prior session
(PVGIS reachable via HTTP but WAF-rejected; Open-Meteo reachable via a relay
at daily resolution) still stand as the most complete connectivity findings.

## A. District/governorate/national measured PV production

| Source | Finding | Verdict |
|---|---|---|
| STEG | No PV-specific production API/dataset found (re-confirmed) | Rejected — not found |
| ANME/PROSOL | No public production API/dataset found | Rejected — not found |
| data.gov.tn | Portal exists (CKAN); blocked from direct query this session; prior session's search found no PV dataset | Not found (needs a direct CKAN query from an unrestricted machine) |
| Zenodo (rooftop PV production, keyword search) | Several real rooftop PV production datasets exist — **none for Tunisia**: La Réunion (France, 4kW plant, 2021-2022), an unnamed "Industrial PV Rooftop daily dataset" (location unspecified, per-minute), an orientation-diversity dataset (location unspecified, hourly, ~1 year) | Rejected — real data, wrong country |
| GENeSYS-MOD Tunisia dataset (Zenodo 16736000/14931688) | Hourly input data for an energy-system optimization model (OSeMOSYS-based) covering Tunisia — but these are **model input assumptions/capacity factors**, not measured generation; provenance of the solar profile itself traces back to a reanalysis/simulation product, not meters | Rejected as a production target — it's itself a modelled input, one level removed from measurement |
| "Suitability Map for SPVPs in Tunisia" (Zenodo 20657185) | Real GIS layers (solar radiation raster, slope, land use) for siting analysis — spatial resource data, not a production time series | Not usable as a production target; potentially usable as an additional real solar-resource layer for the physical reference model (GeoTIFF, would need raster extraction — not implemented this session) |

## B. Real ground-truth solar-radiation studies specific to Tunisia (not production, but scientifically load-bearing)

These don't give a production target, but they matter: they are real, peer-reviewed, Tunisia-specific measurements that validate using satellite/model-based irradiance (PVGIS, Open-Meteo) as a credible physical reference here — which is the next best thing to measured production.

- **Maatallah et al. 2012**, *"Experimental study of solar energy potential in the gulf of Tunis, Tunisia"* — real 10-minute solar radiation measurements at Borj-Cedria (CRTEn), 2008-2010 (notably: the same location as ENSTAB). A conventional clear-sky model matched measurements within 4.1% under clear skies, degrading to ~14.3% under cloud — a real, local error bound for physics-based clear-sky modelling in this exact region.
- **PMC7673135** (NIM Tunisia study) — real GHI measured by the Institut National de la Météorologie at three stations (Bizerte, Nabeul, Djerba) compared directly against **PVGIS** satellite-derived GHI: error percentage **under 10%** across all three stations and seasons.

Together these two studies are the strongest available scientific justification
for treating a future PVGIS-based (or similarly satellite-irradiance-derived)
physical reference dataset as credible for Tunisia specifically — they are
independent, real, local validations of the exact method this repo's
`PVGISProvider` is designed to use, even though PVGIS itself could not be
executed from this build sandbox this session.

## C. Verdict

No real, measured, timestamped PV production dataset for Tunisia (district,
governorate, or national) was found — this is now the third search pass
(original + two continuation sessions) reaching the same conclusion across
STEG, ANME, data.gov.tn, IRENA, JODI, Kaggle, and now Zenodo/academic
literature specifically. **This is documented as a confirmed external
blocker, not a search gap.**

## D. What would change this

1. A direct data-sharing request to STEG or ANME (the only realistic path to
   district-level measured self-consumption/injection data).
2. Running `src/ingestion/pvgis_provider.py` from an unrestricted network —
   would upgrade the physical reference from `SYNTHETIC_CLEARSKY` to a real
   satellite-irradiance-derived estimate, independently validated for
   Tunisia per §B above (still labelled `PVGIS_PHYSICAL_ESTIMATE`, never
   "measured").
3. Extracting the Zenodo Tunisia solar-radiation raster (§A) as an additional
   cross-check on the physical model's radiation assumptions, if pursued.
