# External data requests for MoveOD validation

Drafted 2026-09-18. Each request supports one comparison in the external-validation study
(`analysis/validate_external.py`, `analysis/link_loads.py`). Fill in the sender block before sending.

---

## 1. Chattanooga–Hamilton County / North Georgia TPO (CHCRPA)

To: rpa@chattanooga.gov (Regional Planning Agency, 423-643-5900)
Subject: Data request — home-based-work trip tables from the regional travel demand model

Hello,

I am a researcher at Vanderbilt University working on MoveOD, an open-source pipeline that
synthesizes building-level commute origin–destination data for any U.S. county from Census
LODES and ACS tables (arXiv 2510.18858). We are validating the Hamilton County output against
independent, locally calibrated sources, and the TPO's regional travel demand model is the most
direct comparison available.

Could you share, for the current model base year:

1. The home-based-work (HBW) trip table at TAZ level, daily and, if available, by time period.
2. The TAZ shapefile (or geodatabase) matching that table.
3. If shareable, trip records from the regional household travel survey with origin/destination
   TAZ, purpose, departure time and duration.

We would use these only for aggregate comparison (common part of commuters, trip-length
frequency distribution, per-TAZ residuals) and will acknowledge the TPO as the data source in the
resulting publication. A data-sharing agreement is fine if required.

Thank you,
[name, title, department, email, phone]

---

## 2. TDOT Traffic Monitoring (Long Range Planning Division)

To: [TDOT traffic monitoring contact — see https://www.tn.gov/tdot/long-range-planning-home]
Subject: Data request — hourly volumes, Hamilton County continuous count stations, March 2025

Hello,

I am a researcher at Vanderbilt University validating a synthetic commute-trip dataset for
Hamilton County against observed traffic. The Traffic History portal exposes AADT; for this study
we need the underlying hourly counts.

Could you provide, for all continuous count stations in Hamilton County:

1. Hourly volumes by direction for March 2025 (at minimum 10 and 17 March 2025), and for the
   weekdays of a comparable month in 2024 if 2025 is not yet finalized.
2. Station metadata: station id, coordinates, route, direction, functional class.

Hourly summaries are described as available on request on the Traffic Monitoring Program page.
Any standard export format (CSV, TMG-format files) is fine.

Thank you,
[name, title, department, email, phone]

---

## 3. Dewey Data — Advan Neighborhood Patterns (via Vanderbilt subscription)

To: [Vanderbilt library / Dewey account administrator]
Subject: Access request — Advan Neighborhood Patterns, Tennessee, March 2024 and March 2025

Hello,

For a validation study of synthetic commute OD data (MoveOD, arXiv 2510.18858) I need the Advan
Neighborhood Patterns product through Vanderbilt's Dewey subscription:

1. Neighborhood Patterns for Tennessee, months 2025-03 and 2024-03 (all census block groups),
   specifically the columns `work_behavior_device_home_areas`, `device_home_areas`,
   `stops_by_each_hour`, `raw_device_counts`, `raw_stop_counts`.
2. The current data dictionary, so the "work behavior" definition and the noise / minimum-count
   rules can be cited precisely.

If Weekly Patterns Plus is more accessible than Neighborhood Patterns, the same request applies
to it for Hamilton and Davidson counties.

Thank you,
[name, title, department, email, phone]
