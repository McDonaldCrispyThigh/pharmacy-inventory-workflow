"""Fetch ACS Table B08201 (Household Size by Vehicles Available) for all
Hennepin (053) and Ramsey (123) tracts in Minnesota (state 27), 2023 5-year.

Outputs: data/acs_tracts_vehicles_2023.csv
  GEOID, total_hh, hh_no_vehicle, carless_rate
"""
import csv
import json
import urllib.request
from pathlib import Path

DATA = Path("data")
OUT = DATA / "acs_tracts_vehicles_2023.csv"

URL = "https://api.census.gov/data/2023/acs/acs5"
VARS = ["B08201_001E", "B08201_002E"]  # total / no-vehicle households

rows = []
for county in ("053", "123"):
    qs = (
        "?get=NAME," + ",".join(VARS)
        + f"&for=tract:*&in=state:27%20county:{county}"
    )
    req = urllib.request.Request(URL + qs, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    header = data[0]
    for record in data[1:]:
        rec = dict(zip(header, record))
        geoid = rec["state"] + rec["county"] + rec["tract"]
        total = int(rec["B08201_001E"]) if rec["B08201_001E"] not in (None, "null", "") else 0
        no_veh = int(rec["B08201_002E"]) if rec["B08201_002E"] not in (None, "null", "") else 0
        carless = (no_veh / total) if total > 0 else 0.0
        rows.append({"GEOID": geoid, "total_hh": total,
                     "hh_no_vehicle": no_veh,
                     "carless_rate": round(carless, 4)})

rows.sort(key=lambda r: r["GEOID"])
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# Quick stats
carless = [r["carless_rate"] for r in rows if r["total_hh"] > 0]
print(f"tracts: {len(rows)}")
print(f"min carless: {min(carless):.3f}")
print(f"max carless: {max(carless):.3f}")
print(f"median carless: {sorted(carless)[len(carless)//2]:.3f}")
print(f"wrote {OUT}")
