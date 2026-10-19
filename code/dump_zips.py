"""Dump the union of target ZIPs + AI-discovered ZIPs to CSV.

Originally written to define the geographic clip for a NETS data request; that
comparison was descoped when NETS proved unobtainable. The ZIP union is still
the canonical study-area definition and is used to scope the Board of Pharmacy
reference set and any future comparative source.
"""
import csv
from pathlib import Path

DATA = Path("data")

def load_zips(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["zip5_ai"] for row in reader if row.get("zip5_ai")}

ai_zips = (
    load_zips(DATA / "audit2_tp_analysis_20260414_172017.csv")
    | load_zips(DATA / "audit2_fp_analysis_20260414_172017.csv")
)

target_zips = {
    "55401","55402","55403","55404","55405","55406","55407","55408","55409","55410",
    "55411","55412","55413","55414","55415","55454","55455",
    "55101","55102","55103","55104","55105","55106","55107","55108","55116","55117",
    "55118","55119","55130","55113","55126",
    "55421","55422","55423","55424","55425","55426","55427","55428","55429","55430",
    "55431","55432","55433","55434","55435","55436","55437","55438","55439",
    "55441","55442","55443","55444","55445","55446","55447","55448","55369",
}

# Drop any non-Twin-Cities ZIPs that may have crept in (5xxxx range only, MN starts with 55)
ai_zips_clean = {z for z in ai_zips if z and z.startswith("55") and len(z) == 5}

union = sorted(target_zips | ai_zips_clean)

print(f"target ZIPs    : {len(target_zips)}")
print(f"AI-discovered  : {len(ai_zips_clean)}")
print(f"target only (zero AI hits): {sorted(target_zips - ai_zips_clean)}")
print(f"AI-only (grid spillover, not in 60 targets): {sorted(ai_zips_clean - target_zips)}")
print(f"union          : {len(union)}")

out = DATA / "twin_cities_zips.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["zip_code"])
    for z in union:
        w.writerow([z])
print(f"wrote {out}")
