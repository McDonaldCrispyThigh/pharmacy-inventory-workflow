"""Historical class name for the upstream candidate-collection workflow.

``NETSAgentWorkflow`` is retained for compatibility with the existing command
entry point; it does not consume NETS data.  The module covers:

  1. Search  — loop over ZIP codes via Google Places Text Search
  2. Enrich  — fetch place details per unique result
  3. Label   — call a constrained LLM step for task-specific candidate labels
  4. Save    — write a timestamped CSV to /data

This module is upstream of the authenticated downstream rebuild.  The frozen
analysis input does not retain enough query, response, model, and prompt
provenance to establish end-to-end reproduction; see ``analysis/README.md``.

Usage (programmatic):
    from agent_workflow import NETSAgentWorkflow
    workflow = NETSAgentWorkflow(config, maps_agent, openai_client,
                                 city_name="Minneapolis",
                                 zip_codes=["55401", ...])
    workflow.run(output_dir="../data", task_name="coffee")
"""

import json
import os
import re
import sys
from datetime import datetime

import pandas as pd

_ZIP_RE = re.compile(r",\s+[A-Z]{2}\s+(\d{5})")

# Ensure project root is on sys.path so skills/ is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

FINAL_COLUMNS = [
    "Company", "Calculated_NAICS", "NAICS_Version", "Target_NAICS",
    "Is_Target_Match", "Confidence",
    "Match_Reasoning", "Business_Status", "Review_Count", "Has_Reviews",
    "Latitude", "Longitude", "Street_Address", "City", "State", "Zip_Code",
    "Operating_Hours", "Hard_Attributes", "Price_Level", "Business_Website",
    "Last_Review_Date",
]

# Retired columns -- deliberately NOT emitted.
#
# Employees_Estimated and Year_Established were free-text guesses from the
# language model with no external corroboration. A model asked for a store's
# headcount or opening year answers from priors about the brand, so
# "Walgreens, founded 1901" gets attached to a branch that opened in 2015.
# Treating either as observed data would put unverifiable numbers into the
# analysis under the same column conventions as measured fields.
#
# Establishment year is recoverable, but only with provenance: Board licensing
# records give an initial-licence date, and archival evidence supports an
# "active by" bound. If reinstated, do it as an explicit set --
# Location_Opening_Year / Opening_Year_Type (verified|proxy|active_by|unknown) /
# Opening_Year_Source / Opening_Year_Evidence -- never as a bare integer.
RETIRED_COLUMNS = ["Employees_Estimated", "Year_Established"]


class NETSAgentWorkflow:
    """Upstream pipeline: retrieve, enrich, label, and save candidates."""

    def __init__(self, config: dict, maps_agent, openai_client,
                 city_name: str, zip_codes: list[str]):
        """
        Parameters
        ----------
        config       : category config dict (search_term, target_naics, definition, …)
        maps_agent   : GoogleMapsAgent instance
        openai_client: openai.OpenAI instance
        city_name    : human-readable city name, e.g. "Minneapolis"
        zip_codes    : list of ZIP codes to scan
        """
        self.config = config
        self.maps = maps_agent
        self.ai = openai_client
        self.city = city_name
        self.zip_codes = zip_codes
        self.today_str = datetime.now().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, output_dir: str, task_name: str) -> str:
        """
        Execute the full pipeline and save results.

        Returns
        -------
        str  — absolute path of the saved CSV file.
        """
        print(f">>> [NETSAgentWorkflow] Starting task: {task_name}")

        unique_places = self._search_all_zips()
        processed_data = self._process_all_places(unique_places)
        output_path = self._save(processed_data, output_dir, task_name)
        return output_path

    # ------------------------------------------------------------------
    # Step 1: Search
    # ------------------------------------------------------------------

    # 2x2 grid offsets in degrees (~1.5 km at lat 45).
    # Each offset pair is (delta_lat, delta_lon).
    _GRID_OFFSETS = [(-0.007, -0.010), (-0.007, 0.010),
                     ( 0.007, -0.010), ( 0.007, 0.010)]

    @staticmethod
    def _zip_centroid(zip_code: str):
        """Return (lat, lon) for a ZIP code using pgeocode, or None."""
        try:
            import pgeocode
            nomi = pgeocode.Nominatim("us")
            row = nomi.query_postal_code(zip_code)
            if row is not None and row.latitude == row.latitude:  # NaN check
                return float(row.latitude), float(row.longitude)
        except Exception:
            pass
        return None

    def _search_all_zips(self) -> list[dict]:
        """
        Search Google Maps across all ZIP codes; return de-duplicated places.

        Strategy (bypasses the 60-result-per-query hard cap):
          1. Resolve ZIP centroid via pgeocode.
          2. Tile a 2x2 grid of Nearby Search calls (radius 1500 m each) around
             the centroid -- each cell surfaces different nearby places.
          3. Fall back to Text Search variants if centroid lookup fails.
        De-duplication is by place_id throughout.
        """
        raw: dict[str, dict] = {}
        keyword: str = self.config.get("search_term", "Pharmacy")
        fallback_variants: list[str] = self.config.get(
            "search_variants", [keyword]
        )
        print(f">>> Scanning {len(self.zip_codes)} ZIP codes (2x2 grid per ZIP) …")

        for zip_code in self.zip_codes:
            before = len(raw)
            print(f"\n--- ZIP {zip_code} ---")

            centroid = self._zip_centroid(zip_code)
            if centroid:
                lat, lon = centroid
                for dlat, dlon in self._GRID_OFFSETS:
                    results = self.maps.search_nearby(
                        lat + dlat, lon + dlon,
                        keyword=keyword, radius=1500,
                    )
                    new = sum(
                        1 for p in results
                        if p["place_id"] not in raw
                        and not raw.update({p["place_id"]: {**p, "_source_zip": zip_code}})
                    )
                    print(f"    [grid ({dlat:+.3f},{dlon:+.3f})] "
                          f"{len(results)} results, {new} new unique")
            else:
                # pgeocode unavailable -- fall back to text search variants
                print(f"    [centroid lookup failed, using text search]")
                for term in fallback_variants:
                    query = f"{term} in {self.city} {zip_code}"
                    results = self.maps.search_places(query)
                    new = 0
                    for place in results:
                        pid = place["place_id"]
                        if pid not in raw:
                            place["_source_zip"] = zip_code
                            raw[pid] = place
                            new += 1
                    print(f"    [{term}] {len(results)} results, {new} new unique")

            print(f"    ZIP total new: {len(raw) - before}")

        unique = list(raw.values())
        print(f"\n>>> TOTAL UNIQUE PLACES: {len(unique)}")
        return unique

    # ------------------------------------------------------------------
    # Step 2 + 3: Enrich & Classify
    # ------------------------------------------------------------------

    def _process_all_places(self, places: list[dict]) -> list[dict]:
        """Fetch details and classify each place via GPT."""
        results = []
        total = len(places)
        print(">>> Starting AI analysis …")

        for i, place in enumerate(places):
            print(f"[{i + 1}/{total}] {place.get('name', '?')}")
            try:
                details = self.maps.get_place_details(place["place_id"])
            except Exception:
                continue

            row = self._classify_place(place, details)
            if row:
                results.append(row)

        return results

    def _classify_place(self, place: dict, details: dict) -> dict | None:
        """Build a row dict for one place using GPT for NAICS classification."""
        name = details.get("name", place.get("name", "Unknown"))

        # --- Extract facts ---
        geometry = details.get("geometry", {}).get("location", {})
        lat = geometry.get("lat")
        lng = geometry.get("lng")
        addr = details.get("formatted_address", "")
        price_level = details.get("price_level", "N/A")

        attrs = []
        for attr in ("serves_breakfast", "serves_lunch", "serves_dinner",
                     "serves_beer", "serves_wine"):
            if details.get(attr):
                attrs.append(attr.replace("serves_", "").capitalize())
        attr_str = ", ".join(attrs) if attrs else "None"

        hours_list = details.get("opening_hours", {}).get("weekday_text", [])
        operating_hours = "; ".join(hours_list) if hours_list else "Unknown"

        reviews = details.get("reviews", [])
        review_count = len(reviews)
        has_reviews = bool(reviews)
        last_review_date = "N/A"
        reviews_text = "NO REVIEWS. Judge based on Name, Hours, and Attributes only."

        if reviews:
            timestamps = [r.get("time", 0) for r in reviews]
            last_review_date = datetime.fromtimestamp(max(timestamps)).strftime("%Y-%m-%d")
            snippets = [f"- {r['text'][:200]}…" for r in reviews[:3] if r.get("text")]
            reviews_text = "\n".join(snippets)

        # --- GPT prompt ---
        default_logic = (
            "1. Hours: Opens 6-8 AM suggests Coffee/Bakery. Opens 4 PM suggests Bar.\n"
            "2. Food attributes: Breakfast served suggests Coffee. Dinner+Beer without Breakfast suggests Bar.\n"
            "3. Reviews confirm the category."
        )
        logic = self.config.get("classification_logic", default_logic)

        system_msg = (
            f"You are a data researcher. Today is {self.today_str}.\n"
            f"TASK: Assign the correct NAICS code for '{name}'.\n"
            f"Target category: {self.config['search_term']} (NAICS {self.config['target_naics']}).\n"
            f"Definition: {self.config['definition']}\n\n"
            f"LOGIC:\n{logic}"
        )
        user_msg = (
            f"Name: {name}\n"
            f"FACTS: Hours: {operating_hours} | Attrs: {attr_str} | Price: {price_level}\n"
            f"REVIEWS:\n{reviews_text}\n\n"
            'Return JSON only:\n'
            '{"Calculated_NAICS":"6-digit",'
            '"Status":"Active/Inactive","Confidence":"High/Low","Reasoning":"brief"}'
        )
        # Employees / Year_Established are no longer requested: the model has no
        # evidence for either and would answer from brand priors. See
        # RETIRED_COLUMNS.

        try:
            resp = self.ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
            )
            raw_content = resp.choices[0].message.content
            ai = json.loads(raw_content.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            print(f"    [AI Error] {e}")
            return None

        website = details.get("website")

        return {
            "Company":             name,
            "Calculated_NAICS":    ai.get("Calculated_NAICS"),
            "NAICS_Version":       self.config.get("naics_version", "2017"),
            "Target_NAICS":        self.config["target_naics"],
            "Is_Target_Match":     ai.get("Calculated_NAICS") == self.config["target_naics"],
            "Confidence":          ai.get("Confidence"),
            "Match_Reasoning":     ai.get("Reasoning"),
            "Business_Status":     ai.get("Status"),
            "Review_Count":        review_count,
            "Has_Reviews":         "Yes" if has_reviews else "No",
            "Latitude":            lat,
            "Longitude":           lng,
            "Street_Address":      addr,
            "City":                self.city,
            "State":               "MN",
            "Zip_Code":            (_ZIP_RE.search(addr).group(1)
                                    if _ZIP_RE.search(addr) else place.get("_source_zip")),
            "Operating_Hours":     operating_hours,
            "Hard_Attributes":     attr_str,
            "Price_Level":         price_level,
            "Business_Website":    website,
            "Last_Review_Date":    last_review_date,
        }

    # ------------------------------------------------------------------
    # Step 4: Save
    # ------------------------------------------------------------------

    def _save(self, data: list[dict], output_dir: str, task_name: str) -> str:
        if not data:
            print(">>> No data to save.")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.city}_{task_name}_{timestamp}.csv"
        path = os.path.join(output_dir, filename)

        df = pd.DataFrame(data).reindex(columns=FINAL_COLUMNS)
        df.to_csv(path, index=False)

        print(f">>> Saved {len(df)} records -> {filename}")
        return os.path.abspath(path)
