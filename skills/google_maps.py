"""Google Places adapters for the frozen v1 collector and transient v2 Details use.

The helper returns query-matched platform records.  Its output is neither a
complete establishment census nor evidence of operation, public eligibility,
or service use.  Canonical downstream rebuilding uses a frozen local snapshot
and makes no live request through this module.
"""

import json
import hashlib
import re
from datetime import datetime, timezone

import googlemaps
import os
import time
from dotenv import load_dotenv

class PlacesNewDetailsError(RuntimeError):
    """Safe transport/response failure that never includes response or key content."""


def _decode_json_object(raw: bytes):
    """Return decoded JSON without retaining hostile bytes in an exception chain."""
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value


class GooglePlacesNewDetailsAgent:
    """Transient Places API (New) Details adapter with an injected HTTP transport.

    The transport must implement ``get(url, headers, timeout) -> (status, bytes)``.
    Requiring injection prevents this offline-verified component from making a live
    request before the repository's written-permission gate has been satisfied.
    """

    __slots__ = ("_api_key", "_transport", "timeout")
    endpoint = "https://places.googleapis.com/v1/places"
    field_mask = ",".join((
        "id", "displayName", "formattedAddress", "nationalPhoneNumber",
        "internationalPhoneNumber", "location", "websiteUri",
        "regularOpeningHours", "types", "rating", "userRatingCount",
        "businessStatus", "reviews", "addressComponents",
    ))

    def __init__(self, api_key: str, transport, *, timeout: float = 30.0):
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("Places API key is required")
        if not hasattr(transport, "get"):
            raise TypeError("Places New transport must provide get()")
        if type(timeout) not in {int, float} or timeout <= 0:
            raise ValueError("positive timeout is required")
        self._api_key = api_key
        self._transport = transport
        self.timeout = float(timeout)

    def __repr__(self):
        return "GooglePlacesNewDetailsAgent(transport=<injected>)"

    def get_place_details(self, place_id: str) -> dict:
        if not isinstance(place_id, str) or re.fullmatch(
                r"[A-Za-z0-9_-]{10,256}", place_id) is None:
            raise ValueError("invalid Place ID")
        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": self.field_mask,
            "Accept": "application/json",
        }
        try:
            status, body = self._transport.get(
                f"{self.endpoint}/{place_id}", headers, self.timeout,
            )
        except Exception:
            raise PlacesNewDetailsError("Places New transport failed") from None
        if type(status) is not int or not isinstance(body, bytes):
            raise PlacesNewDetailsError("invalid Places New transport response")
        if status != 200:
            raise PlacesNewDetailsError(f"Places New HTTP {status}")
        if len(body) > 2_000_000:
            raise PlacesNewDetailsError("Places New response exceeded size limit")
        payload = _decode_json_object(body)
        if payload is None:
            raise PlacesNewDetailsError("invalid Places New JSON response")
        if type(payload) is not dict or payload.get("id") != place_id:
            raise PlacesNewDetailsError("Places New response identity mismatch")
        return payload


class GooglePlacesNewProvider(GooglePlacesNewDetailsAgent):
    """Transient Text Search (New) plus Details (New) provider for v2.

    Search responses request Place IDs only.  Full details and the returned review
    subset are fetched later and remain in memory under ``v2_workflow``'s retention
    boundary.  The injected transport must provide both ``post`` and ``get``.
    """

    search_endpoint = "https://places.googleapis.com/v1/places:searchText"
    search_field_mask = "places.id,nextPageToken"
    # Places API (New) may return a nextPageToken after the historically
    # documented three-page/60-result boundary (the provider documents that
    # boundary as subject to change). Follow explicit, non-cyclic tokens while
    # retaining a defensive ceiling and the existing fail-closed page_cap state.
    max_pages_per_query = 10
    query_definitions = {
        "pharmacy": (
            ("pharmacy_places_new_v2", "pharmacy"),
            ("drug_store_places_new_v2", "drug store"),
            ("drugstore_places_new_v2", "drugstore"),
        ),
    }

    def __init__(self, api_key: str, transport, *, timeout: float = 30.0,
                 clock=None):
        super().__init__(api_key, transport, timeout=timeout)
        if not hasattr(transport, "post"):
            raise TypeError("Places New transport must provide post()")
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return "GooglePlacesNewProvider(transport=<injected>)"

    def _search_page(self, text_query: str,
                     page_token: str | None) -> tuple[dict, dict]:
        body = {"textQuery": text_query, "pageSize": 20}
        if page_token is not None:
            body["pageToken"] = page_token
        raw_request = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": self.search_field_mask,
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        try:
            status, raw_response = self._transport.post(
                self.search_endpoint, headers, raw_request, self.timeout,
            )
        except Exception:
            raise PlacesNewDetailsError("Places New search transport failed") from None
        if type(status) is not int or not isinstance(raw_response, bytes):
            raise PlacesNewDetailsError("invalid Places New search response")
        if status != 200:
            raise PlacesNewDetailsError(f"Places New search HTTP {status}")
        if len(raw_response) > 1_000_000:
            raise PlacesNewDetailsError("Places New search response exceeded size limit")
        payload = _decode_json_object(raw_response)
        if payload is None:
            raise PlacesNewDetailsError("invalid Places New search JSON")
        if (type(payload) is not dict
                or not set(payload) <= {"places", "nextPageToken"}):
            raise PlacesNewDetailsError("invalid Places New search payload")
        retrieved = self._clock()
        if not isinstance(retrieved, datetime) or retrieved.tzinfo is None:
            raise PlacesNewDetailsError("Places New search clock is invalid")
        metadata = {
            "retrieved_at": retrieved.astimezone(timezone.utc).isoformat()
            .replace("+00:00", "Z"),
            "response_sha256": hashlib.sha256(raw_response).hexdigest(),
        }
        return payload, metadata

    def list_candidates(self, *, task: str, city: str,
                        zips: tuple[str, ...], limit: int | None) -> dict:
        if task not in self.query_definitions:
            raise ValueError("unsupported v2 Places task")
        if (not isinstance(city, str) or not city.strip() or len(city) > 200
                or any(ord(char) < 32 or ord(char) == 127 for char in city)):
            raise ValueError("invalid Places city")
        if (not isinstance(zips, (tuple, list)) or not zips
                or any(not isinstance(item, str)
                       or re.fullmatch(r"\d{5}", item) is None for item in zips)):
            raise ValueError("invalid Places ZIP frame")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("invalid Places candidate limit")

        results: list[dict] = []
        occurrences: list[dict] = []
        query_ledger: list[dict] = []
        any_capped = False
        seen_place_ids: set[str] = set()
        for source_zip in zips:
            for query_id, term in self.query_definitions[task]:
                page_token: str | None = None
                seen_tokens: set[str] = set()
                pages: list[dict] = []
                added_before = len(results)
                terminal_state = "exhausted"
                for page_index in range(self.max_pages_per_query):
                    payload, page_metadata = self._search_page(
                        f"{term} in {city.strip()} {source_zip}", page_token,
                    )
                    places = payload.get("places", [])
                    if not isinstance(places, list):
                            raise PlacesNewDetailsError(
                                "invalid Places New search place collection"
                        )
                    page_record = {
                        **page_metadata,
                        "page_number": page_index + 1,
                        "returned_place_id_count": len(places),
                        "next_page_token_present": (
                            payload.get("nextPageToken") is not None),
                    }
                    pages.append(page_record)
                    limit_hit = False
                    for place in places:
                        if type(place) is not dict or set(place) != {"id"}:
                            raise PlacesNewDetailsError(
                                "invalid Places New search place row"
                            )
                        place_id = place.get("id")
                        if (not isinstance(place_id, str)
                                or re.fullmatch(r"[A-Za-z0-9_-]{10,256}",
                                                place_id) is None):
                            raise PlacesNewDetailsError(
                                "invalid Places New search Place ID"
                            )
                        occurrences.append({
                            "place_id": place_id,
                            "query_definition_id": query_id,
                            "source_zip": source_zip,
                            "page_number": page_index + 1,
                        })
                        if (place_id not in seen_place_ids
                                and (limit is None or len(results) < limit)):
                            results.append({"place_id": place_id,
                                            "source_zip": source_zip,
                                            "query_definition_id": query_id})
                            seen_place_ids.add(place_id)
                            if limit is not None and len(results) >= limit:
                                limit_hit = True
                    if limit_hit:
                        terminal_state = "limit_reached"
                        query_ledger.append({
                            "query_definition_id": query_id,
                            "source_zip": source_zip,
                            "terminal_state": terminal_state,
                            "unique_candidates_added": len(results) - added_before,
                            "pages": pages,
                        })
                        return {
                            "status": "limit_reached",
                            "candidates": results,
                            "occurrences": occurrences,
                            "queries": query_ledger,
                        }
                    next_token = payload.get("nextPageToken")
                    if next_token is None:
                        terminal_state = "exhausted"
                        break
                    if (not isinstance(next_token, str) or not next_token
                            or len(next_token) > 4096
                            or any(ord(char) < 32 or ord(char) == 127
                                   for char in next_token)
                            or next_token in seen_tokens):
                        raise PlacesNewDetailsError(
                            "invalid Places New pagination token"
                        )
                    if page_index + 1 == self.max_pages_per_query:
                        terminal_state = "page_cap"
                        any_capped = True
                        break
                    seen_tokens.add(next_token)
                    page_token = next_token
                query_ledger.append({
                    "query_definition_id": query_id,
                    "source_zip": source_zip,
                    "terminal_state": terminal_state,
                    "unique_candidates_added": len(results) - added_before,
                    "pages": pages,
                })
        return {
            "status": "capped" if any_capped else "complete",
            "candidates": results,
            "occurrences": occurrences,
            "queries": query_ledger,
        }

class GoogleMapsAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            print("Warning: Missing GOOGLE_MAPS_API_KEY in .env")
            self.client = None
        else:
            self.client = googlemaps.Client(key=api_key)

    def search_places(self, query):
        """Run legacy Places Text Search and follow available page tokens."""
        if not self.client:
            return []
        
        all_results = []
        next_token = None
        
        # The loop has a defensive upper bound; the endpoint normally stops
        # after its documented result cap by omitting the next-page token.
        for page_num in range(5): 
            try:
                response = None
                
                # --- RETRY LOGIC FOR TOKEN ---
                # If we have a token, we must wait until it is valid.
                if next_token:
                    attempts = 0
                    while attempts < 5:
                        print(f"    ... Fetching Page {page_num + 1} (Attempt {attempts+1})...")
                        time.sleep(2 + attempts) # Wait 2s, then 3s, then 4s...
                        try:
                            response = self.client.places(query=query, page_token=next_token)
                            if response['status'] == 'OK':
                                break # Success!
                            else:
                                print(f"    [Google Status: {response['status']}] Retrying...")
                        except Exception as e:
                            print(f"    [Error] {e}. Retrying...")
                        
                        attempts += 1
                else:
                    # First page (No token needed)
                    response = self.client.places(query=query)
                
                # --- PROCESS RESULTS ---
                if response and response.get('status') == 'OK':
                    results = response['results']
                    all_results.extend(results)
                    
                    # Check for next page
                    next_token = response.get('next_page_token')
                    if not next_token:
                        print("    ... No more pages for this specific query.")
                        break
                else:
                    print("    ... Failed to get valid response.")
                    break
            
            except Exception as e:
                print(f"Search Critical Error: {e}")
                break
                
        return all_results

    def search_nearby(self, lat: float, lon: float,
                      keyword: str = "Pharmacy", radius: int = 1500) -> list:
        """
        Location-biased Text Search centered on (lat, lon).

        Uses places() (Text Search) with location+radius so Google ranks results
        by proximity to the given point. Different center points return different
        top-60 results -- this is the mechanism for exceeding the per-query cap.
        """
        if not self.client:
            return []

        all_results = []
        next_token = None
        _places = getattr(self.client, "places")  # type: ignore[attr-defined]

        for page_num in range(5):
            try:
                response = None
                if next_token:
                    attempts = 0
                    while attempts < 5:
                        print(f"    ... Grid page {page_num + 1} (attempt {attempts + 1})...")
                        time.sleep(2 + attempts)
                        try:
                            response = _places(
                                query=keyword,
                                location=(lat, lon),
                                radius=radius,
                                page_token=next_token,
                            )
                            if response["status"] == "OK":
                                break
                        except Exception as e:
                            print(f"    [Error] {e}. Retrying...")
                        attempts += 1
                else:
                    response = _places(
                        query=keyword,
                        location=(lat, lon),
                        radius=radius,
                    )

                if response and response.get("status") in ("OK", "ZERO_RESULTS"):
                    all_results.extend(response.get("results", []))
                    next_token = response.get("next_page_token")
                    if not next_token:
                        break
                else:
                    break
            except Exception as e:
                print(f"Grid search error: {e}")
                break

        return all_results

    def get_place_details(self, place_id):
        """Historical v1 legacy Details call; v2 never calls this method."""
        if not self.client:
            return {}
        try:
            # Request newest-first ordering for the limited review sample.
            # Last_Review_Date therefore describes the newest returned review,
            # not a complete audit of all reviews or business activity.
            result = self.client.place(
                place_id=place_id,
                fields=[
                    'name', 'formatted_address', 'formatted_phone_number',
                    'type', 'business_status', 'price_level',
                    'geometry', 'website', 'opening_hours',
                    'reviews',
                    'serves_beer', 'serves_wine', 'serves_breakfast',
                    'serves_lunch', 'serves_dinner'
                ],
                reviews_sort="newest",
            )
            return result.get('result', {})
        except Exception as e:
            return {}
