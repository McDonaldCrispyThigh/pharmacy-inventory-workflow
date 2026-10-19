"""Transient, typed location envelopes for v2.1 downstream analysis.

These objects deliberately expose no persistence serializer.  Exact provider name,
address, and coordinates may be used in-memory only after the live authorization
gate has passed; offline tests use fictional records.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from v2_evidence_models import QueryProvenanceV2, evidence_id


_PLACE_ID = re.compile(r"^[A-Za-z0-9_-]{10,256}$")
_CANDIDATE_ID = re.compile(r"^candidate_[0-9a-f]{64}$")
_ZIP5 = re.compile(r"^\d{5}$")


def _bounded_text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is missing")
    text = " ".join(value.split())
    if not text or len(text) > maximum or any(ord(char) < 32 for char in text):
        raise ValueError(f"invalid {label}")
    return text


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("retrieval time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _coordinate(value: Any, label: str, low: float, high: float) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise ValueError(f"invalid {label}")
    result = float(value)
    if not low <= result <= high:
        raise ValueError(f"invalid {label}")
    return result


@dataclass(frozen=True, slots=True, repr=False)
class CandidateLocationV2:
    """One Place-ID-specific location packet for downstream adapters only."""

    place_id: str
    candidate_id: str
    name: str
    formatted_address: str
    returned_postal_code: str | None
    latitude: float | None
    longitude: float | None
    retrieved_at: datetime
    query_provenance: QueryProvenanceV2

    def __post_init__(self) -> None:
        if not isinstance(self.place_id, str) or _PLACE_ID.fullmatch(self.place_id) is None:
            raise ValueError("invalid downstream Place ID")
        expected = evidence_id("candidate", {"source_place_id": self.place_id})
        if (
            not isinstance(self.candidate_id, str)
            or _CANDIDATE_ID.fullmatch(self.candidate_id) is None
            or self.candidate_id != expected
        ):
            raise ValueError("downstream candidate ID does not match Place ID")
        object.__setattr__(self, "name", _bounded_text(self.name, "candidate name", 500))
        object.__setattr__(
            self,
            "formatted_address",
            _bounded_text(self.formatted_address, "formatted address", 1000),
        )
        if self.returned_postal_code is not None and (
            not isinstance(self.returned_postal_code, str)
            or _ZIP5.fullmatch(self.returned_postal_code) is None
        ):
            raise ValueError("invalid returned postal code")
        latitude = _coordinate(self.latitude, "latitude", -90.0, 90.0)
        longitude = _coordinate(self.longitude, "longitude", -180.0, 180.0)
        if (latitude is None) != (longitude is None):
            raise ValueError("candidate coordinates must be complete or both missing")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "retrieved_at", _utc(self.retrieved_at))
        if not isinstance(self.query_provenance, QueryProvenanceV2):
            raise TypeError("invalid candidate query provenance")

    @property
    def candidate_snapshot_sha256(self) -> str:
        """Authenticate the exact transient location observation without exposing it."""
        payload = {
            "place_id": self.place_id,
            "candidate_id": self.candidate_id,
            "name": self.name,
            "formatted_address": self.formatted_address,
            "returned_postal_code": self.returned_postal_code,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "retrieved_at": self.retrieved_at.isoformat().replace("+00:00", "Z"),
            "query_provenance": {
                "query_definition_id": self.query_provenance.query_definition_id,
                "source_zip": self.query_provenance.source_zip,
                "retrieval_channel": self.query_provenance.retrieval_channel,
            },
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()


def returned_postal_code(details: Mapping[str, Any]) -> str | None:
    """Read ZIP5 only from provider-returned ``addressComponents``.

    The search query ZIP is intentionally unavailable to this helper, so it cannot be
    substituted when provider location evidence is missing.
    """
    if not isinstance(details, Mapping):
        raise ValueError("invalid Places Details payload")
    components = details.get("addressComponents")
    if components is None:
        return None
    if not isinstance(components, list) or len(components) > 100:
        raise ValueError("invalid address components")
    found: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("invalid address component")
        # Proto JSON omits empty repeated fields. Treat an omitted ``types``
        # array as the documented empty list, while still rejecting explicit
        # nulls and malformed values.
        types = component.get("types", [])
        if not isinstance(types, list) or any(not isinstance(item, str) for item in types):
            raise ValueError("invalid address component types")
        if "postal_code" not in types:
            continue
        short = component.get("shortText")
        long = component.get("longText")
        raw = short if isinstance(short, str) and short.strip() else long
        if not isinstance(raw, str):
            raise ValueError("postal-code component lacks text")
        match = re.match(r"^\s*(\d{5})(?:-\d{4})?\s*$", raw)
        if match is None:
            raise ValueError("provider postal code is not ZIP5/ZIP+4")
        found.add(match.group(1))
    if len(found) > 1:
        raise ValueError("provider returned conflicting postal codes")
    return next(iter(found), None)


def candidate_location_from_details(
    details: Mapping[str, Any],
    provenance: QueryProvenanceV2,
    retrieved_at_utc: datetime,
) -> CandidateLocationV2:
    """Build the transient downstream envelope from one authenticated Details row."""
    if not isinstance(details, Mapping):
        raise ValueError("invalid Places Details payload")
    place_id = details.get("id")
    if not isinstance(place_id, str) or _PLACE_ID.fullmatch(place_id) is None:
        raise ValueError("Places Details lacks a valid Place ID")
    display = details.get("displayName")
    if not isinstance(display, Mapping):
        raise ValueError("Places Details lacks display name")
    location = details.get("location")
    if location is None:
        latitude = longitude = None
    elif isinstance(location, Mapping):
        latitude = location.get("latitude")
        longitude = location.get("longitude")
    else:
        raise ValueError("invalid Places Details location")
    return CandidateLocationV2(
        place_id=place_id,
        candidate_id=evidence_id("candidate", {"source_place_id": place_id}),
        name=display.get("text"),
        formatted_address=details.get("formattedAddress"),
        returned_postal_code=returned_postal_code(details),
        latitude=latitude,
        longitude=longitude,
        retrieved_at=retrieved_at_utc,
        query_provenance=provenance,
    )


__all__ = [
    "CandidateLocationV2",
    "candidate_location_from_details",
    "returned_postal_code",
]
