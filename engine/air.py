"""Live air quality for the CPCB station network.

Primary source  : CPCB via data.gov.in (hourly, ~40 Delhi monitors).
Fallback source : Open-Meteo CAMS, queried at the same station coordinates, used
                  automatically when data.gov.in is unreachable (it frequently
                  refuses connections). Both return the same ``Reading`` shape,
                  and the active source is always reported, never hidden.
"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from .net import LiveDataError, get_json

IST = timezone(timedelta(hours=5, minutes=30))
CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_STATE = "Delhi"
CPCB_ENDPOINT = f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"
OPENMETEO_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"
# Free, read-only public-data key; override with the DATA_GOV_IN_API_KEY env var.
DEFAULT_CPCB_KEY = "579b464db66ec23bdd000001406ee0ceeec944a97f7a3368a647c393"


@dataclass
class Reading:
    station: str
    lat: float
    lng: float
    observed_at: datetime
    pollutants: dict = field(default_factory=dict)
    source: str = ""

    @property
    def pm25(self) -> Optional[float]:
        return self.pollutants.get("PM2.5")


def _num(raw) -> Optional[float]:
    text = str(raw if raw is not None else "").strip()
    if not text or text.upper() in {"NA", "N/A", "-", "NONE", "NULL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _stamp(raw) -> Optional[datetime]:
    try:
        return datetime.strptime(str(raw).strip(), "%d-%m-%Y %H:%M:%S").replace(tzinfo=IST)
    except ValueError:
        return None


def fetch_cpcb(api_key: Optional[str] = None) -> list[Reading]:
    key = api_key or os.environ.get("DATA_GOV_IN_API_KEY") or DEFAULT_CPCB_KEY
    records, offset = [], 0
    while True:
        query = urllib.parse.urlencode({"api-key": key, "format": "json", "limit": 100,
                                        "offset": offset, "filters[state]": CPCB_STATE})
        page = get_json(f"{CPCB_ENDPOINT}?{query}", label="CPCB (data.gov.in)")
        batch = page.get("records") or []
        records.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(page.get("total", 0) or 0) or len(records) >= 5000:
            break

    stations: dict[str, dict] = {}
    for row in records:
        name = (row.get("station") or "").strip()
        if not name:
            continue
        entry = stations.setdefault(name, {
            "lat": _num(row.get("latitude")), "lng": _num(row.get("longitude")),
            "at": _stamp(row.get("last_update")), "pollutants": {}})
        value = _num(row.get("avg_value"))
        pollutant = (row.get("pollutant_id") or "").strip().upper()
        if value is not None and pollutant:
            entry["pollutants"][pollutant] = value

    readings = [Reading(n, e["lat"], e["lng"], e["at"], e["pollutants"], "CPCB / data.gov.in")
                for n, e in stations.items()
                if e["lat"] is not None and e["lng"] is not None and e["at"] and e["pollutants"]]
    if not readings:
        raise LiveDataError("CPCB: no usable station records returned.")
    return readings


# Open-Meteo CAMS variable -> CPCB pollutant id. CO is converted from ug/m3 to
# mg/m3 to match CPCB's unit; DUST is CAMS-only (no CPCB equivalent).
_OM_VARS = {"pm2_5": ("PM2.5", 1.0), "pm10": ("PM10", 1.0), "nitrogen_dioxide": ("NO2", 1.0),
            "sulphur_dioxide": ("SO2", 1.0), "carbon_monoxide": ("CO", 0.001),
            "ozone": ("OZONE", 1.0), "dust": ("DUST", 1.0)}


def fetch_openmeteo(stations: Sequence[tuple[str, float, float]]) -> list[Reading]:
    """Current pollutants at each ``(name, lat, lng)`` from Open-Meteo CAMS."""
    now = datetime.now(timezone.utc)
    target = now.strftime("%Y-%m-%dT%H:00")
    out: list[Reading] = []
    for i in range(0, len(stations), 25):
        chunk = stations[i:i + 25]
        query = urllib.parse.urlencode({
            "latitude": ",".join(f"{s[1]:.4f}" for s in chunk),
            "longitude": ",".join(f"{s[2]:.4f}" for s in chunk),
            "hourly": ",".join(_OM_VARS), "past_days": 1, "forecast_days": 1, "timezone": "UTC"})
        data = get_json(f"{OPENMETEO_AIR}?{query}", label="Open-Meteo air quality", timeout=25)
        series = data if isinstance(data, list) else [data]
        for (name, lat, lng), d in zip(chunk, series):
            hourly = d.get("hourly") or {}
            times = hourly.get("time") or []
            pollutants = {}
            for var, (pid, scale) in _OM_VARS.items():
                values = hourly.get(var) or []
                val = values[times.index(target)] if target in times and times.index(target) < len(values) else None
                if val is None:  # latest non-null value at or before now
                    for t, v in zip(times, values):
                        if t <= target and v is not None:
                            val = v
                if val is not None:
                    pollutants[pid] = round(float(val) * scale, 3 if pid == "CO" else 1)
            if "PM2.5" in pollutants:
                out.append(Reading(name, lat, lng, now, pollutants, "Open-Meteo CAMS (live)"))
    if not out:
        raise LiveDataError("Open-Meteo returned no usable PM2.5.")
    return out


def fetch_air(known_stations: Sequence[tuple[str, float, float]]) -> tuple[list[Reading], str]:
    """CPCB first; fall back to Open-Meteo at ``known_stations``. Returns (readings, source)."""
    try:
        readings = fetch_cpcb()
        return readings, "CPCB / data.gov.in"
    except LiveDataError as exc:
        print(f"[air] CPCB unavailable ({exc}); falling back to Open-Meteo", flush=True)
        return fetch_openmeteo(known_stations), "Open-Meteo CAMS (live)"
