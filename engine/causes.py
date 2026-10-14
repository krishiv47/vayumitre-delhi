"""Rule-based likely-cause attribution from the pollutant mix (no AI, no model).

Each source has a fingerprint over the pollutants CPCB publishes (PM2.5, PM10,
NO2, SO2, CO, OZONE, NH3) plus the time of day. A source's score is a weighted sum
of clamped 0..1 evidence terms; sources scoring >= MIN_SHOW are returned ranked.
This ranks the DOMINANT LIKELY source from patterns -- it is not proof; real
source apportionment needs chemical speciation data.

Units: ug/m3 everywhere except CO, which is mg/m3 (CPCB convention).
"""
from __future__ import annotations

MIN_SHOW = 0.25
PM25_MIN, PM10_MIN = 60.0, 100.0   # below both -> nothing to attribute

LABELS = {
    "dust": "Road / construction dust",
    "biomass": "Biomass / waste burning & combustion",
    "traffic": "Vehicle traffic",
    "industry": "Industry / thermal power (sulphur fuels)",
    "photochem": "Photochemical (sunlight) ozone & secondary haze",
    "agri": "Agriculture / waste & drainage (ammonia)",
    "regional": "Regional secondary aerosol (transported haze)",
}
SHORT = {"dust": "Dust", "biomass": "Burning", "traffic": "Traffic", "industry": "Industry",
         "photochem": "Photochemical", "agri": "Agri / NH₃", "regional": "Regional haze"}
# Spraying (lignosulfonate) suppresses dust and resuspension, not combustion particles.
SPRAY_HELPS = {"dust": "yes", "traffic": "partly (resuspended road dust)", "biomass": "little",
               "industry": "no", "photochem": "no", "agri": "no", "regional": "no"}


def _c(x: float) -> float:
    return max(0.0, min(1.0, x))


def attribute(p: dict, hour: int) -> dict:
    """p: pollutant id -> value (PM2.5, PM10, NO2, SO2, CO, OZONE, NH3, DUST). hour: IST 0-23."""
    pm25, pm10 = p.get("PM2.5"), p.get("PM10")
    no2, so2, co = p.get("NO2"), p.get("SO2"), p.get("CO")
    o3, nh3, dust = p.get("OZONE"), p.get("NH3"), p.get("DUST")
    ratio = (pm25 / pm10) if pm25 and pm10 else None
    rush = hour in (7, 8, 9, 10, 17, 18, 19, 20, 21)
    night = hour >= 18 or hour <= 8
    day = 11 <= hour <= 18

    # No attribution when particulates aren't actually elevated.
    elevated = (pm25 is not None and pm25 >= PM25_MIN) or (pm10 is not None and pm10 >= PM10_MIN)
    if not elevated:
        have = [k for k in ("PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3") if p.get(k) is not None]
        return {"causes": [], "pollutants": have, "missing": [k for k in ("PM10", "NO2", "SO2", "CO", "OZONE", "NH3") if p.get(k) is None],
                "note": "particulates within normal range -- no pollution rise to attribute"}

    scores: dict[str, tuple[float, list[str]]] = {}

    def add(key, terms):
        s, ev = 0.0, []
        for w, val, msg in terms:
            if val is not None:
                s += w * val
                if val >= 0.4 and msg:
                    ev.append(msg)
        scores[key] = (s, ev)

    add("dust", [
        (0.55, None if ratio is None else _c((0.55 - ratio) / 0.25), f"PM2.5/PM10 = {ratio:.2f} (coarse-dominated)" if ratio else ""),
        (0.25, None if pm10 is None else _c((pm10 - 100) / 200), f"PM10 {pm10:.0f} µg/m³" if pm10 else ""),
        (0.20, None if dust is None else _c(dust / 100), f"dust {dust:.0f} µg/m³" if dust else ""),
    ])
    add("biomass", [
        (0.45, None if ratio is None else _c((ratio - 0.5) / 0.25), f"PM2.5/PM10 = {ratio:.2f} (fine-dominated)" if ratio else ""),
        (0.35, None if co is None else _c((co - 1.0) / 2.0), f"CO {co:.1f} mg/m³" if co is not None else ""),
        (0.20, _c((pm25 or 0) / 150) if night else 0.0, "evening/night build-up" if night else ""),
    ])
    add("traffic", [
        (0.55, None if no2 is None else _c((no2 - 30) / 60), f"NO₂ {no2:.0f} µg/m³" if no2 is not None else ""),
        (0.25, None if co is None else _c((co - 0.8) / 1.5), f"CO {co:.1f} mg/m³" if co is not None else ""),
        (0.20, (_c(((no2 or 0) - 20) / 40) if rush else 0.0), "rush hour with elevated NO₂"),
    ])
    add("industry", [
        (0.70, None if so2 is None else _c((so2 - 15) / 40), f"SO₂ {so2:.0f} µg/m³" if so2 is not None else ""),
        (0.30, None if no2 is None else _c((no2 - 40) / 60), f"NO₂ {no2:.0f} µg/m³" if no2 is not None else ""),
    ])
    add("photochem", [
        (0.60, None if o3 is None else (_c((o3 - 50) / 70) if day else 0.0), f"ozone {o3:.0f} µg/m³ in daylight" if o3 is not None else ""),
        (0.40, None if no2 is None else (_c((40 - no2) / 40) if day else 0.0), "low NO₂ (ozone not titrated)"),
    ])
    add("agri", [
        (0.80, None if nh3 is None else _c((nh3 - 30) / 100), f"NH₃ {nh3:.0f} µg/m³" if nh3 is not None else ""),
        (0.20, None if ratio is None else _c((ratio - 0.5) / 0.3), "fine particles"),
    ])
    low_local = None
    if ratio is not None:
        low_local = _c((ratio - 0.55) / 0.2) * (1 - _c((no2 or 0) / 60)) * (1 - _c((co or 0) / 1.5))
    if low_local is not None:
        add("regional", [(1.0, low_local * (0.5 + 0.5 * _c((pm25 or 0) / 100)),
                          "fine PM with low local NO₂/CO (not a local source)")])
    ranked = sorted(((k, s, ev) for k, (s, ev) in scores.items() if s >= MIN_SHOW), key=lambda t: -t[1])[:3]
    have = [k for k in ("PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3") if p.get(k) is not None]
    return {
        "causes": [{"key": k, "label": LABELS[k], "short": SHORT[k], "strength": round(100 * min(s, 1.0)),
                    "evidence": ev, "spray_helps": SPRAY_HELPS[k]} for k, s, ev in ranked],
        "pollutants": have,
        "missing": [k for k in ("PM10", "NO2", "SO2", "CO", "OZONE", "NH3") if p.get(k) is None],
    }


def mean_mix(mixes: list[dict]) -> dict:
    """Average each pollutant over the stations that report it (zone-level mix)."""
    out: dict[str, float] = {}
    keys = {k for m in mixes for k in m}
    for k in keys:
        vals = [m[k] for m in mixes if m.get(k) is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
