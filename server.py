#!/usr/bin/env python3
"""Vayumitre server -- static pages + live API routes.

    GET /api/forecasts -> per-station 1/3/6/12/24h XGBoost forecast + local maxima
    GET /api/plan      -> live spray plan (re-tiered from current readings,
                          routes + truck sizing recomputed each cycle)

Self-contained: uses only this folder's engine/ (HTTP helper, CPCB + Open-Meteo
air reader, XGBoost forecaster + models). No dependency on any other project.
Run:  python3 server.py   ->  http://localhost:8124/
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from engine.air import fetch_air  # noqa: E402
from engine.forecast_live import cpcb_points, fetch_history, run_models  # noqa: E402
from engine.causes import attribute, mean_mix  # noqa: E402
from engine.zoning import TRUCKS_PER_ZONE, apply_fixed_trucks, build_zones, p_truck_need, split_p, zone_regions  # noqa: E402

import numpy as np  # noqa: E402
from scipy.interpolate import RBFInterpolator  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "engine", "models")
CLUSTERS_V2_PATH = os.path.join(DATA_DIR, "clusters_v2.json")
PORT = int(os.environ.get("PORT") or (sys.argv[1] if len(sys.argv) > 1 else 8124))
CACHE_TTL_S = 1800  # forecasts don't need 5-min refresh -- 30 min is plenty

# Truck-sizing params (mirrors truck_sizing.py)
SPRAY_DWELL_MIN = 8.0
URGENCY_WINDOW_H = 12

# Same quarterly PM2.5 thresholds validated against the real historical CSV
# (preprocessing/cleaned/clean_hourly.csv), mirrored from admin.html.
THRESHOLDS_PM25 = {1: 174.2, 2: 170.0, 3: 95.5, 4: 190.0}
RISING_FRAC = 0.8


def current_quarter() -> int:
    return (datetime.now().month - 1) // 3 + 1


def active_limit() -> float:
    return THRESHOLDS_PM25[current_quarter()]


def rising_limit() -> float:
    return active_limit() * RISING_FRAC


def find_local_maxima(coords: np.ndarray, values: np.ndarray, names: list[str],
                       rising: float, active: float, grid_n: int = 140):
    """RBF-interpolate the point values into a continuous surface, then flag
    grid cells that are the peak of their neighbourhood and exceed a trigger.
    Same method validated in the offline clustering prototype, now run
    against real live station coordinates (~40 points instead of 6)."""
    lat_pad = (coords[:, 0].max() - coords[:, 0].min()) * 0.12 or 0.05
    lon_pad = (coords[:, 1].max() - coords[:, 1].min()) * 0.12 or 0.05
    lat_lin = np.linspace(coords[:, 0].min() - lat_pad, coords[:, 0].max() + lat_pad, grid_n)
    lon_lin = np.linspace(coords[:, 1].min() - lon_pad, coords[:, 1].max() + lon_pad, grid_n)
    LAT, LON = np.meshgrid(lat_lin, lon_lin, indexing="ij")
    grid_pts = np.column_stack([LAT.ravel(), LON.ravel()])

    rbf = RBFInterpolator(coords, values, kernel="thin_plate_spline", smoothing=1.0)
    surface = rbf(grid_pts).reshape(LAT.shape)

    neighborhood_max = maximum_filter(surface, size=7, mode="nearest")
    is_peak = surface >= neighborhood_max

    peaks = []
    for i, j in zip(*np.where(is_peak)):
        val = float(surface[i, j])
        if val < rising:
            continue
        # attach the nearest real station as the human-readable label
        d = np.hypot(coords[:, 0] - LAT[i, j], coords[:, 1] - LON[i, j])
        nearest = names[int(np.argmin(d))]
        peaks.append({
            "lat": round(float(LAT[i, j]), 4), "lng": round(float(LON[i, j]), 4),
            "value": round(val, 1),
            "state": "ACTIVE" if val >= active else "PRE_POSITION",
            "nearest_station": nearest,
        })
    peaks.sort(key=lambda p: -p["value"])
    # de-dup near-identical plateau peaks within ~1.5km of each other
    kept = []
    for p in peaks:
        if all((p["lat"] - k["lat"]) ** 2 + (p["lng"] - k["lng"]) ** 2 > 1.5e-4 for k in kept):
            kept.append(p)
    return kept[:15]


# --------------------------------------------------------------------------- #
# Live plan: re-tier every station from the CURRENT reading, rebuild spray     #
# routes and truck counts each cycle. Cluster *membership* stays from the      #
# quarterly clusters_v2.json (by design), but which stations are ACTIVE /      #
# sprayed / how many trucks are needed is recomputed live -- so a station      #
# whose reading has since dropped below the limit stops being shown as an      #
# active hotspot instead of staying red from a stale snapshot.                 #
# --------------------------------------------------------------------------- #
import math  # noqa: E402
import subprocess  # noqa: E402

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{}?overview=full&geometries=geojson"
_route_geom_cache: dict = {}   # tuple(ordered station names) -> geometry dict


def _tier(pm: float, al: float, rl: float) -> str:
    if pm >= al:
        return "ACTIVE"
    if pm >= rl:
        return "PRE_POSITION"
    return "NORMAL"


def _haversine_km(a, b):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _nn_order(pts):
    remaining = sorted(pts, key=lambda p: -p["pm25"])
    route = [remaining.pop(0)]
    while remaining:
        last = route[-1]
        remaining.sort(key=lambda p: _haversine_km((last["lat"], last["lng"]), (p["lat"], p["lng"])))
        route.append(remaining.pop(0))
    return route


def _osrm_route(ordered):
    key = tuple(p["station"] for p in ordered)
    if key in _route_geom_cache:
        return _route_geom_cache[key]
    coords = ";".join(f"{p['lng']},{p['lat']}" for p in ordered)
    result = None
    for attempt in range(3):
        try:
            out = subprocess.run(
                ["curl", "-s", "--max-time", "30", "-A", "ecoshield-live/1.0",
                 OSRM_URL.format(coords)], capture_output=True, text=True)
            d = json.loads(out.stdout)
            if d.get("code") == "Ok" and d.get("routes"):
                rt = d["routes"][0]
                result = {"geometry": [[c[1], c[0]] for c in rt["geometry"]["coordinates"]],
                          "distance_km": round(rt["distance"] / 1000, 2),
                          "duration_min": round(rt["duration"] / 60, 1),
                          "straight_line_fallback": False}
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.6 * (attempt + 1))
    if result is None:
        result = {"geometry": [[p["lat"], p["lng"]] for p in ordered],
                  "distance_km": None, "duration_min": None, "straight_line_fallback": True}
    _route_geom_cache[key] = result
    return result


def _hours_to_breach(curve_values, active_limit, now_pm):
    if now_pm >= active_limit:
        return 0.0
    for i, v in enumerate(curve_values or []):
        if v is not None and v >= active_limit:
            return float(i + 1)
    return None


IST = timezone(timedelta(hours=5, minutes=30))


def _cause_for(names, payload):
    """Likely cause for a station or zone: the mean pollutant mix of ``names``."""
    pn = payload.get("pollutants_now", {})
    mixes = [pn[n] for n in names if n in pn]
    if not mixes:
        return None
    return attribute(mean_mix(mixes), datetime.now(IST).hour)


def build_plan(payload: dict) -> dict:
    """Live cluster plan: membership from the quarterly file, everything else
    (tier, spray decision, route, truck count) recomputed from live data."""
    cv2 = payload["zones"]          # clusters formed from THIS refresh's AQI readings
    coord = {s["station"]: (s["lat"], s["lng"]) for s in cv2["stations"]}
    now = payload.get("readings_now", {})
    fc = {s["name"]: s for s in payload["stations"]}
    al, rl = payload["active_limit"], payload["rising_limit"]

    clusters_out, total = [], 0
    p_idx = 0
    for c in cv2["clusters"]:
        label = f"G{c['id']}"

        members = []
        for name in c["stations"]:
            if name not in coord or name not in now:
                continue  # not reporting live -- can't tier it, leave out of live plan
            pm = now[name]
            members.append({"station": name, "lat": coord[name][0], "lng": coord[name][1],
                            "pm25": pm, "tier": _tier(pm, al, rl)})

        # dynamic P ("red") zones: hotspots that stand out inside this cluster,
        # <=2 CPCB stations each, served standalone and removed from the G route
        members, p_groups = split_p(members, al)
        for grp in p_groups:
            p_idx += 1
            ordered_p = _nn_order(grp)
            p_need, p_passes = p_truck_need(grp, rl)
            ge = {"passes": p_passes, "cause": _cause_for([m["station"] for m in grp], payload), "label": f"P{p_idx}", "kind": "priority", "parent": label,
                  "n_members": len(grp), "skipped_normal": [], "checkpoints": ordered_p,
                  "geometry": None, "distance_km": None, "duration_min": None,
                  "trucks_needed": p_need, "dedicated_trucks_for": [],
                  "note": f"red zone inside {label} -- standalone truck"}
            if len(ordered_p) == 2:
                g2 = _osrm_route(ordered_p)
                ge.update(geometry=g2["geometry"], distance_km=g2["distance_km"],
                          duration_min=g2["duration_min"],
                          straight_line_fallback=g2["straight_line_fallback"])
            total += p_need
            clusters_out.append(ge)

        spray = [m for m in members if m["tier"] != "NORMAL"]
        skipped = [m for m in members if m["tier"] == "NORMAL"]

        entry = {"cause": _cause_for([m["station"] for m in members], payload),
                 "label": label, "kind": c["kind"], "n_members": len(members),
                 "skipped_normal": skipped, "checkpoints": [], "geometry": None,
                 "distance_km": None, "duration_min": None, "trucks_needed": 0,
                 "dedicated_trucks_for": []}

        if not spray:
            entry["note"] = "all reporting stations NORMAL right now -- not dispatched"
            clusters_out.append(entry)
            continue

        if len(spray) == 1:
            entry["checkpoints"] = spray
            entry["trucks_needed"] = 1
            entry["note"] = "1 spray point -- truck holds position"
            total += 1
            clusters_out.append(entry)
            continue

        ordered = _nn_order(spray)
        geo = _osrm_route(ordered)
        entry.update(checkpoints=ordered, geometry=geo["geometry"],
                     distance_km=geo["distance_km"], duration_min=geo["duration_min"],
                     straight_line_fallback=geo["straight_line_fallback"])

        # truck sizing: base truck drives the route; any spray point it reaches
        # after that point's forecast breach deadline gets its own truck.
        hop_km = [_haversine_km((ordered[i-1]["lat"], ordered[i-1]["lng"]),
                                 (ordered[i]["lat"], ordered[i]["lng"]))
                  for i in range(1, len(ordered))]
        total_km = sum(hop_km) or 1.0
        drive_min = geo["duration_min"] or (total_km / 25 * 60)
        arr = [0.0]
        for km in hop_km:
            arr.append(arr[-1] + SPRAY_DWELL_MIN + (km / total_km) * drive_min)

        late = []
        for cp, a_min in zip(ordered, arr):
            f = fc.get(cp["station"])
            dl = _hours_to_breach(f["curve_v"] if f else None, al, cp["pm25"])
            if dl is None or dl > URGENCY_WINDOW_H:
                continue
            if a_min > dl * 60:
                late.append({"station": cp["station"], "deadline_h": dl,
                             "route_arrival_min": round(a_min, 1)})
        entry["trucks_needed"] = 1 + len(late)
        entry["dedicated_trucks_for"] = late
        total += entry["trucks_needed"]
        clusters_out.append(entry)

    fleet = apply_fixed_trucks(clusters_out, int(os.environ.get("TRUCKS_PER_ZONE") or TRUCKS_PER_ZONE))
    return {"generated_at": payload["generated_at"], "active_limit": al, "rising_limit": rl,
            "clusters": clusters_out, "total_trucks_needed": total, "fleet": fleet}


# --------------------------------------------------------------------------- #
# Live air-quality fallback: when CPCB (data.gov.in) is unreachable, fetch     #
# current PM2.5 for the same station coordinates from Open-Meteo's CAMS        #
# air-quality API (keyless, different host). Gives full 42-station coverage    #
# live, versus IQAir's ~6 NCR cities.                                          #
# --------------------------------------------------------------------------- #
ZONES_PATH = os.path.join(DATA_DIR, "zones_live.json")


def monthly_zones(stations_now, forecasts, series, al, rl) -> dict:
    """Fixed-for-the-month G zones. Reuses this month's saved zones if present,
    else forms them from typical station levels (mean of the recent hourly
    history, rescaled to each monitor's own reading) and saves them."""
    month = datetime.now().strftime("%Y-%m")
    try:
        with open(ZONES_PATH) as f:
            saved = json.load(f)
        if saved.get("month") == month and saved.get("clusters"):
            return saved
    except (OSError, ValueError):
        pass
    now_key = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%dT%H:00")
    fc = {f.name: f for f in forecasts}
    zs = []
    for st in stations_now:
        typical = st["pm25"]
        f = fc.get(st["station"])
        if f is not None and f.node_id in series:
            sr = series[f.node_id]
            obs = [v for t, v in zip(sr["time"], sr["pm2_5"]) if v is not None and t <= now_key]
            if obs and f.grid_now and f.grid_now > 1.0:
                typical = sum(obs) / len(obs) * (f.pm25_now / f.grid_now)
        zs.append({"station": st["station"], "lat": st["lat"], "lng": st["lng"],
                   "pm25": round(float(typical), 1)})
    zones, _k, work = build_zones(zs, al, rl)
    coord_by = {z_["station"]: (z_["lat"], z_["lng"]) for z_ in zs}
    with open(os.path.join(DATA_DIR, "delhi_boundary.json")) as f_:
        boundary = json.load(f_)["boundary"]
    out = {"month": month, "active_limit": al, "rising_limit": rl, "total_workload": work,
           "priority_clusters": 0, "geo_clusters": len(zones), "clusters": zones, "stations": zs,
           "regions": zone_regions(zones, coord_by, boundary) if zones else {}}
    try:
        with open(ZONES_PATH, "w") as f:
            json.dump(out, f)
    except OSError:
        pass
    print(f"[zones] formed {len(zones)} G zones for {month} (typical workload {work})", flush=True)
    return out


def build_payload() -> dict:
    with open(CLUSTERS_V2_PATH) as f:
        known = [(x["station"], x["lat"], x["lng"]) for x in json.load(f)["stations"]]
    readings, source = fetch_air(known)
    print(f"[air] {source}: {len(readings)} stations", flush=True)
    points, anchors = cpcb_points(readings)
    series = fetch_history(points)
    forecasts = run_models(series, MODEL_DIR, anchors=anchors)

    stations = [f.as_dict() for f in forecasts]
    coords = np.array([[f.lat, f.lng] for f in forecasts])
    names = [f.name for f in forecasts]

    now_vals = np.array([f.pm25_now for f in forecasts])
    h24_vals = np.array([f.horizons.get(24, f.pm25_now) for f in forecasts])

    rl, al = rising_limit(), active_limit()
    maxima_now = find_local_maxima(coords, now_vals, names, rl, al)
    maxima_24h = find_local_maxima(coords, h24_vals, names, rl, al)

    # live current reading for EVERY reporting station (not just those that
    # got a forecast) -- lets the plan re-tier stations with no forecast too.
    readings_now = {r.station: r.pm25 for r in readings if r.pm25 is not None}

    # full station list with coords, so the browser can render markers even
    # where it can't reach data.gov.in directly (it proxies through us).
    stations_now = [{"station": r.station, "lat": r.lat, "lng": r.lng, "pm25": r.pm25}
                    for r in readings
                    if r.pm25 is not None and r.lat is not None and r.lng is not None]

    # G zones: formed ONCE per calendar month from each station's typical (recent)
    # AQI level, then fixed for the whole month. Only P red zones change hourly.
    zones_payload = monthly_zones(stations_now, forecasts, series, al, rl)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zones": zones_payload,
        "air_source": source,
        "quarter": current_quarter(),
        "rising_limit": rl, "active_limit": al,
        "stations": stations,
        "readings_now": readings_now,
        "pollutants_now": {r.station: r.pollutants for r in readings if r.pm25 is not None},
        "causes": {r.station: attribute(r.pollutants, datetime.now(IST).hour)
                   for r in readings if r.pm25 is not None},
        "stations_now": stations_now,
        "maxima_now": maxima_now,
        "maxima_24h": maxima_24h,
    }


_cache = {"data": None, "plan": None, "at": 0.0, "error": None, "building": False}


def refresh_cache():
    _cache["building"] = True
    try:
        payload = build_payload()
        _cache["data"] = payload
        try:
            _cache["plan"] = build_plan(payload)
        except Exception as e:  # noqa: BLE001 - a plan failure shouldn't drop forecasts
            _cache["plan"] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        _cache["error"] = None
        _cache["at"] = time.time()
        pt = _cache["plan"] or {}
        print(f"[forecast] refreshed: {len(payload['stations'])} stations, "
              f"{len(payload['maxima_24h'])} 24h maxima, "
              f"plan: {pt.get('total_trucks_needed', '?')} trucks", flush=True)
    except Exception as e:
        _cache["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _cache["building"] = False


def background_refresher():
    while True:
        refresh_cache()
        # On success wait the full TTL; on failure (network flaky) retry soon
        # instead of sitting on a stale/empty cache for 30 min.
        ok = _cache["data"] is not None and _cache["error"] is None
        time.sleep(CACHE_TTL_S if ok else 30)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/api/forecasts"):
            self._json(_cache["data"])
            return
        if self.path.startswith("/api/zones"):
            self._json((_cache["data"] or {}).get("zones"))
            return
        if self.path.startswith("/api/plan"):
            self._json(_cache["plan"])
            return
        super().do_GET()

    def _json(self, obj):
        body = json.dumps(obj if obj else
                          {"error": _cache["error"] or "still building, retry shortly"}).encode()
        self.send_response(200 if obj else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + (fmt % args) + "\n")


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    threading.Thread(target=background_refresher, daemon=True).start()
    handler = functools.partial(Handler, directory=SCRIPT_DIR)
    with ThreadingServer(("", PORT), handler) as httpd:
        print(f"serving {SCRIPT_DIR} on :{PORT}  (forecasts building in background)", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
