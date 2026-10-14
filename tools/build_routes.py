#!/usr/bin/env python3
"""
Turns each cluster (from fetch_live_clusters.py's clusters_v2.json) into
an ordered checkpoint route a truck can actually drive and spray along.

Checkpoints = the real CPCB station coordinates already in the cluster --
not invented points, so "spray here" always lines up with a place that has
a live PM2.5 reading to confirm the before/after effect.

Ordering: greedy nearest-neighbor by straight-line distance, starting at
the cluster's worst (highest PM2.5) station -- "go fix the worst spot
first, then whichever checkpoint is closest next" per the plan. Straight-
line, not OSRM travel time, decides the ORDER (fast, no rate-limit risk
running it station-by-station); OSRM is used once per cluster afterward to
fetch the actual road-following polyline through that fixed order.

Single-station clusters (most PRIORITY clusters) get no route -- the truck
just holds position there.

Run:
    python3 build_routes.py
"""
from __future__ import annotations
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data"))  # work inside data/

import json
import math
import subprocess
import time
from datetime import datetime

OSRM = "https://router.project-osrm.org/route/v1/driving/{}?overview=full&geometries=geojson"

# Same quarterly PM2.5 thresholds validated against the real historical CSV,
# mirrored from server.py / fetch_live_clusters.py.
THRESHOLDS_PM25 = {1: 174.2, 2: 170.0, 3: 95.5, 4: 190.0}
RISING_FRAC = 0.8


def current_quarter():
    return (datetime.now().month - 1) // 3 + 1


def active_limit():
    return THRESHOLDS_PM25[current_quarter()]


def rising_limit():
    return active_limit() * RISING_FRAC


def tier_of(pm25):
    if pm25 >= active_limit():
        return "ACTIVE"
    if pm25 >= rising_limit():
        return "PRE_POSITION"
    return "NORMAL"


def haversine_km(a, b):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def nearest_neighbor_order(points):
    """points: list of dicts with lat/lng/pm25/station. Returns them reordered,
    starting at the worst PM2.5 and always hopping to the nearest unvisited."""
    remaining = points.copy()
    remaining.sort(key=lambda p: -p["pm25"])
    route = [remaining.pop(0)]
    while remaining:
        last = route[-1]
        remaining.sort(key=lambda p: haversine_km((last["lat"], last["lng"]), (p["lat"], p["lng"])))
        route.append(remaining.pop(0))
    return route


def fetch_osrm_route(ordered_points):
    """One multi-waypoint OSRM request for the whole fixed order -- real
    road-following geometry, not a straight line between checkpoints."""
    coords = ";".join(f"{p['lng']},{p['lat']}" for p in ordered_points)
    url = OSRM.format(coords)
    for attempt in range(4):
        try:
            out = subprocess.run(
                ["curl", "-s", "--max-time", "30", "-A", "ecoshield-build/1.0", url],
                capture_output=True, text=True,
            )
            d = json.loads(out.stdout)
            if d.get("code") == "Ok" and d.get("routes"):
                route = d["routes"][0]
                return {
                    "geometry": [[c[1], c[0]] for c in route["geometry"]["coordinates"]],
                    "distance_km": round(route["distance"] / 1000, 2),
                    "duration_min": round(route["duration"] / 60, 1),
                }
        except Exception as e:  # noqa: BLE001
            print(f"    OSRM attempt {attempt+1} failed: {e}")
        time.sleep(1.0 * (attempt + 1))
    return None


def main():
    with open("clusters_v2.json") as f:
        cv2 = json.load(f)
    station_by_name = {s["station"]: s for s in cv2["stations"]}

    routes = {}
    # labels aren't stored in clusters_v2.json (assigned client-side in JS) --
    # recompute the same scheme here so route keys match the UI.
    p_idx = 0
    for c in cv2["clusters"]:
        if c["kind"] == "priority":
            p_idx += 1
            c["_label"] = f"P{p_idx}"
        else:
            c["_label"] = f"G{c['id']}"

    al, rl = active_limit(), rising_limit()
    print(f"Active limit {al:.1f}, pre-position limit {rl:.1f} -- only ACTIVE/PRE_POSITION "
          f"checkpoints get sprayed; NORMAL ones are pass-through only.\n")

    for c in cv2["clusters"]:
        label = c["_label"]
        pts = [station_by_name[name] for name in c["stations"] if name in station_by_name]
        pts = [{"station": p["station"], "lat": p["lat"], "lng": p["lng"], "pm25": p["pm25"],
                 "tier": tier_of(p["pm25"]), "sprinkle": tier_of(p["pm25"]) != "NORMAL"} for p in pts]

        # Route waypoints = spray points ONLY. Pass-through (NORMAL) stations
        # in the cluster stay visible on the map via the live tier-colored
        # dots (rendered separately, from every live station) -- they just
        # aren't stops on this truck's path, since there's nothing to spray
        # there. This is the actual fix for "only spray the high points."
        spray_pts = [p for p in pts if p["sprinkle"]]
        skipped = [p for p in pts if not p["sprinkle"]]

        if len(pts) < 2:
            routes[label] = {"kind": c["kind"], "checkpoints": pts, "geometry": None,
                              "note": "single checkpoint -- truck holds position, no route needed"}
            print(f"{label}: 1 checkpoint ({pts[0]['tier']}), no route needed")
            continue
        if not spray_pts:
            routes[label] = {"kind": c["kind"], "checkpoints": [], "skipped_normal": pts,
                              "geometry": None,
                              "note": "all stations NORMAL right now -- no spray stops, truck not dispatched"}
            print(f"{label}: {len(pts)} stations, all NORMAL -- no spray needed, skipping route")
            continue
        if len(spray_pts) == 1:
            routes[label] = {"kind": c["kind"], "checkpoints": spray_pts, "skipped_normal": skipped,
                              "geometry": None,
                              "note": "1 spray point -- truck holds position there, no route needed"}
            print(f"{label}: 1 spray point ({spray_pts[0]['station']}), {len(skipped)} NORMAL skipped -- no route needed")
            continue

        ordered = nearest_neighbor_order(spray_pts)
        print(f"{label}: {len(ordered)} spray checkpoints (skipping {len(skipped)} NORMAL stations), order = " +
              " -> ".join(f"{p['station'].split(',')[0]}({p['pm25']:.0f})" for p in ordered))

        osrm = fetch_osrm_route(ordered)
        if osrm:
            print(f"    OSRM: {osrm['distance_km']} km, {osrm['duration_min']} min")
        else:
            print(f"    OSRM failed after retries -- falling back to straight lines between checkpoints")
            osrm = {"geometry": [[p["lat"], p["lng"]] for p in ordered],
                    "distance_km": None, "duration_min": None, "straight_line_fallback": True}

        routes[label] = {
            "kind": c["kind"],
            "checkpoints": ordered,
            "skipped_normal": skipped,
            "geometry": osrm["geometry"],
            "distance_km": osrm.get("distance_km"),
            "duration_min": osrm.get("duration_min"),
            "straight_line_fallback": osrm.get("straight_line_fallback", False),
        }
        time.sleep(0.5)  # be polite to the free demo server

    with open("routes.json", "w") as f:
        json.dump({"generated_at": cv2["generated_at"], "routes": routes}, f, indent=2)
    print(f"\nWrote routes.json -- {len(routes)} cluster routes.")


if __name__ == "__main__":
    main()
