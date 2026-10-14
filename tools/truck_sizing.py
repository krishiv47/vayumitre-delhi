#!/usr/bin/env python3
"""
Time-series-driven truck count: how many trucks does each cluster need
*right now*, given real forecasted arrival at each checkpoint.

The core idea: a single truck follows the cluster's planned patrol route
(build_routes.py) checkpoint by checkpoint. Using each checkpoint's real
24h XGBoost forecast curve (already live on /api/forecasts), we know the
deadline -- how many hours until that station's PM2.5 crosses the active
limit. If the truck's real arrival time at a checkpoint (driving the
actual OSRM route + a spray dwell) is later than that checkpoint's
deadline, one truck can't make it in time -- that checkpoint needs its
own dedicated truck sent directly, instead of waiting in the patrol queue.

    trucks_needed(cluster) = 1 base patrol truck
                              + 1 per checkpoint the base truck arrives at
                                too late (after its predicted breach)

This is a real constraint-satisfaction read on real data, not a guess:
every number here (current PM2.5, 24h curve, drive time) comes from the
live pipeline already running in this project.

Run:
    python3 truck_sizing.py
"""
from __future__ import annotations
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data"))  # work inside data/

import json
import math
import urllib.request
from datetime import datetime, timezone

SPRAY_DWELL_MIN = 8.0        # minutes a truck spends spraying at each stop
URGENCY_WINDOW_H = 12         # only checkpoints breaching within this window count


def haversine_km(a, b):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def hours_to_breach(curve_values, active_limit, already_now):
    """First hour (1-indexed into the forward curve) the forecast reaches
    the active limit. 0 if it's already there. None if it never does
    within the curve's window -- not urgent this cycle."""
    if already_now >= active_limit:
        return 0.0
    for i, v in enumerate(curve_values):
        if v is not None and v >= active_limit:
            return float(i + 1)
    return None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fetch_forecasts():
    with urllib.request.urlopen("http://localhost:8124/api/forecasts", timeout=30) as r:
        return json.loads(r.read().decode())


def size_cluster(label, route, forecast_by_name, active_limit):
    checkpoints = route["checkpoints"]
    if len(checkpoints) < 1:
        return None

    # deadline per checkpoint, from its real 24h forecast curve
    deadlines = {}
    for cp in checkpoints:
        f = forecast_by_name.get(cp["station"])
        if f is None:
            # This station's XGBoost forecast wasn't available this cycle
            # (real per-station Open-Meteo history gaps -- same issue as
            # Wazirpur earlier). Don't silently treat that as "not urgent":
            # fall back to its own live CPCB reading, which needs no
            # forecast at all to know it's already critical.
            deadlines[cp["station"]] = 0.0 if cp["pm25"] >= active_limit else None
            continue
        d = hours_to_breach(f["curve_v"], active_limit, f["now"])
        deadlines[cp["station"]] = d

    # cumulative arrival time for the base truck driving the planned route,
    # in order, at real speed (OSRM total time allocated proportionally to
    # each hop's straight-line share, since we only have one whole-route
    # OSRM call, not per-hop) plus a fixed spray dwell at every prior stop.
    hop_km = []
    for i in range(1, len(checkpoints)):
        hop_km.append(haversine_km(
            (checkpoints[i-1]["lat"], checkpoints[i-1]["lng"]),
            (checkpoints[i]["lat"], checkpoints[i]["lng"])))
    total_straight_km = sum(hop_km) or 1.0
    total_drive_min = route.get("duration_min") or (total_straight_km / 25 * 60)  # 25kph fallback

    arrival_min = [0.0]  # truck is at checkpoint 0 at t=0
    for i, km in enumerate(hop_km):
        hop_min = (km / total_straight_km) * total_drive_min
        arrival_min.append(arrival_min[-1] + SPRAY_DWELL_MIN + hop_min)

    late = []
    on_time = []
    for cp, arr_min in zip(checkpoints, arrival_min):
        dl_h = deadlines[cp["station"]]
        entry = {"station": cp["station"], "pm25": cp["pm25"],
                 "deadline_h": dl_h, "route_arrival_min": round(arr_min, 1)}
        if dl_h is None or dl_h > URGENCY_WINDOW_H:
            continue  # not urgent this cycle -- base truck will get there eventually, fine
        if arr_min > dl_h * 60:
            late.append(entry)
        else:
            on_time.append(entry)

    trucks_needed = (1 if (on_time or late) else 0) + len(late)
    return {
        "label": label, "kind": route["kind"],
        "n_checkpoints": len(checkpoints),
        "trucks_needed": trucks_needed,
        "base_truck_covers": on_time,
        "dedicated_trucks_for": late,
        "route_total_min": round(total_drive_min, 1),
    }


def main():
    cv2 = load_json("clusters_v2.json")
    routes = load_json("routes.json")["routes"]
    fc = fetch_forecasts()
    forecast_by_name = {s["name"]: s for s in fc["stations"]}
    active_limit = fc["active_limit"]

    print(f"Active limit: {active_limit:.1f} µg/m³  |  urgency window: {URGENCY_WINDOW_H}h  "
          f"|  spray dwell: {SPRAY_DWELL_MIN} min/stop\n")

    results = []
    for label, route in routes.items():
        r = size_cluster(label, route, forecast_by_name, active_limit)
        if r is None:
            continue
        results.append(r)
        print(f"{label} ({r['kind']}): {r['n_checkpoints']} checkpoints, "
              f"route ~{r['route_total_min']:.0f} min -> {r['trucks_needed']} truck(s) needed")
        for e in r["base_truck_covers"]:
            print(f"    base truck reaches {e['station']:40s} at +{e['route_arrival_min']:.0f}min, "
                  f"deadline +{e['deadline_h']*60:.0f}min -- ON TIME")
        for e in r["dedicated_trucks_for"]:
            print(f"    ** DEDICATED TRUCK ** {e['station']:40s} deadline +{e['deadline_h']*60:.0f}min, "
                  f"base truck would arrive +{e['route_arrival_min']:.0f}min -- TOO LATE")

    total = sum(r["trucks_needed"] for r in results)
    print(f"\n=== Fleet requirement right now: {total} trucks across {len(results)} clusters with urgent checkpoints ===")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_limit": active_limit, "urgency_window_h": URGENCY_WINDOW_H,
        "spray_dwell_min": SPRAY_DWELL_MIN,
        "clusters": results, "total_trucks_needed": total,
    }
    with open("truck_plan.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote truck_plan.json")


if __name__ == "__main__":
    main()
