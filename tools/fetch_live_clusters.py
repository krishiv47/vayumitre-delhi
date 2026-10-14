#!/usr/bin/env python3
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data"))  # work inside data/
"""
Tiered + geographic hybrid clustering (real live CPCB data).

Two-stage design, replacing the single-pass GMM(lat,lng,pm25) approach:

  Stage 1 -- pull out ACTIVE-tier ("corner") stations first. Each becomes
  its own dedicated cluster (or merges with other ACTIVE stations sitting
  within MERGE_RADIUS_KM of each other) -- these need standing priority
  coverage regardless of what's geographically near them.

  Stage 2 -- geo-cluster everything else (PRE_POSITION + NORMAL) using
  ONLY lat/lng, not pollutant value. This is the fix for the old approach's
  real flaw: clustering on (lat,lng,pm25) let severity dominate distance,
  so a cluster could scatter across the whole city as long as the PM2.5
  values were similar -- useless for a truck that has to physically drive
  between its assigned stations. Geography-only clustering guarantees each
  cluster is a compact, patrol-able zone.

Tier composition per geo-cluster is reported as a diagnostic: how many
PRE_POSITION vs NORMAL stations actually landed together, since real
geography decides that -- it isn't a forced quota.
"""
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
from sklearn.mixture import GaussianMixture

API_KEY = "579b464db66ec23bdd000001406ee0ceeec944a97f7a3368a647c393"
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
STATE = "Delhi"
K_CHOICES = [2, 3, 5]
MERGE_RADIUS_KM = 10.0   # severe stations within this radius share one priority zone
PRIORITY_SIGMA = 1.0     # priority zone = severe outlier: >= mean + this*std (and >= active)
TARGET_PER_CLUSTER = 7   # aim for ~this many service stations per geo cluster
GEO_K_MIN, GEO_K_MAX = 2, 6

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


def greedy_distance_cluster(stations, radius_km):
    """Union-find style merge: any two stations within radius_km end up in
    the same group (transitively), everything else stays its own group."""
    n = len(stations)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km((stations[i]["lat"], stations[i]["lng"]),
                              (stations[j]["lat"], stations[j]["lng"]))
            if d <= radius_km:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(stations[i])
    return list(groups.values())


def best_k_geo(coords, k_choices=K_CHOICES, seed=0):
    """Geography-only GMM (lat, lng) -- no pollutant value, so distance is
    purely 'how far apart are these places'.

    Deliberately NOT BIC-selected here: BIC optimizes statistical fit, and
    on a fairly evenly-spread station network that means it prefers *fewer*
    large clusters (measured: k=2 -> ~40km diagonal, basically all of
    Delhi split in half -- not a patrollable zone). What the truck routing
    actually needs is compactness, so this always takes the largest of the
    allowed k choices -- more clusters is what buys smaller zones. BIC
    scores for all choices are still returned so you can see the tradeoff.
    """
    results = []
    for k in k_choices:
        if k > len(coords):
            continue
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=8)
        gmm.fit(coords)
        results.append((k, gmm.bic(coords), gmm))
    if not results:
        raise RuntimeError(f"Not enough non-extreme stations ({len(coords)}) for any of k={k_choices}")
    k, bic, model = max(results, key=lambda r: r[0])  # largest allowed k, not lowest BIC
    labels = model.predict(coords)
    return k, bic, labels, [(kk, b) for kk, b, _ in results]


def main():
    import sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from engine.air import fetch_air
    with open("clusters_v2.json") as _f:          # known station coords (for the fallback source)
        known = [(x["station"], x["lat"], x["lng"]) for x in json.load(_f)["stations"]]
    live, src = fetch_air(known)
    reporting = [{"station": r.station, "lat": r.lat, "lng": r.lng, "pollutants": r.pollutants}
                 for r in live if r.pm25 is not None]
    for s in reporting:
        s["pm25"] = s["pollutants"]["PM2.5"]
        s["tier"] = tier_of(s["pm25"])

    q = current_quarter()
    print(f"Q{q}: active limit {active_limit():.1f}, pre-position limit {rising_limit():.1f}  [source: {src}]")
    print(f"{len(reporting)} live stations reporting PM2.5\n")

    # Workload-balanced zones (engine/zoning.py): zone count scales with total
    # spray workload (higher PM2.5 -> more zones), each zone carries ~equal
    # workload (=> ~equal trucks/water), and a zone with 1-2 ACTIVE stations is P.
    from engine.zoning import build_zones
    al, rl = active_limit(), rising_limit()
    result_clusters, k, workload = build_zones(reporting, al, rl)
    by_name = {s["station"]: s for s in reporting}
    for c in result_clusters:
        mem = [by_name[n] for n in c["stations"]]
        c["span_km"] = round(haversine_km((min(s["lat"] for s in mem), min(s["lng"] for s in mem)),
                                          (max(s["lat"] for s in mem), max(s["lng"] for s in mem))), 1)
        tag = "P" if c["kind"] == "priority" else "G"
        print(f"  {tag} zone: {len(mem)} stations, workload {c['workload']} passes, "
              f"{c['tier_counts']['ACTIVE']} active, ~{c['span_km']:.0f} km span")
    n_p = sum(1 for c in result_clusters if c["kind"] == "priority")
    print(f"\ntotal workload {workload} passes -> {k} zones ({n_p} P + {k - n_p} G)")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quarter": q, "month": datetime.now().strftime("%Y-%m"),
        "active_limit": al, "rising_limit": rl,
        "total_workload": workload,
        "priority_clusters": n_p,
        "geo_clusters": k - n_p,
        "clusters": result_clusters,
        "stations": [{"station": s["station"], "lat": s["lat"], "lng": s["lng"],
                       "pm25": s["pm25"], "tier": s["tier"]} for s in reporting],
    }
    with open("clusters_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote clusters_v2.json -- {n_p} P + {k - n_p} G zones.")


if __name__ == "__main__":
    main()
