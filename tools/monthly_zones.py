#!/usr/bin/env python3
"""Zone count + area for every calendar month (illustrative / synthetic).

Real history exists only for the current month, so each month's station levels
are modelled: the live station pattern (relative to its own mean) scaled to an
approximate Delhi monthly-mean PM2.5. The SAME demand rules as the live
pipeline then decide the zones -- priority = >= active limit and >= mean+1sigma,
geo k ~ stations needing service / 7 (clamped 2-6) -- using that month's
quarterly limit. Areas are Voronoi cells of the geo centroids clipped to Delhi.

Writes data/monthly_zones.json and prints a table.
"""
from __future__ import annotations
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data"))

import json
import math

import sys
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import numpy as np
from engine.zoning import build_zones, split_p
from shapely.geometry import Polygon, MultiPolygon
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi

THRESHOLDS = {1: 174.2, 2: 170.0, 3: 95.5, 4: 190.0}
RISING_FRAC, PRIORITY_SIGMA, MERGE_KM = 0.8, 1.0, 10.0
TARGET_PER_CLUSTER, K_MIN, K_MAX = 7, 2, 6
# Approximate Delhi monthly-mean PM2.5 (ug/m3) -- illustrative seasonal shape.
MONTH_MEAN = [190, 135, 95, 90, 85, 75, 45, 40, 50, 110, 215, 225]
DIURNAL = [1.18,1.20,1.20,1.18,1.16,1.18,1.22,1.26,1.28,1.20,1.05,0.92,0.82,0.76,0.74,0.75,
           0.82,0.95,1.08,1.16,1.20,1.20,1.19,1.18]   # same winter diurnal shape as build_december
NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HOTSPOTS = ["Bawana", "Anand Vihar", "Wazirpur", "Jahangirpuri", "Narela",
            "Mundka", "Rohini", "Punjabi Bagh", "Nehru Nagar"]


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def merge(stations, km):
    parent = list(range(len(stations)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            if hav((stations[i]["lat"], stations[i]["lng"]), (stations[j]["lat"], stations[j]["lng"])) <= km:
                parent[find(i)] = find(j)
    groups = {}
    for i, s in enumerate(stations):
        groups.setdefault(find(i), []).append(s)
    return list(groups.values())


def voronoi(centroids, bounds):
    pts = np.asarray(centroids); cx, cy = pts.mean(axis=0)
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 10
    far = np.array([[cx - span, cy - span], [cx + span, cy - span], [cx - span, cy + span], [cx + span, cy + span]])
    vor = Voronoi(np.vstack([pts, far])); out = []
    for i in range(len(pts)):
        v = vor.regions[vor.point_region[i]]
        out.append(None if (-1 in v or not v) else Polygon([vor.vertices[k] for k in v]))
    return out


def main():
    cv2 = json.load(open("december_clusters.json"))   # SAME station levels the December page uses
    bpoly = Polygon([(lng, lat) for lat, lng in json.load(open("delhi_boundary.json"))["boundary"]])
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    stations = cv2["stations"]
    # month level = the December scenario's day-average level, scaled by monthly mean / December mean
    # (so December here == the December page exactly)
    rel = np.array([s["pm25"] for s in stations]) / MONTH_MEAN[11]
    rng = np.random.default_rng(7)
    noise = np.zeros(len(stations))
    boost = np.ones(len(stations))

    rows = []
    print(f"{'Month':5} {'limit':>6} {'mean':>5} {'P':>4} {'G':>3} {'work':>5}  geo-zone areas (km2)")
    for m in range(12):
        q = m // 3 + 1
        active = THRESHOLDS[q]; rising = active * RISING_FRAC
        lv = np.clip(rel * boost * (1 + noise) * MONTH_MEAN[m], 10, None)
        rec = [{"lat": s["lat"], "lng": s["lng"], "pm25": float(v)} for s, v in zip(stations, lv)]
        mu = float(lv.mean())
        recs = [{"station": s["station"], "lat": s["lat"], "lng": s["lng"], "pm25": float(v)}
                for s, v in zip(stations, lv)]
        zones, k, work = build_zones(recs, active, rising)
        by = {r["station"]: r for r in recs}
        n_prio = sum(len(split_p([by[n] for n in z["stations"]], active)[1]) for z in zones)
        cents = [(np.mean([by[n]["lng"] for n in z["stations"]]),
                  np.mean([by[n]["lat"] for n in z["stations"]])) for z in zones]
        # extra P-zone trucks hour by hour: levels follow the diurnal curve, P red
        # zones are re-found each hour inside the (fixed) G zones
        dmean = sum(DIURNAL) / 24
        p_hour = []
        for hh in range(24):
            byh = {r["station"]: {**r, "pm25": r["pm25"] * DIURNAL[hh] / dmean} for r in recs}
            p_hour.append(sum(len(split_p([byh[n] for n in z["stations"]], active)[1]) for z in zones))
        if NAMES[m] == "Dec":     # December: take the December page's own hourly plans
            p_hour = [sum(1 for c in h["plan"]["clusters"] if c["kind"] == "priority")
                      for h in json.load(open("december_hourly.json"))["hours"]]
        areas = []
        for poly in voronoi(cents, bpoly.bounds):
            clip = poly.intersection(bpoly) if poly is not None else None
            areas.append(round(clip.area * (111.0 ** 2) * math.cos(math.radians(28.6)), 1) if clip is not None and not clip.is_empty else 0.0)
        areas = sorted(areas, reverse=True)
        rows.append({"month": NAMES[m], "quarter": q, "active_limit": active, "mean_pm25": round(mu),
                     "priority_zones": n_prio, "p_trucks_by_hour": p_hour, "base_trucks": 5 * k, "p_peak": max(p_hour), "p_hours": sum(1 for x in p_hour if x), "geo_zones": k, "total_workload": work, "geo_areas_km2": areas,
                     "total_km2": round(sum(areas), 1)})
        print(f"{NAMES[m]:5} base {5*k:2d} +P peak {max(p_hour):2d} in {sum(1 for x in p_hour if x):2d}h/day | {active:6.1f} {mu:5.0f} {n_prio:4d} {k:3d} {work:5d}  {areas}")
    json.dump({"note": "illustrative: modelled monthly levels, live-pipeline rules", "months": rows},
              open("monthly_zones.json", "w"), indent=1)


if __name__ == "__main__":
    main()
