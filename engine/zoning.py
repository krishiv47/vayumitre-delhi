"""Workload-balanced zoning.

Every station carries a workload: the spray passes it needs to get back below
the pre-position limit (one pass cuts PM2.5 by SPRAY_R), which also tracks the
water it needs. Total workload sets the zone count -- more / dirtier stations
=> more zones, clean air => fewer -- and zones are cut so each carries about
the same workload (so about the same trucks and water), while staying compact.

Zones built here are all G (geographic) zones. P (priority / "red") zones are a
separate, dynamic overlay: see ``split_p``.
"""
from __future__ import annotations

import math

SPRAY_R = 0.425            # per-pass PM2.5 reduction (matches the portals)
PASSES_PER_ZONE = 12       # target workload per zone (~ 1 truck-shift of water)
K_MIN, K_MAX = 1, 8
P_MAX_CENTERS = 2          # a P zone holds at most this many CPCB stations
P_SIGMA = 1.0              # a station is a hotspot if >= cluster mean + this*std
P_MERGE_KM = 10.0          # two hotspots this close can share one P zone
_EPS = 0.05                # NORMAL stations still count a little, so they get placed


def passes_needed(pm25: float, rising: float) -> int:
    if pm25 < rising:
        return 0
    return max(1, math.ceil(math.log(rising / pm25) / math.log(1 - SPRAY_R)))


def _split(items, k):
    """Recursive weighted bisection into k compact, equal-workload parts."""
    if k <= 1 or len(items) <= 1:
        return [items]
    lat0 = sum(i["lat"] for i in items) / len(items)
    xs = [i["lng"] * math.cos(math.radians(lat0)) for i in items]
    ys = [i["lat"] for i in items]
    axis = (lambda i: i["lng"]) if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else (lambda i: i["lat"])
    items = sorted(items, key=axis)
    k1 = k // 2
    goal = sum(i["w"] for i in items) * k1 / k
    acc, cut = 0.0, 1
    for n, it in enumerate(items, 1):
        acc += it["w"]
        cut = n
        if acc >= goal:
            break
    cut = min(max(cut, 1), len(items) - 1)
    return _split(items[:cut], k1) + _split(items[cut:], k - k1)


def build_zones(stations, active: float, rising: float, k: int | None = None):
    """stations: dicts with station, lat, lng, pm25. Returns (zones, k, total_workload).

    zone = {"kind": "priority"|"geo", "stations": [...], "workload": int,
            "tier_counts": {...}}. Geo zones get an ``id`` (their order).
    """
    items = []
    for s in stations:
        p = passes_needed(s["pm25"], rising)
        items.append({**s, "p": p, "w": p if p else _EPS})
    total = sum(i["p"] for i in items)
    if k is None:
        k = max(K_MIN, min(K_MAX, round(total / PASSES_PER_ZONE)))
    k = min(k, len(items))

    zones, gid = [], 0
    for part in _split(items, k):
        n_active = sum(1 for i in part if i["pm25"] >= active)
        z = {"kind": "geo", "id": gid, "stations": [i["station"] for i in part],
             "workload": sum(i["p"] for i in part),
             "tier_counts": {"ACTIVE": n_active,
                             "PRE_POSITION": sum(1 for i in part if rising <= i["pm25"] < active),
                             "NORMAL": sum(1 for i in part if i["pm25"] < rising)}}
        gid += 1
        zones.append(z)
    return zones, k, total


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [a["lat"], a["lng"], b["lat"], b["lng"]])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def split_p(members, active: float):
    """Pull dynamic P ("red") zones out of one G cluster's live members.

    A hotspot is a station at/above the ACTIVE limit that also stands out from
    its own cluster (>= mean + P_SIGMA*std). Each P zone is the worst remaining
    hotspot plus, optionally, its nearest other hotspot within P_MERGE_KM --
    never more than P_MAX_CENTERS stations. Returns (g_members, p_groups); the
    P stations are served standalone and leave the G route. Recomputed every
    hour / refresh, so P zones appear and disappear with the air.
    """
    if len(members) < 2:
        return members, []
    vals = [m["pm25"] for m in members]
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    hot = sorted((m for m in members if m["pm25"] >= active and m["pm25"] > mu
                  and m["pm25"] >= mu + P_SIGMA * sd), key=lambda m: -m["pm25"])
    groups = []
    while hot:
        first = hot.pop(0)
        grp = [first]
        near = [h for h in hot if _km(first, h) <= P_MERGE_KM]
        if near and P_MAX_CENTERS >= 2:
            partner = min(near, key=lambda h: _km(first, h))
            hot.remove(partner)
            grp.append(partner)
        groups.append(grp)
    taken = {m["station"] for g in groups for m in g}
    return [m for m in members if m["station"] not in taken], groups


TRUCKS_PER_ZONE = 5        # fixed trucks assigned to EVERY G zone
PASSES_PER_TRUCK = PASSES_PER_ZONE / TRUCKS_PER_ZONE   # one truck's share of spray passes (~2.4)


def p_truck_need(group, rising: float):
    """Standalone trucks a P red zone needs: its spray passes / passes-per-truck
    (at least 1). Returns (trucks, passes)."""
    passes = sum(passes_needed(m["pm25"], rising) for m in group)
    return max(1, math.ceil(passes / PASSES_PER_TRUCK)), passes


def apply_fixed_trucks(entries, per_zone: int = TRUCKS_PER_ZONE):
    """Every G zone carries the same fixed ``per_zone`` trucks (zones are sized so
    that many trucks can cover their workload); each P red zone gets its own
    standalone trucks sized to its workload (``p_truck_need``). ``trucks_needed`` stays as the zone's live demand, so we can
    report ``short`` (demand above the fixed crew) and ``idle`` (crew above demand).
    """
    deployed = short = idle = g_trucks = p_trucks = 0
    for e in entries:
        n = max(1, e.get("trucks_needed", 0)) if e["kind"] == "priority" else per_zone
        if e["kind"] == "priority":
            p_trucks += n
        else:
            g_trucks += n
        need = e.get("trucks_needed", 0)
        e["trucks_assigned"] = n
        deployed += n
        short += max(0, need - n)
        idle += max(0, n - need)
    return {"per_zone": per_zone, "g_trucks": g_trucks, "p_trucks": p_trucks, "deployed": deployed, "short": short, "idle": idle}


def zone_regions(zones, coord_by_name, boundary_latlng):
    """Voronoi territory per G zone (cell of its station centroid), clipped to
    the Delhi boundary. Returns ``{"G<id>": {"rings": [[[lat,lng],..]], "area_km2": x}}``
    -- the shape the map pages already draw."""
    import numpy as np
    from scipy.spatial import Voronoi
    from shapely.geometry import MultiPolygon, Polygon

    bpoly = Polygon([(lng, lat) for lat, lng in boundary_latlng])
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    cents = [(float(np.mean([coord_by_name[n][1] for n in z["stations"]])),
              float(np.mean([coord_by_name[n][0] for n in z["stations"]]))) for z in zones]
    pts = np.asarray(cents)
    cx, cy = pts.mean(axis=0)
    minx, miny, maxx, maxy = bpoly.bounds
    span = max(maxx - minx, maxy - miny) * 10
    far = np.array([[cx - span, cy - span], [cx + span, cy - span], [cx - span, cy + span], [cx + span, cy + span]])
    vor = Voronoi(np.vstack([pts, far]))
    out = {}
    for i, z in enumerate(zones):
        verts = vor.regions[vor.point_region[i]]
        if -1 in verts or not verts:
            continue
        clip = Polygon([vor.vertices[v] for v in verts]).intersection(bpoly)
        if clip.is_empty:
            continue
        geoms = clip.geoms if isinstance(clip, MultiPolygon) else [clip]
        rings = [[[round(la, 4), round(lo, 4)] for lo, la in g.exterior.coords] for g in geoms if g.area > 0]
        out[f"G{z['id']}"] = {"rings": rings,
                              "area_km2": round(clip.area * (111.0 ** 2) * math.cos(math.radians(28.6)), 1)}
    return out
