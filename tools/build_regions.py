#!/usr/bin/env python3
"""
Builds two things for the map:

  1. delhi_boundary.json  -- the real Delhi NCT admin boundary (OSM relation
     1942586), stitched from its 134 way-members into one ordered ring.

  2. cluster_regions.json -- each GEO cluster's *territory*, not just a hull
     around its stations. Territory = Voronoi cell of the cluster's centroid,
     CLIPPED to the Delhi boundary. This makes clusters tile the whole city:
     a cluster in a sparse/empty area automatically gets a large cell (it owns
     all the empty space nearest to it), and neighbouring clusters share an
     edge instead of leaving gaps -- exactly "grow the cluster to fill the
     boundary." Priority (extreme-station) clusters are left as-is; they're
     point interventions, not territories.

Run (after fetch_live_clusters.py):
    python3 build_regions.py
"""
from __future__ import annotations
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data"))  # work inside data/

import json
import os

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union


def stitch_boundary(overpass_path: str) -> list:
    """Join the relation's way-members end-to-end into one closed ring
    (the longest one, i.e. the outer NCT border)."""
    d = json.load(open(overpass_path))
    rel = d["elements"][0]
    segs = []
    for m in rel.get("members", []):
        if m.get("type") == "way" and m.get("role") in ("outer", ""):
            geom = m.get("geometry") or []
            pts = [(p["lon"], p["lat"]) for p in geom]
            if len(pts) >= 2:
                segs.append(pts)

    # greedy chain: connect segments whose endpoints match (within eps)
    eps = 1e-6
    def close(a, b):
        return abs(a[0]-b[0]) < eps and abs(a[1]-b[1]) < eps

    chains = []
    used = [False]*len(segs)
    for i in range(len(segs)):
        if used[i]:
            continue
        chain = list(segs[i]); used[i] = True
        extended = True
        while extended:
            extended = False
            for j in range(len(segs)):
                if used[j]:
                    continue
                s = segs[j]
                if close(chain[-1], s[0]):
                    chain += s[1:]; used[j] = True; extended = True
                elif close(chain[-1], s[-1]):
                    chain += s[::-1][1:]; used[j] = True; extended = True
                elif close(chain[0], s[-1]):
                    chain = s[:-1] + chain; used[j] = True; extended = True
                elif close(chain[0], s[0]):
                    chain = s[::-1][:-1] + chain; used[j] = True; extended = True
        chains.append(chain)

    longest = max(chains, key=len)
    # return as [lat, lng] for Leaflet
    return [[lat, lng] for lng, lat in longest]


def voronoi_polygons(points, bbox):
    """Finite Voronoi polygons for `points`, using a ring of far-away mirror
    points so every real cell is bounded. Returns list of shapely Polygons
    aligned with `points`."""
    pts = np.asarray(points)
    # add 4 distant points so all real regions are finite
    cx, cy = pts.mean(axis=0)
    span = max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * 10
    far = np.array([[cx-span, cy-span], [cx+span, cy-span],
                    [cx-span, cy+span], [cx+span, cy+span]])
    allpts = np.vstack([pts, far])
    vor = Voronoi(allpts)

    polys = []
    for i in range(len(pts)):
        region_idx = vor.point_region[i]
        verts = vor.regions[region_idx]
        if -1 in verts or not verts:
            polys.append(None)
            continue
        poly = Polygon([vor.vertices[v] for v in verts])
        polys.append(poly)
    return polys


def main():
    # Prefer the raw OSM relation if present; otherwise reuse the already-stitched
    # delhi_boundary.json (the deployed copy doesn't ship the raw overpass dump).
    if os.path.exists("delhi_overpass.json"):
        boundary_latlng = stitch_boundary("delhi_overpass.json")
        with open("delhi_boundary.json", "w") as f:
            json.dump({"boundary": boundary_latlng}, f)
        print(f"delhi_boundary.json: {len(boundary_latlng)} points (restitched)")
    else:
        boundary_latlng = json.load(open("delhi_boundary.json"))["boundary"]
        print(f"delhi_boundary.json: {len(boundary_latlng)} points (existing)")

    # shapely boundary polygon in (lng,lat)
    boundary = Polygon([(lng, lat) for lat, lng in boundary_latlng])
    if not boundary.is_valid:
        boundary = boundary.buffer(0)

    cv2 = json.load(open("clusters_v2.json"))
    station_by_name = {s["station"]: s for s in cv2["stations"]}

    # geo-cluster centroids (in lng,lat)
    geo, centroids, labels = [], [], []
    p_idx = 0
    for c in cv2["clusters"]:
        # every zone (P and G) now owns a territory, since zones are workload-balanced
        if c["kind"] == "priority":
            p_idx += 1
            label = f"P{p_idx}"
        else:
            label = f"G{c['id']}"
        pts = [station_by_name[n] for n in c["stations"] if n in station_by_name]
        if not pts:
            continue
        lng = np.mean([p["lng"] for p in pts])
        lat = np.mean([p["lat"] for p in pts])
        centroids.append((lng, lat))
        labels.append(label)
        geo.append(c)

    minx, miny, maxx, maxy = boundary.bounds
    polys = voronoi_polygons(centroids, (minx, miny, maxx, maxy))

    regions = {}
    for label, poly in zip(labels, polys):
        if poly is None:
            continue
        clipped = poly.intersection(boundary)
        if clipped.is_empty:
            continue
        # a cell can be split by the boundary into pieces -- keep them all
        geoms = clipped.geoms if isinstance(clipped, MultiPolygon) else [clipped]
        rings = []
        for g in geoms:
            if g.area <= 0:
                continue
            rings.append([[lat, lng] for lng, lat in g.exterior.coords])
        regions[label] = {"rings": rings,
                           "area_km2": round(clipped.area * (111.0 ** 2) *
                                             np.cos(np.radians(28.6)), 1)}
        print(f"  {label}: {len(rings)} ring(s), ~{regions[label]['area_km2']} km2")

    with open("cluster_regions.json", "w") as f:
        json.dump({"generated_at": cv2["generated_at"], "regions": regions}, f)
    print(f"cluster_regions.json: {len(regions)} geo territories")


if __name__ == "__main__":
    main()
