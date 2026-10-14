#!/usr/bin/env python3
"""PDF of the zone map for every month (illustrative, modelled levels).

Same monthly model + workload-balanced zoning as monthly_zones.py. Page 1 is a
12-month overview; pages 2-13 are one full map per month.
Writes Vayumitre_monthly_zone_maps.pdf in the project root.
"""
from __future__ import annotations
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_os.chdir(_os.path.join(_ROOT, "data"))

import json
import math
import sys

sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, "tools"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import MultiPolygon, Polygon

from engine.zoning import TRUCKS_PER_ZONE, build_zones, p_truck_need, split_p
from monthly_zones import DIURNAL, HOTSPOTS, MONTH_MEAN, NAMES, RISING_FRAC, THRESHOLDS, voronoi

_os.chdir(_os.path.join(_ROOT, "data"))   # monthly_zones import also chdirs here
GEO_COLORS = ["#3ddc84", "#4fa8ff", "#c084fc", "#ffb020", "#ff6ec7", "#5fe0d0", "#9aa5ff", "#b0d96a"]
P_COLOR = "#ff5a4a"
TIER_COLOR = {"NORMAL": "#2e8b57", "PRE_POSITION": "#e0a72e", "ACTIVE": "#d0392f"}


def month_zones(m, stations, bpoly, rel, noise, boost):
    q = m // 3 + 1
    active = THRESHOLDS[q]; rising = active * RISING_FRAC
    lv = np.clip(rel * boost * (1 + noise) * MONTH_MEAN[m], 10, None)
    recs = [{"station": s["station"], "lat": s["lat"], "lng": s["lng"], "pm25": float(v)}
            for s, v in zip(stations, lv)]
    zones, k, work = build_zones(recs, active, rising)
    by = {r["station"]: r for r in recs}
    cents = [(np.mean([by[n]["lng"] for n in z["stations"]]),
              np.mean([by[n]["lat"] for n in z["stations"]])) for z in zones]
    g_i, p_groups = 0, []
    for i, (z, poly) in enumerate(zip(zones, voronoi(cents, bpoly.bounds))):
        g_i += 1; z["label"] = f"G{g_i}"; z["color"] = GEO_COLORS[(g_i - 1) % len(GEO_COLORS)]
        clip = poly.intersection(bpoly) if poly is not None else None
        z["geoms"] = ([] if clip is None or clip.is_empty else
                      list(clip.geoms) if isinstance(clip, MultiPolygon) else [clip])
        z["area"] = sum(g.area for g in z["geoms"]) * (111.0 ** 2) * math.cos(math.radians(28.6))
        z["cent"] = cents[i]
    # G zones above are fixed for the month. P red zones are re-found EVERY HOUR
    # (levels follow the day curve); the map shows the busiest hour.
    dmean = sum(DIURNAL) / 24
    hours = []
    for hh in range(24):
        byh = {n: {**r, "pm25": r["pm25"] * DIURNAL[hh] / dmean} for n, r in by.items()}
        gs = []
        for z in zones:
            _, groups = split_p([byh[n] for n in z["stations"]], active)
            for grp in groups:
                need, passes = p_truck_need(grp, rising)
                gs.append({"parent": z["label"], "group": grp, "trucks": need, "passes": passes})
        hours.append(gs)
    if NAMES[m] == "Dec":      # December: use the December page's own hourly plans (single source of truth)
        hr = json.load(open("december_hourly.json"))["hours"]
        hours = [[{"parent": f"G{int(c['parent'][1:]) + 1}", "group": c["checkpoints"],
                   "trucks": c["trucks_needed"], "passes": c["passes"]}
                  for c in h["plan"]["clusters"] if c["kind"] == "priority"] for h in hr]
    p_trucks_h = [sum(g["trucks"] for g in gs) for gs in hours]
    peak = max(range(24), key=lambda h: (p_trucks_h[h], len(hours[h])))
    p_groups = [(g["parent"], g["group"], g["trucks"], g["passes"]) for g in hours[peak]]
    return {"month": NAMES[m], "q": q, "active": active, "rising": rising,
            "mean": float(lv.mean()), "zones": zones, "work": work, "p_groups": p_groups, "p_trucks_h": p_trucks_h, "p_count_h": [len(g) for g in hours], "peak_h": peak, "recs": recs}


def draw(ax, d, bpoly, detail):
    for z in d["zones"]:
        for g in z["geoms"]:
            ax.add_patch(MplPoly(list(g.exterior.coords), closed=True, facecolor=z["color"],
                                 alpha=0.22, edgecolor=z["color"], linewidth=1.6 if detail else 1.0,
                                 linestyle="--", hatch="//" if z["kind"] == "priority" else None))
    bx, by = bpoly.exterior.xy
    ax.plot(bx, by, color="#222", linewidth=1.4)
    for r in d["recs"]:
        t = "ACTIVE" if r["pm25"] >= d["active"] else "PRE_POSITION" if r["pm25"] >= d["rising"] else "NORMAL"
        ax.scatter(r["lng"], r["lat"], s=34 if detail else 9, c=TIER_COLOR[t],
                   edgecolors="white" if t == "ACTIVE" else "#111", linewidths=0.5, zorder=3)
    for n, (parent, grp, _t, _p) in enumerate(d["p_groups"], 1):
        lng = [m["lng"] for m in grp]; lat = [m["lat"] for m in grp]
        ax.scatter(lng, lat, s=(300 if detail else 70), facecolors="none", edgecolors=P_COLOR,
                   linewidths=2.2 if detail else 1.2, zorder=5)
        if len(grp) == 2:
            ax.plot(lng, lat, color=P_COLOR, linewidth=2, zorder=4)
        if detail:
            ax.text(np.mean(lng) + 0.012, np.mean(lat) + 0.012, f"P{n}", color=P_COLOR,
                    fontsize=11, fontweight="bold", zorder=6)
    for z in d["zones"]:
        cx, cy = z["cent"]
        ax.text(cx, cy, z["label"], ha="center", va="center", zorder=4,
                fontsize=13 if detail else 8, fontweight="bold",
                color="#111", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=z["color"], lw=1.4))
    ax.set_aspect(1 / math.cos(math.radians(28.6)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def caption(d):
    zs = d["zones"]
    pk = max(d["p_trucks_h"])
    return (f"{len(zs)} G zone{'s' if len(zs) != 1 else ''} fixed all month ({TRUCKS_PER_ZONE} trucks each = {TRUCKS_PER_ZONE * len(zs)})  ·  "
            f"P red zones change hourly: peak +{pk} standalone truck{'s' if pk != 1 else ''}\n"
            f"month workload {d['work']} passes  ·  mean PM2.5 {d['mean']:.0f}  ·  active limit {d['active']:.0f}")


def main():
    cv2 = json.load(open("december_clusters.json"))   # SAME station levels the December page uses
    bpoly = Polygon([(lng, lat) for lat, lng in json.load(open("delhi_boundary.json"))["boundary"]])
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    stations = cv2["stations"]
    # month level = the December scenario's day-average level, scaled by monthly mean / December mean
    # (so December here == the December page exactly)
    rel = np.array([s["pm25"] for s in stations]) / MONTH_MEAN[11]
    noise = np.zeros(len(stations))
    boost = np.ones(len(stations))
    months = [month_zones(m, stations, bpoly, rel, noise, boost) for m in range(12)]

    out = _os.path.join(_ROOT, "Vayumitre_monthly_zone_maps.pdf")
    note = "Illustrative: modelled monthly PM2.5 levels. G zones fixed per month (5 trucks each); P red zones (≤2 stations, own trucks) re-found hourly."
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(3, 4, figsize=(16.5, 11.7))
        fig.suptitle("Vayumitre — Delhi zone clusters, all 12 months", fontsize=18, fontweight="bold", y=0.985)
        for ax, d in zip(axes.ravel(), months):
            draw(ax, d, bpoly, False)
            ax.set_title(f"{d['month']}  (Q{d['q']})", fontsize=13, fontweight="bold", loc="left")
            ax.text(0.5, -0.02, f"{len(d['zones'])} G ({TRUCKS_PER_ZONE * len(d['zones'])} trucks) · P peak +{max(d['p_trucks_h'])} truck{'' if max(d['p_trucks_h']) == 1 else 's'} · mean {d['mean']:.0f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333")
        fig.text(0.5, 0.012, note, ha="center", fontsize=9, color="#555")
        fig.subplots_adjust(top=0.93, bottom=0.05, left=0.02, right=0.98, hspace=0.2, wspace=0.05)
        pdf.savefig(fig); plt.close(fig)

        for d in months:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.04, 0.43, 0.92, 0.45])
            draw(ax, d, bpoly, True)
            fig.text(0.04, 0.955, f"{d['month']} — zone map", fontsize=20, fontweight="bold")
            fig.text(0.04, 0.9, caption(d), fontsize=9.5, color="#333", va="bottom", linespacing=1.5)
            # P standalone trucks needed, hour by hour
            cax = fig.add_axes([0.10, 0.335, 0.86, 0.055])
            ph = d["p_trucks_h"]
            cax.bar(range(24), ph, color=P_COLOR, width=0.75)
            cax.set_xticks(range(0, 24, 3)); cax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=7)
            cax.set_ylim(0, max(max(ph), 1) * 1.25); cax.tick_params(axis="y", labelsize=7)
            cax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            for sp in ("top", "right"):
                cax.spines[sp].set_visible(False)
            cax.set_title(f"P standalone trucks needed by hour (G crews fixed at {TRUCKS_PER_ZONE * len(d['zones'])}); map = busiest hour {d['peak_h']:02d}:00",
                          fontsize=8, loc="left")
            rows = [[z["label"], "G (fixed crew)", len(z["stations"]), z["tier_counts"]["ACTIVE"],
                     z["workload"], TRUCKS_PER_ZONE, f"{z['area']:.0f}"] for z in d["zones"]]
            for n, (parent, grp, trucks, passes) in enumerate(d["p_groups"], 1):
                rows.append([f"P{n}", f"P in {parent} (standalone)", len(grp), len(grp), passes, trucks, "–"])
            tax = fig.add_axes([0.04, 0.055, 0.92, 0.245]); tax.axis("off")
            tb = tax.table(cellText=rows, colLabels=["Zone", "Type", "Stations", "Active", "Workload (passes)", "Trucks", "Area km²"],
                           loc="upper center", cellLoc="center",
                           colWidths=[0.07, 0.27, 0.11, 0.09, 0.19, 0.11, 0.16])
            many = len(rows) > 9
            tb.auto_set_font_size(False); tb.set_fontsize(7.5 if many else 8.5); tb.scale(1, 0.92 if many else 1.15)
            for i, z in enumerate(d["zones"], 1):
                tb[i, 0].set_facecolor(z["color"] + "55")
            fig.text(0.04, 0.025, "tiers: green normal · amber pre-position · red active   |   dashed region = G territory, red ring = P red zone (≤2 stations)",
                     fontsize=8.5, color="#555")
            fig.text(0.04, 0.008, note, fontsize=7.5, color="#777")
            pdf.savefig(fig); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
