#!/usr/bin/env python3
"""
Full SYNTHETIC December (Q4 winter-smog) scenario, generated end to end so the
December portal has its OWN clusters, territories, forecasts and plan -- formed
the same way as the live portal, but from December levels (which produce
different priority corners and different geographic zones than summer).

Pipeline (one script, consistent ordering):
  1. Synthetic December levels + 24h forecast curves per station.
  2. Tiered + geographic clustering ON THE DECEMBER LEVELS:
       Stage 1 -- priority "corner" clusters = the RELATIVE extremes. In a
         bad-air day almost every station is over the absolute ACTIVE limit,
         so "extreme" is defined relative to December's own distribution
         (>= EXTREME_PCTL percentile), not the fixed 190 threshold -- else
         all 42 stations would be priority and nothing would geo-cluster.
       Stage 2 -- geography-only GMM (lat,lng) on the rest, largest k.
  3. Voronoi territories per geo cluster, clipped to the Delhi boundary.
  4. Spray routes + truck sizing on the December levels.

Writes: december_forecasts.json, december_clusters.json,
        december_regions.json, december_plan.json
All deterministic (fixed seed).
"""
from __future__ import annotations
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data"))  # work inside data/

import json
import math
import random
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import maximum_filter
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon
from sklearn.mixture import GaussianMixture

random.seed(1206); np.random.seed(1206)

IST = timezone(timedelta(hours=5, minutes=30))
Q4_ACTIVE = 190.0
RISING_FRAC = 0.8
Q4_RISING = Q4_ACTIVE * RISING_FRAC
EXTREME_PCTL = 80          # (legacy) percentile fallback
PRIORITY_SIGMA = 1.0       # priority zone = severe outlier: >= mean + this*std (and >= active)
MERGE_RADIUS_KM = 10.0
TARGET_PER_CLUSTER = 7     # aim for ~this many service stations per geo cluster
GEO_K_MIN, GEO_K_MAX = 2, 6

EXTREME = ["Bawana", "Anand Vihar", "Wazirpur", "Jahangirpuri", "Narela",
           "Mundka", "Rohini", "Punjabi Bagh", "Nehru Nagar"]
DIURNAL = {0:1.18,1:1.20,2:1.20,3:1.18,4:1.16,5:1.18,6:1.22,7:1.26,8:1.28,9:1.20,
           10:1.05,11:0.92,12:0.82,13:0.76,14:0.74,15:0.75,16:0.82,17:0.95,18:1.08,
           19:1.16,20:1.20,21:1.20,22:1.19,23:1.18}


def tier(pm): return "ACTIVE" if pm>=Q4_ACTIVE else "PRE_POSITION" if pm>=Q4_RISING else "NORMAL"

def haversine_km(a,b):
    r=6371.0; la1,lo1,la2,lo2=map(math.radians,[a[0],a[1],b[0],b[1]])
    dla,dlo=la2-la1,lo2-lo1
    h=math.sin(dla/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return 2*r*math.asin(math.sqrt(h))

def synth_level(station, summer_pm):
    base=180+summer_pm*1.15+random.gauss(0,28)
    if any(k in station for k in EXTREME): base+=random.uniform(90,170)
    return round(max(60.0,base),1)

def synth_curve(now, issued_hour):
    am=DIURNAL[issued_hour%24]; t0=datetime.now(IST).replace(minute=0,second=0,microsecond=0)
    times,vals=[],[]
    for k in range(1,25):
        h=(issued_hour+k)%24
        v=now*(DIURNAL[h]/am)*(1+random.gauss(0,0.04))
        times.append((t0+timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M")); vals.append(round(max(30.0,v),1))
    return times,vals

def local_maxima(coords,values,names):
    lp=(coords[:,0].max()-coords[:,0].min())*0.12; op=(coords[:,1].max()-coords[:,1].min())*0.12
    la=np.linspace(coords[:,0].min()-lp,coords[:,0].max()+lp,140)
    lo=np.linspace(coords[:,1].min()-op,coords[:,1].max()+op,140)
    LAT,LON=np.meshgrid(la,lo,indexing="ij"); grid=np.column_stack([LAT.ravel(),LON.ravel()])
    surf=RBFInterpolator(coords,values,kernel="thin_plate_spline",smoothing=1.0)(grid).reshape(LAT.shape)
    nb=maximum_filter(surf,size=7,mode="nearest"); peaks=[]
    for i,j in zip(*np.where(surf>=nb)):
        v=float(surf[i,j])
        if v<Q4_RISING: continue
        d=np.hypot(coords[:,0]-LAT[i,j],coords[:,1]-LON[i,j])
        peaks.append({"lat":round(float(LAT[i,j]),4),"lng":round(float(LON[i,j]),4),
                      "value":round(v,1),"state":tier(v),"nearest_station":names[int(np.argmin(d))]})
    peaks.sort(key=lambda p:-p["value"]); kept=[]
    for p in peaks:
        if all((p["lat"]-k["lat"])**2+(p["lng"]-k["lng"])**2>1.5e-4 for k in kept): kept.append(p)
    return kept[:15]

def greedy_distance_cluster(stations, radius_km):
    n=len(stations); parent=list(range(n))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        ri,rj=find(i),find(j);
        if ri!=rj: parent[ri]=rj
    for i in range(n):
        for j in range(i+1,n):
            if haversine_km((stations[i]["lat"],stations[i]["lng"]),
                            (stations[j]["lat"],stations[j]["lng"]))<=radius_km: union(i,j)
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(stations[i])
    return list(groups.values())

def geo_cluster(coords, k, seed=0):
    """Geography-only GMM into EXACTLY k compact zones."""
    k=min(k, len(coords))
    gmm=GaussianMixture(n_components=k,covariance_type="full",random_state=seed,n_init=8).fit(coords)
    return k, gmm.predict(coords)

def nn_order(pts):
    rem=sorted(pts,key=lambda p:-p["pm25"]); route=[rem.pop(0)]
    while rem:
        last=route[-1]; rem.sort(key=lambda p:haversine_km((last["lat"],last["lng"]),(p["lat"],p["lng"]))); route.append(rem.pop(0))
    return route

def voronoi_regions(centroids, boundary):
    pts=np.asarray(centroids); cx,cy=pts.mean(axis=0)
    minx,miny,maxx,maxy=boundary.bounds; span=max(maxx-minx,maxy-miny)*10
    far=np.array([[cx-span,cy-span],[cx+span,cy-span],[cx-span,cy+span],[cx+span,cy+span]])
    vor=Voronoi(np.vstack([pts,far])); out=[]
    for i in range(len(pts)):
        verts=vor.regions[vor.point_region[i]]
        if -1 in verts or not verts: out.append(None); continue
        out.append(Polygon([vor.vertices[v] for v in verts]))
    return out


# ---- SYNTHETIC pollutant mix (December scenario has no real multi-pollutant feed) ----
# Each station gets a deterministic source profile; its mix follows PM2.5 and the hour of
# day. Clearly synthetic: it exists so the cause-attribution UI has something to show.
PROFILES = {   # traffic, industry, burn, dust (sum 1)
    "Anand Vihar":(.45,.10,.35,.10), "Bawana":(.05,.45,.35,.15), "Narela":(.05,.45,.35,.15),
    "Mundka":(.05,.45,.35,.15), "Wazirpur":(.20,.40,.30,.10), "Jahangirpuri":(.20,.40,.30,.10),
    "Rohini":(.35,.15,.40,.10), "Punjabi Bagh":(.45,.10,.35,.10), "Nehru Nagar":(.45,.10,.35,.10),
    "Okhla":(.15,.45,.25,.15), "Dwarka":(.25,.10,.30,.35), "Vivek Vihar":(.30,.15,.40,.15),
}
DEFAULT_PROFILE = (.35, .10, .40, .15)

def profile_of(station):
    for k, v in PROFILES.items():
        if k in station: return v
    return DEFAULT_PROFILE

def synth_mix(station, pm25, hod):
    import zlib
    tr, ind, burn, dust = profile_of(station)
    def j(tag): return 1 + ((zlib.crc32(f"{station}|{hod}|{tag}".encode()) % 1000)/1000 - 0.5)*0.16
    night = hod >= 18 or hod <= 8; rush = hod in (7,8,9,10,17,18,19,20,21)
    f = min(0.8, max(0.25, 0.30 + 0.9*burn + 0.15*tr + 0.1*ind - 0.6*dust))   # PM2.5/PM10
    sc = pm25 / 350.0
    return {"PM2.5": round(pm25,1), "PM10": round(pm25/f*j("pm10"),1),
            "NO2": round((18 + 100*tr*sc*(1.25 if rush else 1.0) + 15*ind)*j("no2"),1),
            "SO2": round((8 + 70*ind*sc)*j("so2"),1),
            "CO": round((0.7 + 2.2*burn*(1.4 if night else 0.6)*sc + 1.0*tr*sc)*j("co"),2),
            "OZONE": round((15 + (45 if 11 <= hod <= 17 else 0)*(1 - min(1, pm25/500)))*j("o3"),1),
            "NH3": round((15 + 40*burn*sc)*j("nh3"),1)}


def main():
    import argparse
    ap=argparse.ArgumentParser(description="Synthetic December scenario generator")
    ap.add_argument("--geo-k", type=int, default=None,
                    help="force the TOTAL number of zones. Default: AUTO -- scaled to total "
                         "spray workload (higher PM2.5 -> more zones), ~12 passes per zone.")
    args=ap.parse_args()

    cv2=json.load(open("clusters_v2.json"))          # only for station coords + summer prior
    boundary_latlng=json.load(open("delhi_boundary.json"))["boundary"]
    issued_hour=8

    # 1. synthetic levels + forecasts ---------------------------------------
    stations,readings_now,stations_now,now_by_name=[],{},[],{}
    t0=datetime.now(IST)
    for s in cv2["stations"]:
        now=synth_level(s["station"],s["pm25"]); now_by_name[s["station"]]=now
        readings_now[s["station"]]=now
        stations_now.append({"station":s["station"],"lat":s["lat"],"lng":s["lng"],"pm25":now})
        ct,cv=synth_curve(now,issued_hour)
        hz={"1":cv[0],"3":cv[2],"6":cv[5],"12":cv[11],"24":cv[23]}
        peak=max(hz.values()); peak_h=int(max(hz,key=lambda k:hz[k]))
        stations.append({"id":s["station"],"name":s["station"],"lat":s["lat"],"lng":s["lng"],
            "issued_at":t0.replace(minute=0,second=0,microsecond=0).isoformat(),
            "now":now,"anchored":True,"grid_now":now,"observed_at":t0.isoformat(),
            "h":hz,"peak":round(peak,1),"peak_h":peak_h,"rise":round(peak-now,1),
            "curve_t":ct,"curve_v":cv,"src_history":"SYNTHETIC December (winter diurnal model)",
            "src_model":"Vayumitre synthetic Q4 scenario"})
    coords=np.array([[s["lat"],s["lng"]] for s in stations]); names=[s["name"] for s in stations]
    forecasts={"generated_at":datetime.now(timezone.utc).isoformat(),"quarter":4,
        "active_limit":Q4_ACTIVE,"rising_limit":Q4_RISING,"scenario":"Synthetic December (Q4 winter smog)",
        "stations":stations,"readings_now":readings_now,"stations_now":stations_now,
        "maxima_now":local_maxima(coords,np.array([s["now"] for s in stations]),names),
        "maxima_24h":local_maxima(coords,np.array([s["h"]["24"] for s in stations]),names)}
    json.dump(forecasts,open("december_forecasts.json","w"),indent=1)
    print(f"december_forecasts.json: {len(stations)} stations, "
          f"{sum(1 for s in stations if s['now']>=Q4_ACTIVE)} ACTIVE (>190)")

    # 2. tiered + geo clustering on DECEMBER levels -------------------------
    # G zones are fixed for the whole month: formed from each station's DAY-AVERAGE
    # level (not any single hour). Only the P red zones change hour to hour.
    dmean=sum(DIURNAL.values())/24
    rec=[]
    for s_ in stations:
        lv=round(s_["now"]/DIURNAL[issued_hour]*dmean,1)
        rec.append({"station":s_["name"],"lat":s_["lat"],"lng":s_["lng"],"pm25":lv,"tier":tier(lv)})
    # Workload-balanced zones (engine/zoning.py): count scales with total spray
    # workload (higher PM2.5 -> more zones), each zone ~equal workload (trucks /
    # water), and a zone with 1-2 ACTIVE stations is a P zone.
    import sys; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from engine.causes import attribute, mean_mix
    from engine.zoning import TRUCKS_PER_ZONE, apply_fixed_trucks, build_zones, p_truck_need, split_p, zone_regions
    clusters, k, workload = build_zones(rec, Q4_ACTIVE, Q4_RISING, k=args.geo_k)
    n_priority = sum(1 for c in clusters if c["kind"] == "priority")
    n_geo = len(clusters) - n_priority
    for c in clusters:
        print(f"  {'P' if c['kind']=='priority' else 'G'} zone: {len(c['stations'])} stations, "
              f"workload {c['workload']} passes, {c['tier_counts']['ACTIVE']} active")
    clusters_out={"generated_at":forecasts["generated_at"],"quarter":4,
        "active_limit":Q4_ACTIVE,"rising_limit":Q4_RISING,"total_workload":workload,
        "priority_clusters":n_priority,"geo_clusters":n_geo,
        "clusters":clusters,
        "stations":[{"station":r["station"],"lat":r["lat"],"lng":r["lng"],"pm25":r["pm25"],"tier":r["tier"]} for r in rec]}
    json.dump(clusters_out,open("december_clusters.json","w"),indent=2)
    print(f"december_clusters.json: workload {workload} passes -> {k} zones ({n_priority} P + {n_geo} G)")

    # 3. Voronoi territory per zone (P and G), clipped to Delhi -------------
    bpoly=Polygon([(lng,lat) for lat,lng in boundary_latlng])
    if not bpoly.is_valid: bpoly=bpoly.buffer(0)
    coordm={r["station"]:(r["lat"],r["lng"]) for r in rec}
    labels_geo,centroids=[],[]
    p_i=0
    for c in clusters:
        if c["kind"]=="priority": p_i+=1; labels_geo.append(f"P{p_i}")
        else: labels_geo.append(f"G{c['id']}")
        centroids.append((np.mean([coordm[n][1] for n in c["stations"]]),
                          np.mean([coordm[n][0] for n in c["stations"]])))
    regions={}
    for label,poly in zip(labels_geo, voronoi_regions(centroids,bpoly)):
        if poly is None: continue
        clip=poly.intersection(bpoly)
        if clip.is_empty: continue
        geoms=clip.geoms if isinstance(clip,MultiPolygon) else [clip]
        rings=[[[la,lo] for lo,la in g.exterior.coords] for g in geoms if g.area>0]
        regions[label]={"rings":rings,"area_km2":round(clip.area*(111.0**2)*math.cos(math.radians(28.6)),1)}
    json.dump({"generated_at":forecasts["generated_at"],"regions":regions},
              open("december_regions.json","w"))
    print(f"december_regions.json: {len(regions)} territories")

    # 4. plan builder: routes + truck sizing for a given per-station level map
    def make_plan(level_by_name, zone_list=None, mix=None, hod=issued_hour):
        def cause_of(names):
            if not mix: return None
            return attribute(mean_mix([mix[n] for n in names]), hod)
        cplan, total, p_idx = [], 0, 0
        for c in (zone_list if zone_list is not None else clusters):
            label=f"G{c['id']}"
            members=[{"station":n,"lat":coordm[n][0],"lng":coordm[n][1],
                      "pm25":level_by_name[n],"tier":tier(level_by_name[n])} for n in c["stations"]]
            # dynamic P ("red") zones: hotspots standing out inside this cluster
            # (<=2 stations each), standalone truck, removed from the G route
            members,p_groups=split_p(members,Q4_ACTIVE)
            for grp in p_groups:
                p_idx+=1; og=nn_order(grp); p_need,p_passes=p_truck_need(grp,Q4_RISING); gg=[[m["lat"],m["lng"]] for m in og]
                pd_=sum(haversine_km(gg[i-1],gg[i]) for i in range(1,len(gg)))
                cplan.append({"cause":cause_of([m["station"] for m in grp]),"label":f"P{p_idx}","kind":"priority","parent":label,"n_members":len(grp),
                    "passes":p_passes,"skipped_normal":[],"checkpoints":og,"geometry":gg if len(gg)>1 else None,
                    "distance_km":round(pd_,2) if len(gg)>1 else None,
                    "duration_min":round(pd_/22*60,1) if len(gg)>1 else None,
                    "straight_line_fallback":True,"trucks_needed":p_need,"dedicated_trucks_for":[],
                    "note":f"red zone inside {label} -- standalone truck"})
                total+=p_need
            spray=[m for m in members if m["tier"]!="NORMAL"]; skipped=[m for m in members if m["tier"]=="NORMAL"]
            e={"cause":cause_of([m["station"] for m in members]),"label":label,"kind":c["kind"],"n_members":len(members),"skipped_normal":skipped,
               "checkpoints":[],"geometry":None,"distance_km":None,"duration_min":None,
               "trucks_needed":0,"dedicated_trucks_for":[]}
            if not spray: e["note"]="all NORMAL"; cplan.append(e); continue
            if len(spray)==1: e.update(checkpoints=spray,trucks_needed=1,note="hold position"); total+=1; cplan.append(e); continue
            ordered=nn_order(spray); geom=[[m["lat"],m["lng"]] for m in ordered]
            dist=sum(haversine_km(geom[i-1],geom[i]) for i in range(1,len(geom)))
            arr,acc=[0.0],0.0
            for i in range(1,len(ordered)): acc+=8+haversine_km(geom[i-1],geom[i])/22*60; arr.append(acc)
            late=[{"station":ordered[i]["station"],"deadline_h":0.0,"route_arrival_min":round(arr[i],1)} for i in range(len(ordered)) if arr[i]>0]
            e.update(checkpoints=ordered,geometry=geom,distance_km=round(dist,2),duration_min=round(dist/22*60,1),
                     straight_line_fallback=True,trucks_needed=1+len(late),dedicated_trucks_for=late)
            total+=e["trucks_needed"]; cplan.append(e)
        fleet=apply_fixed_trucks(cplan, TRUCKS_PER_ZONE)
        return {"active_limit":Q4_ACTIVE,"rising_limit":Q4_RISING,"clusters":cplan,"total_trucks_needed":total,"fleet":fleet}

    plan=make_plan(now_by_name, None, {n: synth_mix(n, now_by_name[n], issued_hour) for n in now_by_name}); plan["generated_at"]=forecasts["generated_at"]
    json.dump(plan,open("december_plan.json","w"),indent=1)
    print(f"december_plan.json: {plan['total_trucks_needed']} trucks across {len(plan['clusters'])} clusters")

    # 5. 24-HOUR simulation: one snapshot per hour of the day. Each station's
    # level follows the winter diurnal curve (peak ~08:00 inversion, dip
    # ~14:00 midday mixing), so sensor colours shift and the spray plan is
    # re-solved every hour -> routes and truck counts change through the day.
    daily_base = {n: now_by_name[n] / DIURNAL[issued_hour] for n in now_by_name}
    hours = []
    coords_arr = np.array([[coordm[n][0], coordm[n][1]] for n in now_by_name])
    names_arr = list(now_by_name.keys())
    for hod in range(24):
        levels = {}
        for i, n in enumerate(names_arr):
            noise = 1 + (((zlib.crc32(f"{n}|{hod}".encode()) % 1000)/1000 - 0.5) * 0.06)  # deterministic +-3%
            levels[n] = round(max(30.0, daily_base[n] * DIURNAL[hod] * noise), 1)
        vals = np.array([levels[n] for n in names_arr])
        mix_h = {n: synth_mix(n, levels[n], hod) for n in names_arr}
        hplan = make_plan(levels, None, mix_h, hod)   # same fixed G zones every hour; P zones re-found per hour
        n_active = int(sum(1 for v in vals if v >= Q4_ACTIVE))
        hours.append({
            "h": hod, "label": f"{hod:02d}:00",
            "readings": levels,
            "stations_now": [{"station": n, "lat": coordm[n][0], "lng": coordm[n][1], "pm25": levels[n]}
                              for n in names_arr],
            "maxima": local_maxima(coords_arr, vals, names_arr),
            "plan": hplan,
            "pollutants": mix_h, "causes": {n: attribute(mix_h[n], hod) for n in names_arr},
            "n_active": n_active,
            "avg": round(float(vals.mean()), 1),
        })
    json.dump({"generated_at": forecasts["generated_at"], "active_limit": Q4_ACTIVE,
               "rising_limit": Q4_RISING, "issued_hour": issued_hour, "hours": hours},
              open("december_hourly.json", "w"))
    peak_h = max(hours, key=lambda x: x["avg"]); low_h = min(hours, key=lambda x: x["avg"])
    print(f"december_hourly.json: 24 hourly snapshots "
          f"(worst {peak_h['label']} avg {peak_h['avg']} / {peak_h['plan']['total_trucks_needed']} trucks, "
          f"best {low_h['label']} avg {low_h['avg']} / {low_h['plan']['total_trucks_needed']} trucks)")

    # 6. SPRAY IMPACT: forecasted (no-spray) vs after the main formulation.
    # Per Final_Spray_Formulation_Comparison.docx, the main lignosulfonate
    # solution delivers ~35-50% PM2.5 reduction PER PASS (best-supported by
    # field trials); it also suppresses dust RESUSPENSION, so a treated
    # station stays lower for a few hours -> modelled as a deposited-film
    # suppression S that decays each hour and is refreshed when a truck
    # sprays that station (i.e. it's one of that hour's plan checkpoints).
    R_LO, R_HI = 0.35, 0.50        # PM2.5 one-pass reduction range (docx, main solution)
    FILM_DECAY = 0.65              # hourly erosion of the deposited film (resuspension)

    def mitigate(R):
        S = {n: 0.0 for n in names_arr}
        series, after_readings = [], []
        for h in range(24):
            for n in names_arr:
                S[n] *= FILM_DECAY                      # film erodes
            sprayed = {cp["station"] for c in hours[h]["plan"]["clusters"] for cp in c["checkpoints"]}
            for n in sprayed:
                S[n] = max(S[n], R)                     # fresh pass tops the film back up
            after = {n: round(hours[h]["readings"][n] * (1 - S[n]), 1) for n in names_arr}
            after_readings.append(after)
            series.append(round(sum(after.values()) / len(names_arr), 1))
        return series, after_readings

    baseline_series = [hours[h]["avg"] for h in range(24)]
    after_mid, after_readings = mitigate((R_LO + R_HI) / 2)
    after_best, _ = mitigate(R_HI)     # best-case reduction -> lowest curve
    after_worst, _ = mitigate(R_LO)    # worst-case reduction -> highest curve
    day_base = sum(baseline_series) / 24; day_after = sum(after_mid) / 24
    impact = {
        "formulation": "Main solution (lignosulfonate-based)",
        "source": "Final_Spray_Formulation_Comparison.docx",
        "pm25_reduction_pct": [int(R_LO * 100), int(R_HI * 100)],
        "labels": [f"{h:02d}:00" for h in range(24)],
        "baseline": baseline_series, "after": after_mid,
        "after_best": after_best, "after_worst": after_worst,
        "after_readings": after_readings,
        "active_limit": Q4_ACTIVE, "rising_limit": Q4_RISING,
        "day_avg_baseline": round(day_base, 1), "day_avg_after": round(day_after, 1),
        "day_avg_drop_pct": round((day_base - day_after) / day_base * 100, 1),
    }
    json.dump(impact, open("december_impact.json", "w"))
    print(f"december_impact.json: day-avg PM2.5 {day_base:.0f} -> {day_after:.0f} "
          f"({impact['day_avg_drop_pct']}% lower with the main spray)")


if __name__ == "__main__":
    main()
