// ============================================================================
//  Vayumitre operations dashboard -- one script for both modes:
//    body[data-mode="live"]     -> admin.html, live server APIs
//    body[data-mode="december"] -> december.html, synthetic December files
//  G zones are fixed for the month; P hotspot zones are re-found every hour
//  (live: every refresh) and carry their own standalone trucks.
// ============================================================================
const MODE = document.body.dataset.mode;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const short = n => String(n).split(",")[0];
const agency = n => { const p = String(n).split(" - "); return p.length > 1 ? p[p.length-1].trim() + " station" : "Monitoring station"; };
const fmt = (v, d=0) => v == null || isNaN(v) ? "–" : Number(v).toFixed(d);
const istTime = d => d.toLocaleTimeString("en-GB", {hour:"2-digit", minute:"2-digit", timeZone:"Asia/Kolkata"});
async function getJSON(u){
  const r = await fetch(u, {cache:"no-store"});
  const j = await r.json().catch(() => null);
  if(!r.ok || !j || j.error) throw new Error((j && j.error) || `${u}: HTTP ${r.status}`);
  return j;
}

const COL = {accent:"#1F6B48", amber:"#A85F00", red:"#B3261E", normal:"#7D8C84", ink:"#18211C", blue:"#1D5C8C"};
const TIER_COLOR = {NORMAL:COL.normal, PRE_POSITION:COL.amber, ACTIVE:COL.red};
const TIER_LABEL = {NORMAL:"Normal", PRE_POSITION:"Pre-position", ACTIVE:"Active"};
const THRESH = {1:174.2, 2:170.0, 3:95.5, 4:190.0};          // quarterly PM2.5 p90 limits
const QUARTER = Math.floor(new Date().getMonth()/3) + 1;
const LIM = MODE === "december"
  ? {active:190, rising:152, q:4}
  : {active:THRESH[QUARTER], rising:THRESH[QUARTER]*0.8, q:QUARTER};
const SPRAY_R = 0.425;                                         // mid of 35-50% per pass
const tier = pm => pm == null ? "NORMAL" : pm >= LIM.active ? "ACTIVE" : pm >= LIM.rising ? "PRE_POSITION" : "NORMAL";
const passesFor = pm => pm == null || pm < LIM.rising ? 0
  : Math.max(1, Math.ceil(Math.log(LIM.rising/pm) / Math.log(1 - SPRAY_R)));
const CAUSE_TAG = {
  dust:["#A85F00","#FBF1E2"], biomass:["#B3261E","#FBEAE8"], traffic:["#1D5C8C","#E6EEF5"],
  industry:["#6B3FA0","#EFE8F6"], photochem:["#1F6B48","#E5F0EA"], agri:["#5B6B1F","#EEF2DD"],
  regional:["#4E5A53","#EEF1ED"], none:["#67736C","#EEF1ED"],
};

const S = {
  stations:[], pm:{}, zones:null, regions:{}, boundary:null, plan:null, forecasts:null,
  causes:{}, pollutants:{}, impact:null, hourly:null, hour:8, after:false, sel:null,
  causeView:"zones", causeFilter:"all", source:"", updated:null, ready:false, run:null, playing:null,
};

// ---- plan / zone helpers ----------------------------------------------------
const entries = () => (S.plan && S.plan.clusters) || [];
const gEntries = () => entries().filter(e => e.kind !== "priority");
const pEntries = () => entries().filter(e => e.kind === "priority");
const entryOf = label => entries().find(e => e.label === label);
function zoneStations(label){
  const e = entryOf(label);
  if(e && e.kind === "priority") return e.checkpoints.map(c => c.station);
  const c = S.zones && S.zones.clusters.find(z => `G${z.id}` === label);
  return c ? c.stations.slice() : [];
}
const peakOf = names => names.reduce((m, n) => S.pm[n] != null && S.pm[n] > m ? S.pm[n] : m, -Infinity);
const coordOf = name => { const s = S.stations.find(x => x.station === name)
  || (S.zones && (S.zones.stations||[]).find(x => x.station === name)); return s ? [s.lat, s.lng] : null; };
function centroid(names){
  const pts = names.map(coordOf).filter(Boolean);
  if(!pts.length) return null;
  return [pts.reduce((a,p) => a+p[0], 0)/pts.length, pts.reduce((a,p) => a+p[1], 0)/pts.length];
}
const selectedG = () => { const e = entryOf(S.sel); return e && e.kind === "priority" ? e.parent : S.sel; };
function defaultSelection(){
  const g = gEntries();
  if(!g.length) return null;
  return g.map(e => [e.label, peakOf(zoneStations(e.label))]).sort((a,b) => b[1]-a[1])[0][0];
}
function select(label){ S.sel = label; renderMap(); renderZones(); renderHotspots(); renderSources(); renderCauses(); }

// ---- map --------------------------------------------------------------------
const map = L.map("map", {zoomControl:true, attributionControl:false, scrollWheelZoom:false, zoomSnap:0.25}).setView([28.62, 77.12], 10);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
  {maxZoom:16, opacity:0.5}).addTo(map);
const LY = {};
["zones","route","stations","rings","labels","trucks"].forEach(k => LY[k] = L.layerGroup().addTo(map));
let fitted = false;

function renderMap(){
  Object.values(LY).forEach(l => { if(l !== LY.trucks) l.clearLayers(); });
  const selG = selectedG();
  if(S.boundary && !fitted){ map.invalidateSize(); map.fitBounds(L.polygon(S.boundary).getBounds(), {padding:[6,6]}); fitted = true; }
  // territories (selected one last, so its outline is on top)
  const regs = Object.entries(S.regions || {}).sort((a,b) => (a[0]===selG) - (b[0]===selG));
  regs.forEach(([label, reg], i) => {
    const sel = label === selG, n = parseInt(label.slice(1)) || 0;
    (reg.rings || []).forEach(r => L.polygon(r, {
      color: sel ? COL.accent : "#FFFFFF", weight:2, fillColor: sel ? "#DCEBE2" : (n % 2 ? "#E4E9E5" : "#EBEFEB"),
      fillOpacity:0.92, lineJoin:"round",
    }).on("click", () => select(label)).bindTooltip(`${label} · ~${reg.area_km2} km²`, {sticky:true}).addTo(LY.zones));
  });
  if(S.boundary) L.polygon(S.boundary, {color:"#C9D1CB", weight:1, fill:false, interactive:false}).addTo(LY.zones);
  // route of the selected zone
  const se = entryOf(S.sel);
  if(se && se.geometry && se.geometry.length > 1)
    L.polyline(se.geometry, {color: se.kind === "priority" ? COL.red : COL.accent, weight:1.8, dashArray:"5 4", interactive:false}).addTo(LY.route);
  // stations
  S.stations.forEach(s => {
    if(s.lat == null) return;
    const t = tier(s.pm25);
    L.circleMarker([s.lat, s.lng], {radius:5, color:"#FFFFFF", weight:1.5, fillColor:TIER_COLOR[t], fillOpacity:1})
      .bindTooltip(`<b>${esc(short(s.station))}</b><br>PM2.5 ${fmt(s.pm25)} · ${TIER_LABEL[t]}`)
      .on("click", () => openStation(s.station)).addTo(LY.stations);
  });
  // P hotspot rings
  pEntries().forEach(e => e.checkpoints.forEach(c => {
    L.circleMarker([c.lat, c.lng], {radius:10, color:COL.red, weight: e.label === S.sel ? 2.5 : 1.5, fill:false})
      .bindTooltip(`${e.label} hotspot · inside ${e.parent}`).on("click", () => select(e.label)).addTo(LY.rings);
  }));
  // zone labels
  gEntries().forEach(e => {
    const c = centroid(zoneStations(e.label));
    if(!c) return;
    L.marker(c, {interactive:false, icon:L.divIcon({className:"", iconSize:[30,18], iconAnchor:[15,9],
      html:`<div class="zlabel${e.label === selG ? " sel" : ""}">${e.label}</div>`})}).addTo(LY.labels);
  });
}

// ---- header, KPIs ------------------------------------------------------------
function renderHeader(){
  if(MODE === "december"){
    $("st-dot").style.background = COL.amber;
    $("st-src").textContent = "Scenario";
    $("st-time").textContent = `Hour ${String(S.hour).padStart(2,"0")}:00 IST`;
    $("subtitle").textContent = `Winter-smog day · active limit ${fmt(LIM.active)} µg/m³ · pre-position limit ${fmt(LIM.rising)} µg/m³`;
  } else {
    const cpcb = /CPCB/.test(S.source);
    $("st-dot").style.background = !S.ready ? COL.normal : cpcb ? COL.accent : COL.amber;
    $("st-src").textContent = !S.ready ? "Loading…" : "Live";
    $("st-time").textContent = S.updated ? `Updated ${istTime(S.updated)} IST` : "–";
    $("subtitle").textContent = `Q${LIM.q} · active limit ${fmt(LIM.active,1)} µg/m³ · pre-position limit ${fmt(LIM.rising,1)} µg/m³`;
  }
  const f = S.plan && S.plan.fleet;
  $("gz-sub").textContent = `G · set for the month · ${f ? f.per_zone : 5} trucks each`;
}

function renderKPIs(){
  const vals = S.stations.map(s => s.pm25).filter(v => v != null);
  const avg = vals.length ? vals.reduce((a,b) => a+b, 0)/vals.length : null;
  const above = vals.filter(v => v >= LIM.active).length;
  const f = S.plan && S.plan.fleet;
  const nG = gEntries().length, nP = pEntries().length;
  const tile = (l, v, color) => `<div class="kpi"><div class="l">${l}</div><div class="v"${color ? ` style="color:${color}"` : ""}>${v}</div></div>`;
  $("kpis").innerHTML =
    tile("Stations", vals.length) +
    tile("Above active limit", above, above ? COL.red : null) +
    tile("City avg PM2.5", `${fmt(avg)} <small>µg/m³</small>`) +
    tile("Zones", `${nG} G <span style="color:#C9D1CB">·</span> <span style="color:${COL.red}">${nP} P</span>`) +
    tile("Trucks deployed", f ? f.deployed : "–") +
    tile("Short of demand", f ? f.short : "–", f && f.short ? COL.amber : null);
}

// ---- side panels ----------------------------------------------------------------
function renderZones(){
  const rows = gEntries().slice().sort((a,b) => parseInt(a.label.slice(1)) - parseInt(b.label.slice(1)));
  const selG = selectedG();
  $("gz").innerHTML = rows.map(e => {
    const names = zoneStations(e.label), peak = peakOf(names);
    const a = e.trucks_assigned ?? e.trucks_needed, need = e.trucks_needed;
    return `<tr class="${e.label === selG ? "sel" : ""}" data-l="${e.label}">
      <td class="z">${e.label}</td><td class="r">${names.length}</td>
      <td class="r">${isFinite(peak) ? fmt(peak,1) : "–"}</td>
      <td class="r" style="color:${need > a ? COL.amber : COL.ink}">${a} / ${need}</td>
      <td class="r km">${e.distance_km != null ? fmt(e.distance_km,1) + " km" : "—"}</td></tr>`;
  }).join("") || `<tr><td colspan="5" class="empty">No zones yet.</td></tr>`;
  $("gz").querySelectorAll("tr[data-l]").forEach(tr => tr.onclick = () => select(tr.dataset.l));
}

function renderHotspots(){
  const ps = pEntries().slice().sort((a,b) => peakOf(zoneStations(b.label)) - peakOf(zoneStations(a.label)));
  $("pz-sub").textContent = MODE === "december" ? "P · this hour · own trucks" : "P · this refresh · own trucks";
  $("pz").innerHTML = ps.map(e => {
    const n = e.trucks_assigned ?? e.trucks_needed;
    return `<div class="prow${e.label === S.sel ? " sel" : ""}" data-l="${e.label}">
      <span class="pid">${e.label}</span>
      <div class="w"><b>${esc(e.checkpoints.map(c => short(c.station)).join(" + "))}</b><span>inside ${e.parent}</span></div>
      <span class="pk">${fmt(peakOf(zoneStations(e.label)),1)}</span>
      <span class="tr">${n} truck${n === 1 ? "" : "s"}</span></div>`;
  }).join("") || `<div class="empty">No hotspot zones right now.</div>`;
  $("pz").querySelectorAll(".prow").forEach(r => r.onclick = () => select(r.dataset.l));
}

// ---- likely sources (rule-based cause of the selected zone) -----------------------
function causeBars(cause){
  if(!cause) return `<div class="muted">Not available yet.</div>`;
  if(!cause.causes.length) return `<div class="muted">No pollution rise to attribute.</div>`;
  return cause.causes.map(c => {
    const helps = /^(yes|partly)/.test(c.spray_helps);
    return `<div class="src"><div class="h"><b>${esc(c.short || c.label)}</b><span>${esc(c.evidence[0] || "")}</span><em>${c.strength}</em></div>
      <div class="bar"><i style="width:${c.strength}%;background:${helps ? COL.accent : COL.normal}"></i></div>
      <span class="helps">Spray helps: ${esc(c.spray_helps)}</span></div>`;
  }).join("");
}
function renderSources(){
  const e = entryOf(S.sel);
  let sub = "";
  if(e){
    const names = zoneStations(e.label);
    const top = names.slice().sort((a,b) => (S.pm[b]||0) - (S.pm[a]||0))[0];
    sub = e.kind === "priority" ? `Hotspot ${e.label} in ${e.parent} · ${short(top)}` : `Zone ${e.label} · ${short(top || "")} area`;
  }
  const miss = e && e.cause && e.cause.missing && e.cause.missing.length
    ? "" : "";
  $("sources-card").innerHTML = `
    <div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap"><h2>Likely sources</h2><span class="hsub">${esc(sub)}</span></div>
    <div class="sources" style="margin-top:14px">${e ? causeBars(e.cause) : `<div class="muted">Select a zone.</div>`}</div>
    ${miss}`;
}

// ---- spray impact chart ------------------------------------------------------------
function impactSeries(){
  if(MODE === "december" && S.impact)
    return {base:S.impact.baseline, after:S.impact.after, labels:["00:00","06:00","12:00","18:00","23:00"], cursor:S.hour,
      note:`Dashed line: active limit ${fmt(LIM.active)}. Day average ${fmt(S.impact.day_avg_baseline)} → ${fmt(S.impact.day_avg_after)} µg/m³ (−${S.impact.day_avg_drop_pct}%).`,
      sub:"City average PM2.5 over 24 h, µg/m³"};
  if(!S.forecasts || !S.forecasts.stations) return null;
  const sprayed = new Set(entries().flatMap(e => e.checkpoints.map(c => c.station)));
  const base = [], after = [];
  for(let k = 0; k < 24; k++){
    let b = 0, a = 0, n = 0;
    S.forecasts.stations.forEach(f => {
      const v = k === 0 ? f.now : (f.curve_v || [])[k-1];
      if(v == null) return;
      b += v; a += sprayed.has(f.name) ? v*(1-SPRAY_R) : v; n++;
    });
    if(n){ base.push(b/n); after.push(a/n); }
  }
  if(base.length < 2) return null;
  return {base, after, labels:["now","+6 h","+12 h","+18 h","+23 h"], cursor:null,
    note:`Dashed line: active limit ${fmt(LIM.active)}.`,
    sub:"City average PM2.5, next 24 h, µg/m³"};
}
function renderImpact(){
  const d = impactSeries();
  const head = `<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px"><h2>Spray impact</h2>
    <span class="hsub">${d ? d.sub : ""}</span>
    <div class="legend"><span><i class="ln" style="background:${COL.normal}"></i>Forecast</span><span><i class="ln" style="background:${COL.accent}"></i>After spray</span></div></div>`;
  if(!d){ $("impact-card").innerHTML = head + `<p class="note">Waiting for the forecast…</p>`; return; }
  const top = Math.ceil(Math.max(LIM.active, ...d.base, ...d.after) * 1.1 / 50) * 50;
  const W = 600, H = 180, n = d.base.length;
  const x = i => (i/(n-1)*W).toFixed(1), y = v => (H - v/top*H).toFixed(1);
  const pts = arr => arr.map((v,i) => `${x(i)},${y(v)}`).join(" ");
  const cur = d.cursor == null ? "" : `<line x1="${x(d.cursor)}" y1="0" x2="${x(d.cursor)}" y2="${H}" stroke="#18211C" stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke" opacity="0.5"/>`;
  $("impact-card").innerHTML = head + `
    <div class="chart"><div class="yax"><span>${top}</span><span>${top/2}</span><span>0</span></div>
      <div style="flex:1 1 auto;min-width:0">
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Forecast vs after spray" style="width:100%;height:180px;display:block">
          <line x1="0" y1="0.5" x2="${W}" y2="0.5" stroke="#EEF1ED"/><line x1="0" y1="${H/2}" x2="${W}" y2="${H/2}" stroke="#EEF1ED"/>
          <line x1="0" y1="${H-0.5}" x2="${W}" y2="${H-0.5}" stroke="#E2E6E1"/>
          <line x1="0" y1="${y(LIM.active)}" x2="${W}" y2="${y(LIM.active)}" stroke="${COL.red}" stroke-dasharray="3 4" stroke-width="1" vector-effect="non-scaling-stroke"/>
          ${cur}
          <polyline fill="none" stroke="${COL.normal}" stroke-width="2" vector-effect="non-scaling-stroke" points="${pts(d.base)}"/>
          <polyline fill="none" stroke="${COL.accent}" stroke-width="2" vector-effect="non-scaling-stroke" points="${pts(d.after)}"/>
        </svg>
        <div class="xax">${d.labels.map(l => `<span>${l}</span>`).join("")}</div>
      </div></div>
    <p class="note">${d.note}</p>`;
}

// ---- causes by area ------------------------------------------------------------------
function renderCauses(){
  let items = [];
  if(S.causeView === "zones"){
    items = entries().map(e => ({key:e.label, name:`${e.label} · ${e.kind === "priority" ? "hotspot in " + e.parent : "G zone"}`,
      sub:(n => `${n} station${n === 1 ? "" : "s"}`)(zoneStations(e.label).length), cause:e.cause, pm:peakOf(zoneStations(e.label)), open:() => select(e.label)}));
  } else {
    items = Object.entries(S.causes || {}).map(([n, c]) => ({key:n, name:short(n), cause:c, pm:S.pm[n], open:() => openStation(n)}));
  }
  const top = it => it.cause && it.cause.causes && it.cause.causes[0];
  const k = it => top(it) ? top(it).key : "none";
  const counts = {};
  items.forEach(it => counts[k(it)] = (counts[k(it)] || 0) + 1);
  const label = key => key === "none" ? "No clear cause" : (items.map(top).find(t => t && t.key === key) || {}).short || key;
  if(S.causeFilter !== "all" && !counts[S.causeFilter]) S.causeFilter = "all";
  const shown = items.filter(it => S.causeFilter === "all" || k(it) === S.causeFilter).sort((a,b) => (b.pm||0) - (a.pm||0));
  $("causes-card").innerHTML = `
    <div class="card-h" style="align-items:center"><h2>Causes by area</h2><span class="sub">top likely source per ${S.causeView === "zones" ? "zone" : "station"}</span>
      <div class="seg"><button type="button" data-v="zones" class="${S.causeView === "zones" ? "on" : ""}">Zones</button><button type="button" data-v="stations" class="${S.causeView === "stations" ? "on" : ""}">Stations</button></div></div>
    <div class="chips"><button type="button" class="chip${S.causeFilter === "all" ? " on" : ""}" data-f="all">All ${items.length}</button>
      ${Object.entries(counts).sort((a,b) => b[1]-a[1]).map(([key, n]) => `<button type="button" class="chip${S.causeFilter === key ? " on" : ""}" data-f="${key}">${esc(label(key))} ${n}</button>`).join("")}</div>
    <div class="causes-list">${shown.map((it, i) => {
      const t = top(it), [fg, bg] = CAUSE_TAG[t ? t.key : "none"] || CAUSE_TAG.none;
      return `<div class="crow" data-i="${i}"><span class="nm">${esc(it.name)}${it.sub ? ` <span>· ${esc(it.sub)}</span>` : ""}</span>
        <span class="pv" style="color:${TIER_COLOR[tier(it.pm)]}">${fmt(it.pm)}</span>
        <span class="tag" style="color:${fg};background:${bg}">${t ? esc(t.short) + " " + t.strength : "No clear cause"}</span>
        <span class="sp">${t ? "spray: " + esc(t.spray_helps.split(" ")[0]) : "–"}</span></div>`;
    }).join("") || `<div class="empty">${S.causeView === "stations" && !Object.keys(S.causes||{}).length ? "Loading…" : "Nothing for this filter."}</div>`}</div>`;
  $("causes-card").querySelectorAll(".seg button").forEach(b => b.onclick = () => { S.causeView = b.dataset.v; S.causeFilter = "all"; renderCauses(); });
  $("causes-card").querySelectorAll(".chip").forEach(b => b.onclick = () => { S.causeFilter = b.dataset.f; renderCauses(); });
  $("causes-card").querySelectorAll(".crow").forEach(r => r.onclick = () => shown[+r.dataset.i].open());
}

// ---- station detail modal ----------------------------------------------------------------
function curveSVG(vals, limit){
  if(vals.length < 2) return "";
  const W = 520, H = 130, top = Math.ceil(Math.max(limit, ...vals) * 1.1 / 50) * 50;
  const x = i => (i/(vals.length-1)*W).toFixed(1), y = v => (H - v/top*H).toFixed(1);
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:130px;display:block" role="img" aria-label="Forecast curve">
    <line x1="0" y1="${H-0.5}" x2="${W}" y2="${H-0.5}" stroke="#E2E6E1"/>
    <line x1="0" y1="${y(limit)}" x2="${W}" y2="${y(limit)}" stroke="${COL.red}" stroke-dasharray="3 4" vector-effect="non-scaling-stroke"/>
    <polyline fill="none" stroke="${COL.accent}" stroke-width="2" vector-effect="non-scaling-stroke" points="${vals.map((v,i) => `${x(i)},${y(v)}`).join(" ")}"/>
    <circle cx="0" cy="${y(vals[0])}" r="4" fill="${COL.ink}"/></svg>
    <div class="xax"><span>now</span><span>+6 h</span><span>+12 h</span><span>+18 h</span><span>+${vals.length-1} h</span></div>`;
}
function openStation(name){
  const pm = S.pm[name], t = tier(pm);
  const f = S.forecasts && (S.forecasts.stations || []).find(s => s.name === name);
  const mix = S.pollutants && S.pollutants[name];
  const POL = {"PM2.5":"PM2.5","PM10":"PM10","NO2":"NO₂","SO2":"SO₂","CO":"CO","OZONE":"O₃","NH3":"NH₃"};
  const mixTxt = mix ? Object.keys(POL).filter(k => mix[k] != null).map(k => `${POL[k]} <b class="mono">${k === "CO" ? fmt(mix[k],1) : fmt(mix[k])}</b>`).join(" · ") : "";
  const zone = (entries().find(e => e.checkpoints.some(c => c.station === name)) || {}).label
    || (S.zones && (S.zones.clusters.find(c => c.stations.includes(name)) || {}).id != null ? "G" + S.zones.clusters.find(c => c.stations.includes(name)).id : "–");
  const hz = f ? [1,3,6,12,24].map(h => { const v = f.h[String(h)];
    return `<div><div class="l">+${h} h</div><div class="v" style="color:${v >= LIM.active ? COL.red : COL.ink}">${fmt(v)}</div></div>`; }).join("") : "";
  $("modal").innerHTML = `
    <div class="modal-h"><div><h2>${esc(short(name))}</h2><p>${esc(agency(name))} · zone ${esc(zone)} · ${TIER_LABEL[t]}</p></div>
      <button type="button" aria-label="Close" id="m-close">×</button></div>
    <div class="modal-b">
      <div class="mkpis">
        <div><div class="l">PM2.5 now</div><div class="v" style="color:${TIER_COLOR[t]}">${fmt(pm,1)}</div></div>
        <div><div class="l">Peak next 24 h</div><div class="v">${f ? fmt(f.peak) + ` <small class="muted">+${f.peak_h} h</small>` : "–"}</div></div>
        <div><div class="l">Spray passes</div><div class="v">${passesFor(pm)}</div></div>
      </div>
      ${f ? `<div><div class="sect" style="margin-bottom:8px">Forecast</div>
        <div class="hz">${hz}</div><div style="margin-top:12px">${curveSVG([f.now, ...(f.curve_v || [])], LIM.active)}</div></div>`
        : `<p class="muted">No forecast available for this station.</p>`}
      <div><div class="sect">Likely cause of pollution</div>${mixTxt ? `<p class="muted" style="margin:2px 0 10px">${mixTxt}</p>` : ""}
        <div class="sources">${causeBars(S.causes && S.causes[name])}</div></div>
    </div>`;
  $("overlay").classList.add("show");
  $("m-close").onclick = closeModal;
}
function closeModal(){ $("overlay").classList.remove("show"); }
$("overlay").addEventListener("click", e => { if(e.target.id === "overlay") closeModal(); });
document.addEventListener("keydown", e => { if(e.key === "Escape") closeModal(); });

// ---- render all -----------------------------------------------------------------------------
function renderAll(){
  S.pm = Object.fromEntries(S.stations.map(s => [s.station, s.pm25]));
  if(!entryOf(S.sel)) S.sel = defaultSelection();   // e.g. a P hotspot that is gone this hour
  renderHeader(); renderKPIs(); renderMap(); renderZones(); renderHotspots();
  renderImpact(); renderSources(); renderCauses();
}

// ---- export / dispatch buttons ------------------------------------------------------------
$("btn-export").onclick = () => {
  if(!S.plan) return;
  const blob = new Blob([JSON.stringify({mode:MODE, hour:MODE === "december" ? S.hour : null,
    exported_at:new Date().toISOString(), limits:LIM, plan:S.plan}, null, 1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `vayumitre-plan-${MODE}${MODE === "december" ? "-" + String(S.hour).padStart(2,"0") + "h" : ""}.json`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
$("btn-dispatch").onclick = () => {
  if(MODE === "december") startRun();
  else location.href = `driver.html?src=live&zone=${encodeURIComponent(S.sel || "")}`;
};

// ============================================================================
//  LIVE mode
// ============================================================================
async function loadLive(){
  try{ const z = await getJSON("/api/zones"); S.zones = z; S.regions = z.regions || {}; }
  catch(e){
    try{ S.zones = await getJSON("data/clusters_v2.json"); S.regions = (await getJSON("data/cluster_regions.json")).regions || {}; }catch(_){}
  }
  try{ S.plan = await getJSON("/api/plan"); }catch(e){ S.plan = null; }
  try{
    const f = await getJSON("/api/forecasts");
    S.forecasts = f; S.stations = f.stations_now || []; S.causes = f.causes || {}; S.pollutants = f.pollutants_now || {};
    S.source = f.air_source || ""; S.updated = new Date(f.generated_at); S.ready = true;
  }catch(e){
    S.ready = false;
    if(!S.stations.length && S.zones) S.stations = (S.zones.stations || []).map(s => ({...s}));
  }
}
function liveFooter(){
  $("mapfoot").innerHTML = `
    <button type="button" class="iconbtn" id="refresh" aria-label="Refresh now">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#18211C" stroke-width="1.6" stroke-linecap="round"><path d="M13 8a5 5 0 1 1-1.5-3.6"/><path d="M13 2.5v3h-3"/></svg></button>
    <span style="font-size:12px;color:#67736C">${S.ready ? "Live readings" : "Loading…"}</span>
    <span class="clock" style="margin-left:auto">${S.updated ? istTime(S.updated) : "–"}</span>`;
  $("refresh").onclick = refreshLive;
}
async function refreshLive(){ await loadLive(); liveFooter(); renderAll(); }

// ============================================================================
//  DECEMBER mode -- hourly time-lapse, after-spray view, phased dispatch run
// ============================================================================
async function loadDecember(){
  const [cl, rg, fc, hr, im] = await Promise.all([
    getJSON("data/december_clusters.json"), getJSON("data/december_regions.json"),
    getJSON("data/december_forecasts.json"), getJSON("data/december_hourly.json"),
    getJSON("data/december_impact.json").catch(() => null)]);
  S.zones = cl; S.regions = rg.regions || {}; S.forecasts = fc; S.hourly = hr; S.impact = im;
  S.hour = hr.issued_hour ?? 8;
}
function applyHour(h){
  S.hour = ((h % 24) + 24) % 24;
  const snap = S.hourly.hours[S.hour];
  S.plan = snap.plan; S.causes = snap.causes || {}; S.pollutants = snap.pollutants || {};
  const after = S.after && S.impact && S.impact.after_readings ? S.impact.after_readings[S.hour] : null;
  S.stations = snap.stations_now.map(s => ({...s, pm25: after ? after[s.station] : s.pm25}));
  decFooter(); renderAll();
}
function decFooter(){
  const playIcon = S.playing
    ? `<svg width="14" height="14" viewBox="0 0 14 14" fill="#18211C"><rect x="3" y="2" width="3" height="10" rx="0.8"/><rect x="8" y="2" width="3" height="10" rx="0.8"/></svg>`
    : `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="#18211C" stroke-width="1.6" stroke-linejoin="round"><path d="M3.5 2 L11.5 7 L3.5 12 Z"/></svg>`;
  if(!$("hour")){
    $("mapfoot").innerHTML = `
      <button type="button" class="iconbtn" id="play" aria-label="Play time-lapse"></button>
      <label for="hour">Hour</label>
      <input id="hour" type="range" min="0" max="23" step="1">
      <span class="clock" id="clock"></span>
      <label class="toggle"><input type="checkbox" id="after">After spray</label>`;
    $("hour").oninput = () => { const h = +$("hour").value; stopRun(); pause(); applyHour(h); };
    $("play").onclick = () => { stopRun(); S.playing ? pause() : play(); };
    $("after").onchange = () => { stopRun(); S.after = $("after").checked; applyHour(S.hour); };
  }
  $("play").innerHTML = playIcon;
  $("play").setAttribute("aria-label", S.playing ? "Pause time-lapse" : "Play time-lapse");
  $("hour").value = S.hour;
  $("clock").textContent = `${String(S.hour).padStart(2,"0")}:00`;
}
function play(){ S.playing = setInterval(() => applyHour(S.hour + 1), 900); decFooter(); }
function pause(){ if(S.playing){ clearInterval(S.playing); S.playing = null; } if($("play")) decFooter(); }

// ---- phased dispatch run: trucks drive, spray (-42.5%/pass), AQI is re-read,
//      the next phase goes only to what is still above the pre-position limit ----
const RUN_STATES = ["PENDING","ASSIGNED","EN_ROUTE","SPRAYING","COMPLETE"];
const RUN_ABBR = {PENDING:"Pending", ASSIGNED:"Assigned", EN_ROUTE:"En route", SPRAYING:"Spraying", COMPLETE:"Done"};
function startRun(){
  pause(); stopRun();
  const lvl = {...S.pm};
  const inP = new Set(pEntries().flatMap(e => e.checkpoints.map(c => c.station)));
  const items = {}, trucks = [];
  entries().forEach(e => {
    const pool = e.kind === "priority" ? zoneStations(e.label) : zoneStations(e.label).filter(n => !inP.has(n));
    pool.forEach(n => items[n] = {zone:e.label, state: lvl[n] >= LIM.rising ? "PENDING" : "COMPLETE"});
    const home = centroid(pool) || [28.62, 77.12];
    const n = Math.min(12, Math.max(1, e.trucks_assigned ?? e.trucks_needed ?? 1));
    for(let i = 0; i < n; i++) trucks.push({zone:e.label, pool, home, pos:home.slice(), target:null, name:null, state:"idle", dwell:0});
  });
  S.run = {lvl, items, trucks, phase:0, history:[], log:[], timer:null, next:null, done:false, startAvg:avgOf(lvl)};
  $("run-card").hidden = false;
  $("btn-dispatch").disabled = true; $("btn-dispatch").textContent = "Dispatching…";
  runPhase();
}
const avgOf = lvl => { const v = Object.values(lvl).filter(x => x != null); return v.reduce((a,b) => a+b, 0) / Math.max(1, v.length); };
function stopRun(){
  if(!S.run) return;
  clearInterval(S.run.timer); clearTimeout(S.run.next);
  LY.trucks.clearLayers(); S.run = null;
  $("run-card").hidden = true;
  $("btn-dispatch").disabled = false; $("btn-dispatch").textContent = "Dispatch trucks";
}
function logRun(name, to, why){
  const R = S.run, it = R.items[name];
  if(!it || it.state === to) return;
  it.state = to;
  R.log.unshift({t:`P${R.phase}`, name, zone:it.zone, to, why});
  if(R.log.length > 200) R.log.pop();
}
function runPhase(){
  const R = S.run; R.phase++;
  const taken = new Set();
  R.trucks.forEach(t => {
    const pick = t.pool.filter(n => R.lvl[n] >= LIM.rising && !taken.has(n)).sort((a,b) => R.lvl[b] - R.lvl[a])[0];
    if(pick){
      taken.add(pick); t.name = pick; t.target = coordOf(pick); t.state = "moving";
      logRun(pick, "ASSIGNED", `worst in ${t.zone} (PM2.5 ${fmt(R.lvl[pick])})`);
      logRun(pick, "EN_ROUTE", "truck left the depot");
    } else { t.name = null; t.target = t.home; t.state = "home"; }
  });
  const supply = R.trucks.filter(t => t.name).length;
  const demand = Object.values(R.lvl).filter(v => v >= LIM.rising).length;
  R.timer = setInterval(() => {
    let busy = 0;
    R.trucks.forEach(t => {
      if(!t.target) return;
      if(t.state === "spraying"){
        busy++;
        if(--t.dwell <= 0){
          const before = R.lvl[t.name], after = Math.round(before * (1 - SPRAY_R) * 10) / 10;
          R.lvl[t.name] = after;
          if(after < LIM.rising) logRun(t.name, "COMPLETE", `${fmt(before)} → ${fmt(after)}, below ${fmt(LIM.rising)}`);
          else logRun(t.name, "PENDING", `${fmt(before)} → ${fmt(after)}, needs another pass`);
          t.state = "sprayed";
        }
        return;
      }
      const dLa = t.target[0] - t.pos[0], dLo = t.target[1] - t.pos[1];
      if(Math.hypot(dLa, dLo) > 0.003){ busy++; t.pos = [t.pos[0] + dLa*0.14, t.pos[1] + dLo*0.14]; }
      else if(t.state === "moving"){ logRun(t.name, "SPRAYING", "arrived, spraying"); t.state = "spraying"; t.dwell = 6; busy++; }
    });
    drawTrucks();
    if(busy === 0){ clearInterval(R.timer); finishPhase(supply, demand); }
  }, 110);
  renderRun();
}
function drawTrucks(){
  LY.trucks.clearLayers();
  if(!S.run) return;
  S.run.trucks.forEach(t => L.marker(t.pos, {interactive:false, icon:L.divIcon({className:"", iconSize:[18,18], iconAnchor:[9,9],
    html:`<div class="truck">${t.state === "spraying" || t.state === "sprayed" ? "💧" : "🚛"}</div>`})}).addTo(LY.trucks));
}
function finishPhase(supply, demand){
  const R = S.run;
  R.history.push({phase:R.phase, supply, demand, avg:avgOf(R.lvl)});
  S.stations = S.stations.map(s => ({...s, pm25: R.lvl[s.station] ?? s.pm25}));
  S.pm = Object.fromEntries(S.stations.map(s => [s.station, s.pm25]));
  renderKPIs(); renderMap(); renderZones(); renderHotspots(); renderRun();
  const left = Object.values(R.lvl).filter(v => v >= LIM.rising).length;
  if(left > 0 && R.phase < 8) R.next = setTimeout(runPhase, 650);
  else {
    R.done = true;
    R.trucks.forEach(t => { t.target = t.home; t.state = "home"; t.pos = t.home.slice(); });
    drawTrucks(); renderRun();
    $("btn-dispatch").disabled = false; $("btn-dispatch").textContent = "Dispatch trucks";
  }
}
function renderRun(){
  const R = S.run; if(!R) return;
  const h = R.history, last = h[h.length-1];
  const byZone = {};
  Object.values(R.items).forEach(it => { const b = byZone[it.zone] ||= Object.fromEntries(RUN_STATES.map(s => [s,0])); b[it.state]++; });
  const W = 400, H = 90, n = Math.max(h.length, 2), maxSD = Math.max(4, ...h.map(d => Math.max(d.supply, d.demand)));
  const x = i => (i/(n-1)*W).toFixed(1), y = v => (H - v/maxSD*H).toFixed(1);
  const line = (k, c) => h.length ? `<polyline fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke" points="${h.map((d,i) => `${x(i)},${y(d[k])}`).join(" ")}"/>` : "";
  $("run-card").innerHTML = `
    <div class="card-h"><h2>Dispatch run</h2><span class="sub">${R.done ? "finished" : "phase " + R.phase} · avg ${fmt(R.startAvg)} → ${fmt(last ? last.avg : R.startAvg)} µg/m³</span>
      <button type="button" class="btn" id="run-stop" style="margin-left:auto;min-height:32px;padding:0 12px">${R.done ? "Close" : "Stop"}</button></div>
    <div style="padding:14px 18px 6px">
      <div class="legend" style="margin:0 0 8px"><span><i class="ln" style="background:${COL.amber}"></i>Stations needing spray</span><span><i class="ln" style="background:${COL.blue}"></i>Trucks out</span></div>
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:90px;display:block" role="img" aria-label="Supply and demand by phase">
        <line x1="0" y1="${H-0.5}" x2="${W}" y2="${H-0.5}" stroke="#E2E6E1"/>${line("demand", COL.amber)}${line("supply", COL.blue)}</svg>
      <div class="xax">${h.map(d => `<span>P${d.phase}</span>`).join("") || "<span>phase 1 running…</span>"}</div>
    </div>
    <div class="states">${Object.keys(byZone).sort().map(z => `<div class="srow"><span class="lab" style="color:${z[0] === "P" ? COL.red : COL.ink}">${z}</span>
      ${RUN_STATES.map(s => `<span class="st ${s}${byZone[z][s] ? "" : " z"}">${RUN_ABBR[s]} ${byZone[z][s]}</span>`).join(" ")}</div>`).join("")}</div>
    <div class="card-h" style="border-top:1px solid #E2E6E1"><h2 style="font-size:13px">Change history</h2><span class="sub">${R.log.length} events</span></div>
    <div class="log">${R.log.slice(0, 40).map(e => `<div class="li"><span class="t">${e.t}</span><span><b>${esc(e.zone)}</b> ${esc(short(e.name))} → ${RUN_ABBR[e.to]}<br><span class="why">${esc(e.why)}</span></span></div>`).join("")}</div>`;
  $("run-stop").onclick = () => { stopRun(); applyHour(S.hour); };
}

// ============================================================================
//  boot
// ============================================================================
(async function init(){
  try{ S.boundary = (await getJSON("data/delhi_boundary.json")).boundary; }catch(_){}
  if(MODE === "december"){
    await loadDecember();
    applyHour(S.hour);
  } else {
    await loadLive();
    liveFooter(); renderAll();
    // fast polling until the server's first build lands, then every 5 minutes
    let tick = 0;
    setInterval(() => { tick++; if(!S.ready || tick % 15 === 0) refreshLive(); }, 20000);
  }
})();
const refit = () => { map.invalidateSize(); if(S.boundary) map.fitBounds(L.polygon(S.boundary).getBounds(), {padding:[6,6]}); };
setTimeout(refit, 250);
window.addEventListener("resize", refit);
