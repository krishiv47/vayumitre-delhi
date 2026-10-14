"""
Apply the trained PM2.5 models to live stations, and find the coming spike.

``forecast.py`` trains and validates the XGBoost models offline.  This module is
the deployment half: it assembles the recent history each model needs for every
station on the map, runs the models forward, and reports where and when the next
peak is expected.

Anchoring: measured level, modelled trajectory
----------------------------------------------
Forecasts are produced **per CPCB monitor**, because the
monitor is where the measurement actually happens.

The models need 168 hours of lagged and rolling PM2.5 to compute their features,
and CPCB's real-time resource publishes **current values only** -- no history
endpoint.  History therefore comes from Open-Meteo's hourly archive.  That grid
is coarse (roughly 40 km over India), so all of Delhi's monitors fall into only
two or three cells: it can say *how the air will move*, but not *how each
neighbourhood differs*.

The two sources are combined along the axis each is good at::

    forecast(station, h) = CPCB_measured_now(station) + modelled_delta(h)

The **level** is each monitor's own ground measurement -- which is why Anand
Vihar and Dwarka Mor give different answers -- while the **trajectory** is the
regional model's.  This works because the trained models predict a *change* from
the current value rather than an absolute concentration, so the anchor is exactly
what they were built to be added to.  The UI labels both halves.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from .net import HTTP_TIMEOUT_S, LiveDataError, get_json as _get_json

IST = timezone(timedelta(hours=5, minutes=30))

AIR_ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# Horizons the trained models cover (must match forecast.HORIZONS).
HORIZONS = [1, 3, 6, 12, 24]

# Open-Meteo caps how many coordinates one request will take comfortably.
BATCH = 25


@dataclass
class StationForecast:
    """Model output plus the raw forward curve for one station."""

    node_id: str
    name: str
    lat: float
    lng: float
    issued_at: datetime
    pm25_now: float                  # the anchor: this monitor's measured value
    horizons: dict[int, float] = field(default_factory=dict)
    curve_times: list[str] = field(default_factory=list)
    curve_values: list[float] = field(default_factory=list)
    grid_now: Optional[float] = None   # what the coarse grid thinks "now" is
    anchored: bool = False             # level came from a real CPCB reading
    observed_at: Optional[datetime] = None
    source_history: str = "Open-Meteo CAMS (hourly archive)"
    source_model: str = "Vayumitre XGBoost"

    @property
    def peak_value(self) -> Optional[float]:
        """Highest value the model expects across its horizons."""
        return max(self.horizons.values()) if self.horizons else None

    @property
    def peak_horizon(self) -> Optional[int]:
        """Hours ahead at which that peak lands."""
        if not self.horizons:
            return None
        return max(self.horizons, key=lambda h: self.horizons[h])

    @property
    def rise(self) -> Optional[float]:
        """Change from the current reading to the expected peak."""
        return None if self.peak_value is None else self.peak_value - self.pm25_now

    def as_dict(self) -> dict:
        return {
            "id": self.node_id, "name": self.name,
            "lat": self.lat, "lng": self.lng,
            "issued_at": self.issued_at.isoformat(),
            "now": round(self.pm25_now, 1),
            "anchored": self.anchored,
            "grid_now": None if self.grid_now is None else round(self.grid_now, 1),
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "h": {str(k): round(v, 1) for k, v in sorted(self.horizons.items())},
            "peak": None if self.peak_value is None else round(self.peak_value, 1),
            "peak_h": self.peak_horizon,
            "rise": None if self.rise is None else round(self.rise, 1),
            "curve_t": self.curve_times,
            "curve_v": [round(v, 1) for v in self.curve_values],
            "src_history": self.source_history,
            "src_model": self.source_model,
        }


# --------------------------------------------------------------------------- #
# Fetching                                                                     #
# --------------------------------------------------------------------------- #
def _batched(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _fetch(endpoint: str, points: Sequence[tuple], hourly: str,
           past_days: int, forecast_days: int, timeout: float) -> list[dict]:
    """Batched Open-Meteo call; always returns one dict per input point."""
    out: list[dict] = []
    for chunk in _batched(list(points), BATCH):
        query = urllib.parse.urlencode({
            "latitude": ",".join(f"{p[0]:.4f}" for p in chunk),
            "longitude": ",".join(f"{p[1]:.4f}" for p in chunk),
            "hourly": hourly,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Asia/Kolkata",
        })
        data = _get_json(f"{endpoint}?{query}", timeout=timeout,
                         label="Open-Meteo forecast history")
        # A single coordinate returns an object; several return a list.
        out.extend(data if isinstance(data, list) else [data])
    if len(out) != len(points):
        raise LiveDataError(
            f"Open-Meteo returned {len(out)} series for {len(points)} stations; "
            f"refusing to guess which is which."
        )
    return out


def fetch_history(points: Sequence[tuple[str, str, float, float]],
                  past_days: int = 8, forecast_days: int = 2,
                  timeout: float = HTTP_TIMEOUT_S) -> dict[str, dict]:
    """Return per-station hourly pollutant + weather series.

    ``points`` is ``(node_id, name, lat, lng)``.  The result maps ``node_id`` to
    a dict of parallel hourly lists, which is the shape the feature builder and
    the browser chart both want.
    """
    coords = [(p[2], p[3]) for p in points]
    air = _fetch(AIR_ENDPOINT, coords,
                 "pm2_5,pm10,nitrogen_dioxide,carbon_monoxide",
                 past_days, forecast_days, timeout)
    wx = _fetch(WX_ENDPOINT, coords,
                "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
                past_days, forecast_days, timeout)

    series: dict[str, dict] = {}
    for (node_id, name, lat, lng), a, w in zip(points, air, wx):
        ah, wh = a.get("hourly") or {}, w.get("hourly") or {}
        times = ah.get("time") or []
        if not times:
            continue
        wx_index = {t: k for k, t in enumerate(wh.get("time") or [])}

        def wcol(key):
            col = wh.get(key) or []
            return [col[wx_index[t]] if t in wx_index and wx_index[t] < len(col) else None
                    for t in times]

        series[node_id] = {
            "name": name, "lat": lat, "lng": lng, "time": times,
            "pm2_5": ah.get("pm2_5") or [],
            "pm10": ah.get("pm10") or [],
            "no2": ah.get("nitrogen_dioxide") or [],
            "co": ah.get("carbon_monoxide") or [],
            "temp_c": wcol("temperature_2m"),
            "humidity": wcol("relative_humidity_2m"),
            "windspeed_kph": wcol("wind_speed_10m"),
            "pressure_mb": wcol("surface_pressure"),
        }
    if not series:
        raise LiveDataError("Open-Meteo returned no usable hourly history.")
    return series


# --------------------------------------------------------------------------- #
# Model application                                                            #
# --------------------------------------------------------------------------- #
def _to_frame(series: dict[str, dict]):
    """Build the long dataframe the offline feature builder expects."""
    import pandas as pd

    rows = []
    for node_id, s in series.items():
        for k, t in enumerate(s["time"]):
            def val(key):
                col = s.get(key) or []
                return col[k] if k < len(col) else None
            rows.append({
                "location": node_id, "ts": t,
                "pm2_5": val("pm2_5"), "pm10": val("pm10"),
                "no2": val("no2"), "co": val("co"),
                "temp_c": val("temp_c"), "humidity": val("humidity"),
                "windspeed_kph": val("windspeed_kph"),
                "pressure_mb": val("pressure_mb"),
            })
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def run_models(series: dict[str, dict], model_dir: str,
               anchors: Optional[dict[str, tuple[float, datetime]]] = None
               ) -> list[StationForecast]:
    """Run every trained horizon model over each station's latest observed hour.

    Raises
    ------
    LiveDataError
        If the trained models are missing, or if no station has enough history
        to populate the feature set -- a partial feature vector would silently
        become a fabricated forecast.
    """
    import os

    import numpy as np
    import pandas as pd
    import xgboost as xgb

    from . import forecast as offline   # reuse the exact training-time features

    df = _to_frame(series)
    now = pd.Timestamp(datetime.now(IST).replace(tzinfo=None)).floor("h")

    # Features may only be built from *observed* hours. Open-Meteo's forward
    # curve is kept separately for display; feeding it into the lag features
    # would let a forecast predict itself.
    observed = df[df["ts"] <= now].copy()
    feat_df = offline.build_features(observed)
    feats = offline.feature_columns(feat_df)

    models = {}
    for h in HORIZONS:
        path = os.path.join(model_dir, f"xgb_pm25_h{h}.json")
        if not os.path.exists(path):
            continue
        m = xgb.XGBRegressor()
        m.load_model(path)
        models[h] = m
    if not models:
        raise LiveDataError(
            f"No trained models in {model_dir}. Run: python3 forecast.py")

    out: list[StationForecast] = []
    for node_id, s in series.items():
        g = feat_df[feat_df["location"] == node_id].sort_values("ts")
        usable = g.dropna(subset=feats)
        if usable.empty:
            continue                      # not enough history; station omitted
        last = usable.iloc[-1:]
        grid_now = float(last["pm2_5"].iloc[0])

        # Anchor the level to this monitor's own measurement where we have one.
        anchor = (anchors or {}).get(node_id)
        pm_now = float(anchor[0]) if anchor else grid_now
        observed_at = anchor[1] if anchor else None

        # Rescale the trajectory *proportionally*, not additively. The model's
        # delta is calibrated to the grid cell's own level, so subtracting it
        # wholesale from a much cleaner monitor drives the forecast to zero --
        # a -85 ug/m3 swing is a plausible fall from 125 and nonsense from 69.
        # A ratio keeps the shape of the predicted change while respecting the
        # station's scale, and cannot produce a negative concentration.
        scale = (pm_now / grid_now) if grid_now > 1.0 else 1.0

        horizons = {}
        for h, m in models.items():
            # Column *order* must match what the booster was fitted with -- the
            # names all match here, but a different order is still a mismatch.
            order = m.get_booster().feature_names or feats
            missing = [c for c in order if c not in last.columns]
            if missing:
                raise LiveDataError(
                    f"Live features are missing {missing[:5]} that the h={h} model "
                    f"expects. Retrain with forecast.py so the two agree.")
            # Models emit a delta from the grid's current value; scale it to
            # this monitor before adding it to the monitor's own reading.
            delta = float(m.predict(last[order])[0]) * scale
            horizons[h] = float(np.clip(pm_now + delta, 0, None))

        fwd = [(t, v) for t, v in zip(s["time"], s["pm2_5"])
               if v is not None and pd.Timestamp(t) > now]
        # Scale the grid's forward curve by the same ratio, so the curve and the
        # horizon markers describe one consistent forecast instead of two that
        # disagree by the grid's local bias.
        curve_vals = [max(0.0, float(v) * scale) for _, v in fwd[:24]]

        out.append(StationForecast(
            node_id=node_id, name=s["name"], lat=s["lat"], lng=s["lng"],
            issued_at=last["ts"].iloc[0].to_pydatetime(),
            pm25_now=pm_now, horizons=horizons,
            curve_times=[t for t, _ in fwd[:24]],
            curve_values=curve_vals,
            grid_now=grid_now, anchored=anchor is not None,
            observed_at=observed_at,
        ))
    if not out:
        raise LiveDataError(
            "No station had enough history to build a complete feature vector; "
            "refusing to emit a partial forecast.")
    return out


def cpcb_points(readings) -> tuple[list[tuple], dict[str, tuple[float, datetime]]]:
    """Turn live CPCB readings into forecast points plus their level anchors.

    Returns ``(points, anchors)`` where ``points`` is the ``(id, name, lat, lng)``
    list for :func:`fetch_history` and ``anchors`` maps each id to its measured
    PM2.5 and observation time.  Forecasting at the monitors keeps the unit of analysis the place the measurement was
    actually taken.
    """
    points, anchors = [], {}
    for r in readings:
        if r.pm25 is None or r.lat is None or r.lng is None:
            continue
        sid = r.station
        points.append((sid, r.station, r.lat, r.lng))
        anchors[sid] = (float(r.pm25), r.observed_at)
    return points, anchors


def spike_ranking(forecasts: Sequence[StationForecast],
                  critical: float) -> list[dict]:
    """Rank stations by the coming spike: how big a rise, and how soon.

    Two different questions get two different answers, so both are returned:

    ``model_peak`` / ``model_peak_h``
        the worst value the *validated* XGBoost expects, across its five
        horizons -- the number a dispatch rule should act on.
    ``curve_peak`` / ``curve_peak_at``
        the peak of the hourly forward curve, which resolves the *timing* of a
        spike far more precisely than five discrete horizons can.

    Sorted by expected rise above the current reading, because a cell climbing
    60 µg/m³ is the operational event even if a permanently dirty cell still
    reads higher in absolute terms.
    """
    rows = []
    for f in forecasts:
        if f.peak_value is None:
            continue
        curve_peak = max(f.curve_values) if f.curve_values else None
        curve_at = (f.curve_times[f.curve_values.index(curve_peak)]
                    if curve_peak is not None else None)
        rows.append({
            "id": f.node_id, "name": f.name,
            "now": round(f.pm25_now, 1),
            "model_peak": round(f.peak_value, 1),
            "model_peak_h": f.peak_horizon,
            "rise": round(f.rise, 1),
            "curve_peak": None if curve_peak is None else round(curve_peak, 1),
            "curve_peak_at": curve_at,
            "curve_rise": (None if curve_peak is None
                           else round(curve_peak - f.pm25_now, 1)),
            "breaches": max(filter(None, [f.peak_value, curve_peak])) >= critical,
        })
    return sorted(rows, key=lambda r: -(r["curve_rise"] if r["curve_rise"] is not None
                                        else r["rise"]))
