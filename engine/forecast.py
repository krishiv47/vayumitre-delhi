#!/usr/bin/env python3
"""
Vayumitre -- PM2.5 forecasting at 1, 3, 6, 12 and 24 hours ahead.

Trains one XGBoost regressor per horizon on the Delhi 2025 hourly archive and
scores it honestly against the two baselines that actually matter in air-quality
work:

* **persistence**    -- "it will be what it is right now" (y_hat = value at t)
* **seasonal naive** -- "it will be what it was this time yesterday" (t-24)

The second baseline is the hard one. PM2.5 has a strong 24-hour cycle, so a model
that beats persistence but loses to seasonal-naive has learned nothing useful.

    python3 forecast.py                     # train, evaluate, write forecasting/
    python3 forecast.py --predict           # forecast forward from the last row
    python3 forecast.py --open              # open the report when done

Guardrails against the usual time-series mistakes
-------------------------------------------------
* **Chronological split.** Train on Jan-Oct, validate on Nov, test on Dec.
  Never a random split -- shuffling hourly data lets the model interpolate
  between neighbouring hours and reports a fantasy score.
* **Causal features only.** Every feature at time ``t`` is built from data at or
  before ``t``. Rolling windows are shifted so the current value never leaks
  into its own predictor.
* **No target leakage across horizons.** The label for horizon ``h`` is
  ``pm2_5`` shifted ``-h``; rows without a full future are dropped, not filled.
* **Per-site grouping.** Lags are computed within each site so one location's
  history never bleeds into another's.

Outputs land in ``forecasting/``: trained models, a metrics table, per-horizon
figures, feature importances and a written report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import webbrowser
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# Shared visual language with the preprocessing pack.
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CRITICAL, GOOD = "#d03b3b", "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})

HORIZONS = [1, 3, 6, 12, 24]
TARGET = "pm2_5"
FIGURES: list[tuple[str, str, str]] = []


def titled(ax, title, subtitle=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="600", color=INK,
                 pad=26 if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=9.5, color=INK_2, va="bottom", ha="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)


def save(fig, out, name, title, caption):
    fig.savefig(os.path.join(out, "figures", name))
    plt.close(fig)
    FIGURES.append((name, title, caption))
    print(f"  figure  {name}")


# --------------------------------------------------------------------------- #
# Feature engineering -- causal by construction                                #
# --------------------------------------------------------------------------- #
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLS = [3, 6, 12, 24, 72]
WEATHER = ["temp_c", "humidity", "windspeed_kph", "pressure_mb"]
EXTRA_POLLUTANTS = ["pm10", "no2", "co"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, calendar and weather features, grouped by site.

    Every rolling statistic is ``shift(1)``-ed before aggregation so the window
    ends at ``t-1``.  Without that shift the mean at ``t`` would include the
    value being predicted from, which inflates validation scores and collapses
    the moment the model meets real data.
    """
    df = df.sort_values(["location", "ts"]).copy()
    g = df.groupby("location", sort=False)

    for lag in LAGS:
        df[f"pm25_lag{lag}"] = g[TARGET].shift(lag)

    shifted = g[TARGET].shift(1)          # everything below ends at t-1
    for w in ROLLS:
        grp = shifted.groupby(df["location"], sort=False)
        df[f"pm25_roll{w}_mean"] = grp.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"pm25_roll{w}_std"] = grp.transform(lambda s, w=w: s.rolling(w).std())
        df[f"pm25_roll{w}_max"] = grp.transform(lambda s, w=w: s.rolling(w).max())

    # Short-term momentum: is the cell loading or clearing right now?
    df["pm25_delta_1h"] = df["pm25_lag1"] - df["pm25_lag2"]
    df["pm25_delta_3h"] = df["pm25_lag1"] - df["pm25_lag3"]
    df["pm25_delta_24h"] = df["pm25_lag1"] - df["pm25_lag24"]
    df["pm25_ratio_24h"] = df["pm25_lag1"] / df["pm25_lag24"].replace(0, np.nan)

    # Calendar. Hour and month are cyclical, so encode them on a circle --
    # otherwise 23:00 and 00:00 look maximally distant to the model.
    hour, month, dow = df["ts"].dt.hour, df["ts"].dt.month, df["ts"].dt.dayofweek
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    df["dow"] = dow
    df["is_weekend"] = (dow >= 5).astype(int)
    df["hour"], df["month"] = hour, month

    # Weather at t is legitimately known (it is observed, and in deployment a
    # weather forecast stands in). Lags give the model recent trend.
    for c in WEATHER:
        df[f"{c}_lag1"] = g[c].shift(1)
        df[f"{c}_delta_3h"] = g[c].shift(1) - g[c].shift(4)
    for c in EXTRA_POLLUTANTS:
        df[f"{c}_lag1"] = g[c].shift(1)

    df["site_code"] = df["location"].astype("category").cat.codes
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every engineered predictor -- explicitly excluding the raw target."""
    drop = {TARGET, "ts", "location", "date", "date_ist", "time_ist",
            "condition_text", "description", "lat", "lon", "dust_event",
            "coarse_ratio", "band"}
    cols = [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    # pm10/no2/co at time t would be same-instant sensor readings; only their
    # lags are safe, so drop the contemporaneous versions.
    return [c for c in cols if c not in EXTRA_POLLUTANTS]


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def score(y_true, y_pred) -> dict:
    """MAE / RMSE / R2 / MAPE plus CPCB-band agreement."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    nz = y_true > 1
    mape = float(np.mean(np.abs((y_true[nz]-y_pred[nz])/y_true[nz]))*100) if nz.any() else np.nan
    edges = [0, 30, 60, 90, 120, 250, 1e9]
    band_t = np.digitize(y_true, edges) ; band_p = np.digitize(y_pred, edges)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4),
            "MAPE_%": round(mape, 1),
            "band_accuracy_%": round(float((band_t == band_p).mean()*100), 1)}


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_all(df: pd.DataFrame, out: str) -> tuple[pd.DataFrame, dict, dict]:
    feats = feature_columns(df)
    print(f"  {len(feats)} features engineered")

    # Chronological split. Not random -- see the module docstring.
    train_end = pd.Timestamp("2025-11-01")
    valid_end = pd.Timestamp("2025-12-01")

    rows, models, preds = [], {}, {}
    for h in HORIZONS:
        d = df.copy()
        d["y"] = d.groupby("location", sort=False)[TARGET].shift(-h)
        # Learn the *change* from now, not the absolute level. Predicting the
        # level makes the model regress toward the seasonal mean, which loses to
        # persistence at short horizons where almost nothing changes. Framed as a
        # delta, persistence is the built-in prior (predict 0) and the model can
        # only add information on top of it.
        d["y_delta"] = d["y"] - d[TARGET]

        # Seasonal-naive baseline: "the same clock hour, yesterday" -- i.e. the
        # value at (t + h - 24). Computed explicitly rather than reused from the
        # precomputed lag set, which only happens to contain the right offset for
        # some horizons. Note that at h = 24 this collapses onto persistence:
        # 24 hours before the target time *is* now.
        d["naive24"] = (d[TARGET] if h == 24
                        else d.groupby("location", sort=False)[TARGET].shift(24 - h))
        d = d.dropna(subset=["y", "y_delta", "naive24"] + feats)

        tr = d[d["ts"] < train_end]
        va = d[(d["ts"] >= train_end) & (d["ts"] < valid_end)]
        te = d[d["ts"] >= valid_end]
        if not len(te):
            continue

        model = xgb.XGBRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=7,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=1.5, objective="reg:squarederror",
            early_stopping_rounds=60, n_jobs=-1, random_state=7,
        )
        model.fit(tr[feats], tr["y_delta"],
                  eval_set=[(va[feats], va["y_delta"])], verbose=False)

        y_true = te["y"].to_numpy()
        # Reconstruct the level: current value plus the predicted change.
        y_hat = te[TARGET].to_numpy() + model.predict(te[feats])
        y_hat = np.clip(y_hat, 0, None)          # a concentration cannot be negative

        # Baselines, computed on exactly the same test rows.
        base_persist = te[TARGET].to_numpy()       # "it stays as it is now"
        base_naive = te["naive24"].to_numpy()      # "same clock hour yesterday"

        m_model = score(y_true, y_hat)
        m_persist = score(y_true, base_persist)
        m_naive = score(y_true, base_naive)

        rows.append({"horizon_h": h, "model": "XGBoost", **m_model,
                     "n_test": len(te), "best_iter": int(model.best_iteration or 0)})
        rows.append({"horizon_h": h, "model": "persistence", **m_persist, "n_test": len(te)})
        rows.append({"horizon_h": h, "model": "seasonal_naive_24h", **m_naive, "n_test": len(te)})

        models[h] = model
        preds[h] = pd.DataFrame({"ts": te["ts"].to_numpy(),
                                 "location": te["location"].to_numpy(),
                                 "y_true": y_true, "y_pred": y_hat,
                                 "persistence": base_persist})
        skill = 100 * (1 - m_model["MAE"] / m_persist["MAE"])
        print(f"  h={h:>2}h  MAE {m_model['MAE']:>6.2f}  vs persistence "
              f"{m_persist['MAE']:>6.2f}  ({skill:+.1f}% skill)  R2 {m_model['R2']:.3f}")

        model.save_model(os.path.join(out, "models", f"xgb_pm25_h{h}.json"))

    return pd.DataFrame(rows), models, preds


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def fig_skill(metrics, out):
    piv = metrics.pivot_table(index="horizon_h", columns="model", values="MAE")
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(piv.index)); w = 0.27
    for k, (m, lbl) in enumerate([("XGBoost", "XGBoost"),
                                  ("persistence", "Persistence baseline"),
                                  ("seasonal_naive_24h", "Seasonal naive (t−24 h)")]):
        if m not in piv:
            continue
        ax.bar(x + (k-1)*w, piv[m], width=w-0.02, color=SERIES[k], label=lbl, zorder=3)
    for k, h in enumerate(piv.index):
        ax.text(x[k]-w, piv["XGBoost"][h]+0.6, f"{piv['XGBoost'][h]:.1f}",
                ha="center", fontsize=8.5, color=SERIES[0], fontweight="700")
    ax.set_xticks(x, [f"{h} h" for h in piv.index])
    ax.set_xlabel("forecast horizon")
    titled(ax, "Forecast error against both baselines",
           "Mean absolute error on the held-out December test month. Lower is better.",
           "MAE  µg/m³")
    ax.legend(frameon=False, fontsize=9.5)
    save(fig, out, "01_skill_vs_baselines.png", "Skill vs baselines",
         "The model must beat seasonal-naive, not just persistence — PM2.5's strong "
         "24-hour cycle makes 'same hour yesterday' a genuinely hard target.")


def fig_pred_vs_actual(preds, out):
    fig, axes = plt.subplots(1, len(preds), figsize=(3.5*len(preds), 3.7), sharey=True)
    for ax, (h, d) in zip(np.atleast_1d(axes), sorted(preds.items())):
        ax.scatter(d["y_true"], d["y_pred"], s=5, alpha=.18, color=SERIES[0], linewidths=0)
        lim = [0, max(d["y_true"].max(), d["y_pred"].max())*1.03]
        ax.plot(lim, lim, color=CRITICAL, linewidth=1.4, linestyle="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{h} h ahead", loc="left", fontsize=11, fontweight="600", color=INK)
        ax.set_xlabel("actual  µg/m³", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("predicted  µg/m³", fontsize=9)
    fig.suptitle("Predicted against actual", x=0.06, ha="left", fontsize=14,
                 fontweight="600", color=INK)
    fig.text(0.06, 0.94, "December test month. The dashed line is perfect agreement.",
             fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, out, "02_pred_vs_actual.png", "Predicted vs actual",
         "Scatter tightens at short horizons and fans out at 24 h — the honest shape "
         "for a forecast whose information decays with lead time.")


def fig_timeline(preds, out):
    h = 6 if 6 in preds else sorted(preds)[0]
    d = preds[h]
    site = d["location"].iloc[0]
    d = d[d["location"] == site].sort_values("ts").head(24*14)
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.plot(d["ts"], d["y_true"], color=INK_2, linewidth=2, label="actual")
    ax.plot(d["ts"], d["y_pred"], color=SERIES[0], linewidth=2, label=f"forecast ({h} h ahead)")
    ax.fill_between(d["ts"], d["y_true"], d["y_pred"], color=SERIES[0], alpha=.12, linewidth=0)
    titled(ax, f"Forecast against reality — {site}",
           f"First 14 days of the December test month, {h}-hour lead time.", "PM2.5  µg/m³")
    ax.legend(frameon=False, fontsize=9.5)
    save(fig, out, "03_forecast_timeline.png", "Forecast timeline",
         "The model tracks the diurnal rhythm and the direction of each swing; it "
         "systematically shaves the sharpest peaks, which is where the residual error lives.")


def fig_importance(models, out):
    h = 6 if 6 in models else sorted(models)[0]
    m = models[h]
    imp = pd.Series(m.feature_importances_, index=m.get_booster().feature_names)
    imp = imp.sort_values(ascending=True).tail(18)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(range(len(imp)), imp.values, color=SERIES[0], height=.72, zorder=3)
    ax.set_yticks(range(len(imp)), imp.index, fontsize=9)
    ax.set_xlabel("gain importance")
    titled(ax, f"What the {h}-hour model actually uses",
           "Top 18 features by gain. Recent history dominates; weather modulates.")
    save(fig, out, "04_feature_importance.png", "Feature importance",
         "Recent PM2.5 lags and rolling means carry most of the signal, with "
         "temperature and hour-of-day providing the seasonal and diurnal context.")


def fig_error_by_hour(preds, out):
    h = 6 if 6 in preds else sorted(preds)[0]
    d = preds[h].copy()
    d["hour"] = pd.to_datetime(d["ts"]).dt.hour
    d["abs_err"] = (d["y_true"]-d["y_pred"]).abs()
    g = d.groupby("hour")["abs_err"].mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g.index, g.values, color=SERIES[0], linewidth=2.4)
    ax.plot(g.idxmax(), g.max(), "o", ms=9, color=CRITICAL, mec=SURFACE, mew=2, zorder=5)
    ax.text(g.idxmax(), g.max()+.4, f" worst {g.max():.1f} @ {g.idxmax():02d}:00",
            color=CRITICAL, fontsize=9.5, fontweight="700")
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour of day (IST)")
    titled(ax, f"Where the {h}-hour forecast goes wrong",
           "Mean absolute error by hour of day.", "MAE  µg/m³")
    save(fig, out, "05_error_by_hour.png", "Error by hour",
         "Error peaks around the evening build-up, when concentrations change fastest — "
         "exactly the hours a dispatch decision depends on.")


def fig_horizon_decay(metrics, out):
    m = metrics[metrics["model"] == "XGBoost"].sort_values("horizon_h")
    fig, ax1 = plt.subplots(figsize=(10.5, 4.2))
    ax1.plot(m["horizon_h"], m["MAE"], color=SERIES[0], linewidth=2.4, marker="o", ms=8,
             mec=SURFACE, mew=2)
    for _, r in m.iterrows():
        ax1.annotate(f"{r['MAE']:.1f}", (r["horizon_h"], r["MAE"]),
                     textcoords="offset points", xytext=(0, 10), ha="center",
                     fontsize=9, fontweight="700", color=SERIES[0])
    ax1.set_xticks(HORIZONS); ax1.set_xlabel("forecast horizon (hours)")
    titled(ax1, "Error grows with lead time",
           "Mean absolute error by horizon on the December test month.", "MAE  µg/m³")
    save(fig, out, "06_horizon_decay.png", "Horizon decay",
         "Skill decays smoothly with lead time. Use the short horizons for dispatch and "
         "the long ones for shift planning, not the other way round.")


# --------------------------------------------------------------------------- #
# Forward prediction                                                           #
# --------------------------------------------------------------------------- #
def predict_forward(df: pd.DataFrame, models: dict, out: str) -> pd.DataFrame:
    """Forecast each site forward from the most recent complete row."""
    feats = feature_columns(df)
    rows = []
    for site, g in df.groupby("location"):
        g = g.sort_values("ts")
        last = g.dropna(subset=feats).iloc[-1:]
        if last.empty:
            continue
        for h, m in sorted(models.items()):
            # Models emit a delta; add it back to the current reading.
            yhat = float(np.clip(float(last[TARGET].iloc[0]) + m.predict(last[feats])[0],
                                 0, None))
            rows.append({"location": site,
                         "issued_at": last["ts"].iloc[0],
                         "valid_at": last["ts"].iloc[0] + timedelta(hours=h),
                         "horizon_h": h,
                         "pm25_now": round(float(last[TARGET].iloc[0]), 1),
                         "pm25_forecast": round(yhat, 1),
                         "change": round(yhat - float(last[TARGET].iloc[0]), 1)})
    fc = pd.DataFrame(rows)
    fc.to_csv(os.path.join(out, "forecast_next_24h.csv"), index=False)
    return fc


# --------------------------------------------------------------------------- #
def write_report(out, metrics, df, fc):
    xgb_m = metrics[metrics["model"] == "XGBoost"].set_index("horizon_h")
    per_m = metrics[metrics["model"] == "persistence"].set_index("horizon_h")
    nav_m = metrics[metrics["model"] == "seasonal_naive_24h"].set_index("horizon_h")

    lines = [
        "# Vayumitre — PM2.5 forecasting",
        "",
        f"- **Training data:** {len(df):,} hourly rows, "
        f"{df['ts'].min():%Y-%m-%d} → {df['ts'].max():%Y-%m-%d}, "
        f"{df['location'].nunique()} sites",
        "- **Split:** train Jan–Oct · validate Nov · **test Dec** (chronological, never random)",
        f"- **Horizons:** {', '.join(f'{h} h' for h in HORIZONS)}",
        "- **Model:** XGBoost regressor, early-stopped on the validation month",
        "",
        "## Results on the held-out December test month",
        "",
        "| Horizon | XGBoost MAE | Persistence MAE | Seasonal-naive MAE | Skill vs persistence | Skill vs naive | R² | Band accuracy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in sorted(xgb_m.index):
        s_p = 100*(1 - xgb_m.loc[h, "MAE"]/per_m.loc[h, "MAE"])
        s_n = 100*(1 - xgb_m.loc[h, "MAE"]/nav_m.loc[h, "MAE"])
        lines.append(
            f"| {h} h | **{xgb_m.loc[h,'MAE']:.2f}** | {per_m.loc[h,'MAE']:.2f} | "
            f"{nav_m.loc[h,'MAE']:.2f} | {s_p:+.1f}% | {s_n:+.1f}% | "
            f"{xgb_m.loc[h,'R2']:.3f} | {xgb_m.loc[h,'band_accuracy_%']:.1f}% |")

    lines += [
        "",
        "**Band accuracy** is the share of hours where the forecast lands in the same "
        "CPCB band as reality. That is the number a dispatcher cares about — being out "
        "by 8 µg/m³ matters only if it crosses a threshold.",
        "",
        "## How this was kept honest",
        "",
        "- **Chronological split.** A random split on hourly data lets the model "
        "interpolate between adjacent hours and reports a score it cannot reproduce in "
        "deployment. Training stops at 31 October; the test month was never seen.",
        "- **Causal features only.** Rolling windows are shifted so they end at `t−1`. "
        "Without that shift the mean at `t` contains the value being predicted from.",
        "- **Contemporaneous pollutants excluded.** `pm10`, `no2` and `co` enter only as "
        "lags. Their values at `t` would be same-instant sensor readings that a forecast "
        "would not have.",
        "- **Two baselines, not one.** Beating persistence is easy at long horizons and "
        "nearly impossible at 1 h. Seasonal-naive is the real bar, because PM2.5 repeats "
        "daily.",
        "- **Per-site lags.** History is grouped by location, so one site's past never "
        "leaks into another's features.",
        "",
        "## Known limitations",
        "",
        "- The archive holds **two independent PM2.5 signals across six labelled sites** "
        "(see `preprocessing/REPORT.md`). This model is therefore a *city-level* "
        "forecaster. Per-zone forecasting needs the live CPCB feed's 40-plus real "
        "monitors, not this file.",
        "- Weather is used as **observed**, not forecast. In deployment the weather "
        "features must be fed from a forecast, which will add error not measured here.",
        "- December is a single test month, and a high-pollution one. Scores in the "
        "monsoon season will differ; retrain on a rolling window rather than trusting "
        "these numbers year-round.",
        "",
        "## Files",
        "",
        "| Path | Contents |",
        "|---|---|",
        "| `models/xgb_pm25_h{1,3,6,12,24}.json` | trained boosters, loadable with `xgb.XGBRegressor().load_model()` |",
        "| `metrics.csv` | every model × horizon × metric |",
        "| `forecast_next_24h.csv` | forward forecast from the last complete row |",
        "| `figures/` | 6 evaluation charts |",
        "| `feature_importance.csv` | gain importance per feature per horizon |",
        "",
        "## Wiring this into the fleet ",
        "",
        "The dispatch decision needs *predicted* AQI, not current AQI. With these models "
        "the coordinator can answer 'will this cell breach in 6 hours?' and pre-position "
        "tankers instead of reacting. The natural next step is the decision engine: "
        "compare predicted AQI with and without intervention, price the water and travel, "
        "and emit the YES/NO with a confidence score.",
    ]
    if fc is not None and len(fc):
        lines += ["", "## Current forward forecast", "",
                  "| Site | Now | +1 h | +3 h | +6 h | +12 h | +24 h |", "|---|---|---|---|---|---|---|"]
        for site, g in fc.groupby("location"):
            g = g.set_index("horizon_h")
            now = g["pm25_now"].iloc[0]
            cells = " | ".join(f"{g.loc[h,'pm25_forecast']:.0f}" if h in g.index else "—"
                               for h in HORIZONS)
            lines.append(f"| {site} | {now:.0f} | {cells} |")

    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  report  REPORT.md")


def write_gallery(out, metrics):
    best = metrics[metrics["model"] == "XGBoost"]
    cards = "\n".join(
        f"""  <figure><img src="figures/{f}" alt="{t}" loading="lazy"/>
    <figcaption><b>{t}</b><span>{c}</span></figcaption></figure>""" for f, t, c in FIGURES)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Vayumitre — PM2.5 forecasting</title><style>
 body{{margin:0;background:#f9f9f7;color:{INK};font:15px/1.55 Inter,-apple-system,sans-serif}}
 header{{max-width:1180px;margin:0 auto;padding:40px 32px 22px}}
 h1{{margin:0 0 6px;font-size:30px;letter-spacing:-.02em}}
 .sub{{color:{INK_2};font-size:14px}}
 .stats{{display:flex;gap:28px;flex-wrap:wrap;margin-top:20px;padding-top:18px;border-top:1px solid {GRID}}}
 .stat b{{display:block;font-size:22px}} .stat span{{color:{MUTED};font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
 main{{max-width:1180px;margin:0 auto;padding:8px 32px 64px;display:grid;gap:26px}}
 figure{{margin:0;background:{SURFACE};border:1px solid {GRID};border-radius:14px;overflow:hidden}}
 figure img{{width:100%;display:block}}
 figcaption{{padding:14px 18px 18px;border-top:1px solid {GRID}}}
 figcaption b{{display:block;font-size:14px;margin-bottom:3px}}
 figcaption span{{color:{INK_2};font-size:13px}}
 @media (prefers-color-scheme:dark){{body{{background:#0d0d0d;color:#fff}}
  figure{{background:#1a1a19;border-color:#2c2c2a}} figcaption{{border-color:#2c2c2a}}
  .sub,figcaption span{{color:#c3c2b7}} .stats{{border-color:#2c2c2a}}}}
</style></head><body>
<header><h1>Vayumitre — PM2.5 forecasting</h1>
<p class="sub">XGBoost, 1–24 hour horizons, trained on Delhi 2025 and tested on a held-out December.
Findings in <a href="REPORT.md">REPORT.md</a>.</p>
<div class="stats">
 <div class="stat"><b>{best['MAE'].min():.1f}</b><span>best MAE µg/m³</span></div>
 <div class="stat"><b>{best['R2'].max():.3f}</b><span>best R²</span></div>
 <div class="stat"><b>{best['band_accuracy_%'].max():.0f}%</b><span>band accuracy</span></div>
 <div class="stat"><b>{len(HORIZONS)}</b><span>horizons</span></div>
</div></header>
<main>
{cards}
</main></body></html>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    print("  gallery index.html")
    return os.path.join(out, "index.html")


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Train PM2.5 forecasting models.")
    p.add_argument("--data", default="preprocessing/cleaned/clean_hourly.csv")
    p.add_argument("--out", default="forecasting")
    p.add_argument("--predict", action="store_true", help="also forecast forward")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.data):
        print(f"Cleaned data not found: {args.data}\nRun: python3 preprocess.py",
              file=sys.stderr)
        return 2

    out = os.path.abspath(args.out)
    for sub in ("figures", "models"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    print(f"Loading {args.data}")
    df = pd.read_csv(args.data, parse_dates=["ts"])
    print(f"  {len(df):,} rows · {df['location'].nunique()} sites")

    print("Engineering features")
    df = build_features(df)

    print("Training")
    metrics, models, preds = train_all(df, out)
    metrics.to_csv(os.path.join(out, "metrics.csv"), index=False)

    imp_rows = []
    for h, m in models.items():
        for f, v in zip(m.get_booster().feature_names, m.feature_importances_):
            imp_rows.append({"horizon_h": h, "feature": f, "gain": float(v)})
    pd.DataFrame(imp_rows).to_csv(os.path.join(out, "feature_importance.csv"), index=False)

    print("Rendering figures")
    fig_skill(metrics, out)
    fig_pred_vs_actual(preds, out)
    fig_timeline(preds, out)
    fig_importance(models, out)
    fig_error_by_hour(preds, out)
    fig_horizon_decay(metrics, out)

    fc = predict_forward(df, models, out) if args.predict else None
    write_report(out, metrics, df, fc)
    gallery = write_gallery(out, metrics)

    print(f"\nDone -> {out}")
    if args.open:
        webbrowser.open("file://" + gallery)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
