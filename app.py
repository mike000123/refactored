"""
app.py
------
Entry point for the MRMI platform.
All computation lives in feature_builder, model_runners, and the tabs packages.
This file contains only Streamlit coordination: page config, data loading,
model dispatch, tab rendering, and report export.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import data_loader  # sets data_loader.refresh_remote_data flag
from config import AppConfig, RunResults
from feature_builder import build_features
from macro_models import compute_accel_features
from model_runners import (
    DIRS_STRUCT, LABELS, LABELS_ACCEL,
    build_weight_dicts, run_accel, run_crisis, run_market_structural, run_structural,
)
from report_builder import build_word_report, build_word_report_compare, LEGEND_MAP
from sidebar import render_sidebar, simple_explainer
from tabs import tab_gold, tab_screener, tab_backtest
from ui_themes import apply_theme


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MRMI – Macro Regime & Market Intelligence", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
cfg, theme, threshold_box = render_sidebar()
apply_theme(theme)

# ── Sync global refresh flag ──────────────────────────────────────────────────
data_loader.refresh_remote_data = bool(st.session_state.get("force_refresh_data", False))

# ── Page header ────────────────────────────────────────────────────────────────
st.title("MRMI - Macro Regime & Market Intelligence platform")
with st.expander("How to interpret this model"):
    st.markdown("""
**Model Modes**
- **Structural Regime (today):** Evaluates whether the current macro backdrop is structurally supportive or hostile to gold.
- **Crisis Similarity (template):** Measures how similar today's macro conditions are to a selected historical crisis.
- **Market Acceleration (today):** Evaluates whether market dynamics are accelerating or slowing gold price movements.

**Regime Labels**
- 🟢 Structural Bull → Strong macro tailwind
- 🟢 Positive → Mild tailwind
- 🟡 Neutral → Balanced
- 🔴 Vulnerable → Mild headwind
- 🔴 Structural Headwind → Strong macro pressure

**Triggers**
Bull trigger activates when the signal stays above the upper threshold for the selected persistence period.
Bear trigger activates when it stays below the lower threshold.
    """)

# ── Live-update auto-refresh ───────────────────────────────────────────────────
refresh_token = 0
if cfg.live_updates and cfg.refresh_seconds and cfg.refresh_seconds > 0:
    refresh_token = int(time.time() // int(cfg.refresh_seconds))
    components.html(f"<meta http-equiv='refresh' content='{int(cfg.refresh_seconds)}'>", height=0, width=0)

# ── Data load ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60 * 60 * 6, show_spinner="Loading macro data…")
def _load(monthly_method: str) -> tuple[pd.DataFrame, list[str]]:
    return build_features(monthly_method=monthly_method)


df, load_warnings = _load(cfg.monthly_method)

for w in load_warnings:
    st.warning(w)

if df.empty:
    st.error("No usable rows after feature construction. Check your FRED/Yahoo connectivity.")
    st.stop()

# ── Intraday data (Acceleration mode only) ─────────────────────────────────────
if cfg.mode == "Market Acceleration (fast)" and cfg.show_intraday_rsi:
    from data_loader import yf_intraday_close
    from feature_builder import compute_intraday_rsi_pack

    @st.cache_data(ttl=65, show_spinner=False)
    def _intraday(interval: str, lookback: int, token: int) -> pd.Series:
        return yf_intraday_close(["GC=F", "XAUUSD=X", "GLD"], interval=interval, lookback_days=lookback)

    gold_intra = _intraday(cfg.intraday_interval, cfg.intraday_lookback_days, refresh_token)
    if isinstance(gold_intra, pd.Series) and not gold_intra.dropna().empty:
        df.attrs["gold_intraday"] = gold_intra
        pack = compute_intraday_rsi_pack(gold_intra, rsi_period=14)
        for key, val in pack.items():
            df.attrs[key] = val
        fair_month = df.attrs.get("fair_gold_monthly_raw")
        if isinstance(fair_month, pd.Series) and not fair_month.dropna().empty:
            df.attrs["fair_gold_intraday"] = fair_month.reindex(gold_intra.index, method="ffill")
    else:
        df.attrs["gold_intraday"] = pd.Series(dtype=float)
        df.attrs["RSI_14_INTRA"] = pd.Series(dtype=float)

# ── Build weight dicts ─────────────────────────────────────────────────────────
res_base = df.copy()
res_accel_base = compute_accel_features(res_base)

weights_struct, weights_crisis = build_weight_dicts(
    cfg.w_real, cfg.w_infl, cfg.w_usd, cfg.w_curve, cfg.w_fisc,
    cfg.w_tips, cfg.w_hy, cfg.include_tips, cfg.include_hy, res_base, cfg.mode,
)

# ── Mode dispatch ──────────────────────────────────────────────────────────────
crisis_fit = {"fit_pct": np.nan, "coverage_used": 0, "coverage_total": 0, "by_indicator": {}}
error_msg: str | None = None

if cfg.mode == "Structural Regime (today)":
    gold_stats = None
    res, core_keys, thresholds, dirs, weights = run_structural(res_base, weights_struct)
    labels = LABELS
elif cfg.mode == "Crisis Similarity (template)":
    if not (cfg.win_start and cfg.win_end):
        st.error("Crisis Similarity mode requires a preset window.")
        st.stop()
    res, core_keys, thresholds, dirs, weights, gold_stats, crisis_fit, error_msg = run_crisis(
        res_base, weights_crisis, win_start=cfg.win_start, win_end=cfg.win_end,
    )
    labels = LABELS
    if error_msg:
        st.error(error_msg)
        # Don't stop — fall through so the rest of the page remains usable
else:
    gold_stats = None
    res, core_keys, thresholds, dirs, weights = run_accel(
        res_accel_base, cfg.accel_method, cfg.w_g, cfg.w_ry, cfg.w_st, cfg.w_u,
    )
    labels = LABELS_ACCEL

# ── Market structural (always run) ────────────────────────────────────────────
res_market, core_keys_market, thresholds_market, dirs_market, weights_market = run_market_structural(res_base)
latest_market = res_market.iloc[-1] if not res_market.empty else None
market_regime_now = str(latest_market.get("REGIME", "—")) if latest_market is not None else "—"
market_signal_now = float(latest_market["SIGNAL"]) if latest_market is not None and "SIGNAL" in latest_market else np.nan

# ── Sanity check ───────────────────────────────────────────────────────────────
if res.empty:
    st.error("No usable rows produced. One or more core FRED series may have failed.")
    st.stop()

# ── Pack results ───────────────────────────────────────────────────────────────
results = RunResults(
    res=res, res_base=res_base, res_accel_base=res_accel_base,
    core_keys=core_keys, thresholds=thresholds, dirs=dirs, weights=weights, labels=labels,
    gold_stats=gold_stats, crisis_fit=crisis_fit,
    res_market=res_market, core_keys_market=core_keys_market,
    thresholds_market=thresholds_market, dirs_market=dirs_market, weights_market=weights_market,
    latest_market=latest_market, market_regime_now=market_regime_now, market_signal_now=market_signal_now,
    weights_struct=weights_struct, weights_crisis=weights_crisis,
)

# ── Threshold matrix in sidebar ───────────────────────────────────────────────
with threshold_box:
    _title = {
        "Structural Regime (today)": "Structural Threshold Matrix",
        "Crisis Similarity (template)": "Crisis Threshold Matrix",
    }.get(cfg.mode, "Acceleration Threshold Matrix")
    st.caption(_title)

    common_keys = ["REAL_YIELD_CPI", "USD_12M_CHG", "CURVE_10Y_3M"]
    gold_keys = ["CPI_YOY", "DEFICIT_GDP", "REAL_YIELD_TIPS10"]
    market_keys = ["HY_OAS", "QQQ_ABOVE_MA200", "QQQ_MA50_SLOPE_20D", "MARKET_BREADTH_ABOVE_MA200"]

    def _render_threshold_group(group_title: str, keys_to_show: list) -> None:
        shown = [k for k in keys_to_show if k in labels]
        if not shown:
            return
        st.markdown(f"#### {group_title}")
        for key in shown:
            gold_scored = key in core_keys and key in thresholds
            market_scored = (core_keys_market is not None and thresholds_market is not None
                             and key in core_keys_market and key in thresholds_market)
            if market_scored and not gold_scored:
                th = thresholds_market[key]
                higher_is_bull = dirs_market.get(key, False)
            elif gold_scored:
                th = thresholds[key]
                higher_is_bull = dirs.get(key, False)
            else:
                continue
            st.markdown(f"**{labels.get(key, key)}**")
            if key == "QQQ_ABOVE_MA200":
                st.caption("Bull if = Yes (above MA200)")
                st.caption("Bear if = No (below MA200)")
            elif higher_is_bull:
                st.caption(f"Bull if ≥ {th['bull']:.2f}")
                st.caption(f"Bear if ≤ {th['bear']:.2f}")
            else:
                st.caption(f"Bull if ≤ {th['bull']:.2f}")
                st.caption(f"Bear if ≥ {th['bear']:.2f}")
            st.markdown("---")

    if cfg.mode in ["Structural Regime (today)", "Crisis Similarity (template)"]:
        _render_threshold_group("Common Macro Indicators", common_keys)
        _render_threshold_group("Gold-Specific Indicators", gold_keys)
        _render_threshold_group("Market-Specific Indicators", market_keys)
    else:
        for key in core_keys:
            if key not in thresholds:
                continue
            th = thresholds[key]
            higher_is_bull = dirs.get(key, False)
            st.markdown(f"**{labels.get(key, key)}**")
            st.caption(f"Bull if {'≥' if higher_is_bull else '≤'} {th['bull']:.2f}")
            st.caption(f"Bear if {'≤' if higher_is_bull else '≥'} {th['bear']:.2f}")
            st.markdown("---")

# ── Crisis gold response block ─────────────────────────────────────────────────
if cfg.mode == "Crisis Similarity (template)":
    st.subheader("Crisis-conditioned Gold Response (in-template)")
    if gold_stats and gold_stats.get("6m", {}).get("ok"):
        g6 = gold_stats["6m"]
        st.write(
            f"Similar macro signal months (±0.15): n={g6['n_samples']} | "
            f"6M fwd mean={g6['mean_ret']:.1f}% | median={g6['median_ret']:.1f}% | "
            f"P(>0)={100*g6['p_pos']:.0f}% | IQR=({g6['q25']:.1f}%, {g6['q75']:.1f}%)"
        )
    else:
        st.warning(
            "No reliable gold-conditioned stats: "
            + (gold_stats["6m"].get("reason") if gold_stats and "6m" in gold_stats else "missing")
        )

# ── Tabs ───────────────────────────────────────────────────────────────────────
if cfg.mode == "Market Acceleration (fast)":
    tab1, tab2, tab3, tab4 = st.tabs(["Gold Acceleration", "Intraday RSI Screener", "Monte Carlo", "Walk-Forward Backtest"])
    with tab1:
        bull_now, bear_now = tab_gold.render(results, cfg, theme)
    with tab2:
        tab_screener.render_screener(results, cfg, refresh_token)
    with tab3:
        tab_screener.render_monte_carlo(results, cfg, refresh_token)
    with tab4:
        tab_backtest.render(results, cfg)
else:
    bull_now, bear_now = tab_gold.render(results, cfg, theme)

# ── Report export ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Report")

latest = results.res.iloc[-1]
prev = results.res.iloc[-2] if len(results.res) > 1 else latest

trigger_info = {
    "bull_now": bull_now,
    "bear_now": bear_now,
    "trig_hi": cfg.trig_hi,
    "trig_lo": cfg.trig_lo,
    "persist": cfg.persist,
}

if st.button("Generate Word report (.docx)"):
    end_dt = results.res.index.max()
    if cfg.history_view == "Last 15y":
        res_plot_for_report = results.res.loc[end_dt - pd.DateOffset(years=15):]
    elif cfg.history_view == "Last 5y":
        res_plot_for_report = results.res.loc[end_dt - pd.DateOffset(years=5):]
    elif cfg.history_view == "Crisis window only" and cfg.win_start and cfg.win_end:
        res_plot_for_report = results.res.loc[pd.to_datetime(cfg.win_start):pd.to_datetime(cfg.win_end)]
    elif cfg.history_view == "Crisis window ±5y" and cfg.win_start and cfg.win_end:
        start_dt = pd.to_datetime(cfg.win_start) - pd.DateOffset(years=5)
        end_win = pd.to_datetime(cfg.win_end) + pd.DateOffset(years=5)
        res_plot_for_report = results.res.loc[start_dt:end_win]
    else:
        res_plot_for_report = results.res

    docx_bytes = build_word_report(
        latest_row=latest, prev_row=prev, thresholds=results.thresholds,
        weights=results.weights, trigger_info=trigger_info, labels=results.labels,
        legend_map=LEGEND_MAP, core_keys=results.core_keys, mode=cfg.mode,
        crisis_year=cfg.crisis or "—", res_plot=res_plot_for_report,
        gold_stats=results.gold_stats if cfg.mode == "Crisis Similarity (template)" else None,
    )
    fname = f"MRMI_Report_{latest.name.date()}.docx"
    st.download_button(
        label="Download report", data=docx_bytes, file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

st.caption("v6 — Refactored architecture. Real yield + inflation use long-history proxies to support 1970s/1980s presets.")
