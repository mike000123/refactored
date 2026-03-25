"""
sidebar.py
----------
Renders the entire Streamlit sidebar and returns a populated AppConfig.
Import this from app.py only — Streamlit calls live here.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from cache_store import FRED_CACHE_DIR, YF_DAILY_CACHE_DIR, clear_cache_dir
from config import AppConfig
from model_runners import CRISIS_PRESET_WINDOWS
from ui_themes import theme_picker


def simple_explainer(title: str, body: str) -> None:
    with st.expander(title):
        st.markdown(body)


def render_sidebar(res_base: pd.DataFrame | None = None) -> tuple[AppConfig, object]:
    """
    Render the full sidebar. Returns (AppConfig, theme).
    *res_base* is only needed for the threshold matrix placeholder (optional).
    """
    cfg = AppConfig()

    with st.sidebar:
        st.header("Settings")
        simple_explainer("What this section means in simple words", """
This is where you choose **how the dashboard thinks and what it shows**.

Select the model mode, weights, thresholds, chart windows, and optional fast indicators.
        """)
        theme = theme_picker(key="theme_picker_global")

    with st.sidebar:
        # ── Mode ──────────────────────────────────────────────────────────────
        cfg.mode = st.selectbox(
            "Model mode",
            ["Structural Regime (today)", "Crisis Similarity (template)", "Market Acceleration (fast)"],
            index=2,
        )

        cfg.enable_compare = st.checkbox(
            "Enable Comparison View",
            value=False,
            help="Compare any two modes side-by-side.",
        )
        cfg.compare_modes = None
        if cfg.enable_compare:
            cfg.compare_modes = st.multiselect(
                "Compare exactly two modes",
                ["Structural Regime (today)", "Crisis Similarity (template)", "Market Acceleration (fast)"],
                default=["Structural Regime (today)", "Market Acceleration (fast)"],
                max_selections=2,
            )

        # ── Crisis window ──────────────────────────────────────────────────────
        need_crisis_ui = (cfg.mode == "Crisis Similarity (template)") or (
            cfg.enable_compare and cfg.compare_modes and "Crisis Similarity (template)" in cfg.compare_modes
        )

        if need_crisis_ui:
            st.subheader("Crisis Threshold Package")
            cfg.crisis = st.selectbox(
                "Preset window",
                list(CRISIS_PRESET_WINDOWS.keys()) + ["Custom"],
                index=2,
            )
            if cfg.crisis == "Custom":
                cfg.win_start = st.text_input("Custom start (YYYY-MM-DD)", "1979-01-01")
                cfg.win_end = st.text_input("Custom end (YYYY-MM-DD)", "1983-12-31")
            else:
                cfg.win_start, cfg.win_end = CRISIS_PRESET_WINDOWS.get(cfg.crisis, (None, None))

        # ── Indicator weights ─────────────────────────────────────────────────
        st.subheader("Indicator Weights")
        simple_explainer("What this section means in simple words", """
These sliders set how much each indicator contributes to the overall score.
A higher weight means that indicator has more influence.
        """)
        cfg.w_real = st.slider("Weight: Real yield (10Y−CPI)", 0.0, 1.0, 0.30, 0.01)
        cfg.w_infl = st.slider("Weight: Inflation (CPI YoY)", 0.0, 1.0, 0.20, 0.01)
        cfg.w_usd = st.slider("Weight: USD index (12M change)", 0.0, 1.0, 0.20, 0.01)
        cfg.w_curve = st.slider("Weight: Yield curve (10Y−3M)", 0.0, 1.0, 0.15, 0.01)
        cfg.w_fisc = st.slider(
            "Weight: Deficit % GDP",
            0.0, 1.0, 0.15, 0.01,
            disabled=(cfg.mode != "Structural Regime (today)"),
        )

        cfg.include_tips = st.checkbox(
            "Include TIPS real yield in Gold Structural model",
            value=False,
            disabled=(cfg.mode != "Structural Regime (today)"),
        )
        cfg.w_tips = st.slider(
            "Weight: TIPS real yield",
            0.0, 1.0, 0.10, 0.01,
            disabled=(not cfg.include_tips or cfg.mode != "Structural Regime (today)"),
        )

        cfg.include_hy = st.checkbox(
            "Include HY credit spread in Gold Structural model",
            value=False,
            disabled=(cfg.mode != "Structural Regime (today)"),
        )
        cfg.w_hy = st.slider(
            "Weight: HY OAS",
            0.0, 1.0, 0.10, 0.01,
            disabled=(not cfg.include_hy or cfg.mode != "Structural Regime (today)"),
        )

        if cfg.mode != "Structural Regime (today)":
            cfg.w_fisc = 0.0

        # ── Monthly aggregation ───────────────────────────────────────────────
        st.subheader("Monthly aggregation")
        cfg.monthly_method = st.selectbox("Method for daily series", ["avg", "eom"], index=0)

        # ── Intraday RSI screener tickers ─────────────────────────────────────
        st.subheader("Intraday RSI Screener (Top 10)")
        _default_tickers = "AAPL MSFT NVDA AMZN META GOOGL TSLA AVGO COST AMD WMT ASML MU NFLX PLTR CSCO AMAT LRCX PEP INTC"
        MAX_TICKERS = 20
        raw_top10 = st.text_area(
            f"Tickers (space/comma/newline separated) — max {MAX_TICKERS}",
            value=_default_tickers,
        )
        parsed = (
            pd.Series(raw_top10.replace(",", " ").split())
            .astype(str).str.strip().str.upper().tolist()
        )
        cfg.tickers_top10 = list(dict.fromkeys(t for t in parsed if t))[:MAX_TICKERS]

        # ── Chart controls ────────────────────────────────────────────────────
        st.subheader("Charts")
        lookback_options = [6, 12, 24, 36, 60, 120, 180, 240, 360]
        lookback_labels = [str(x) for x in lookback_options] + ["All available"]
        lookback_label = st.selectbox(
            "Indicators' Plot Timeframe (months)",
            lookback_labels,
            index=lookback_labels.index("240"),
        )
        cfg.lookback = None if lookback_label == "All available" else int(lookback_label)
        cfg.history_view = st.selectbox(
            "History view",
            ["Full history", "Last 15y", "Last 5y", "Crisis window only", "Crisis window ±5y"],
            index=0,
        )
        cfg.show_indicator_thresholds = st.checkbox(
            "Show indicator bull/bear thresholds on sparklines",
            value=True,
        )

        # ── Triggers ─────────────────────────────────────────────────────────
        st.subheader("Trigger")
        cfg.trig_hi = st.slider("Bull trigger (+)", 0.0, 1.5, 0.60, 0.05)
        cfg.trig_lo = st.slider("Bear trigger (−)", -1.5, 0.0, -0.60, 0.05)
        cfg.persist = st.slider("Persistence months", 1, 3, 2, 1)

        # ── Monte Carlo ───────────────────────────────────────────────────────
        st.subheader("Monte Carlo Settings")
        cfg.mc_horizon = st.slider("Monte Carlo horizon (months)", 3, 24, 12, 1)
        cfg.mc_n_sims = st.slider("Monte Carlo simulations", 500, 5000, 2000, 500)

        # ── Live updates ──────────────────────────────────────────────────────
        st.subheader("Live updates")
        cfg.live_updates = st.checkbox("Enable live updates (auto-refresh)", value=False)
        cfg.refresh_seconds = st.selectbox("Refresh every…", [60], index=0, disabled=not cfg.live_updates)

        # ── Acceleration settings ─────────────────────────────────────────────
        need_accel_ui = (cfg.mode == "Market Acceleration (fast)") or (
            cfg.enable_compare and cfg.compare_modes and "Market Acceleration (fast)" in cfg.compare_modes
        )
        if need_accel_ui:
            st.subheader("Acceleration Settings")
            cfg.accel_method = st.selectbox(
                "Acceleration thresholds",
                ["Fixed (recommended)", "Quantiles (2000-present)"],
                index=0,
            )
            st.caption("Weights (auto-normalized)")
            cfg.w_g = st.slider("Gold momentum (6M return)", 0.0, 1.0, 0.35, 0.01)
            cfg.w_ry = st.slider("Real yield velocity (3M change)", 0.0, 1.0, 0.25, 0.01)
            cfg.w_st = st.slider("Stress velocity (3M change)", 0.0, 1.0, 0.25, 0.01)
            cfg.w_u = st.slider("USD velocity (3M change)", 0.0, 1.0, 0.15, 0.01)

            st.subheader("Intraday RSI (optional)")
            cfg.show_intraday_rsi = st.checkbox("Show intraday RSI", value=True)
            cfg.intraday_interval = st.selectbox(
                "Intraday interval", ["1m", "5m", "15m", "30m", "60m"], index=0,
                disabled=(not cfg.show_intraday_rsi),
            )
            LOOKBACK_BY_INTERVAL = {
                "1m": [1, 2, 3],
                "5m": [1, 2, 3, 5, 7, 10, 15, 30, 60],
                "15m": [1, 2, 3, 5, 7, 10, 15, 30, 60],
                "30m": [1, 2, 3, 5, 7, 10, 15, 30, 60],
                "60m": [5, 7, 10, 15, 30, 60, 90, 180, 365, 730],
            }
            valid_lookbacks = LOOKBACK_BY_INTERVAL.get(cfg.intraday_interval, [1, 2, 3, 5, 10])
            prev_lb = st.session_state.get("intraday_lookback_days", None)
            default_lb = prev_lb if prev_lb in valid_lookbacks else valid_lookbacks[-1]
            cfg.intraday_lookback_days = st.selectbox(
                "Intraday lookback (days)", valid_lookbacks,
                index=valid_lookbacks.index(default_lb),
                disabled=(not cfg.show_intraday_rsi),
                key="intraday_lookback_days",
            )

        # ── Threshold matrix placeholder ──────────────────────────────────────
        st.markdown("---")
        st.subheader("Threshold Matrix")
        simple_explainer("What this section means in simple words", """
This matrix shows the **rules behind the score**.
For each indicator, it shows what the platform treats as bullish, bearish, and in-between.
        """)
        threshold_box = st.container()

        # ── Refresh / clear ───────────────────────────────────────────────────
        st.markdown("---")
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.session_state["force_refresh_data"] = True
            st.rerun()

        if st.button("Clear local data cache"):
            for folder in [FRED_CACHE_DIR, YF_DAILY_CACHE_DIR]:
                clear_cache_dir(folder)
            st.cache_data.clear()
            st.success("Local cache cleared.")
            st.rerun()

    return cfg, theme, threshold_box
