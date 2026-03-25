"""
tabs/tab_gold.py
----------------
Tab 1: Gold Acceleration / Gold Analysis — full restoration including:
  - KPI strip (all modes)
  - Indicator cards with sparklines / crisis comparison charts
  - Triggers (action layer)
  - Acceleration charts: RSI+Gold+FairGold row, contributions row, intraday row
  - History panel: Signal / dual-axis Gold+Signal+Market / stacked contributions
  - Current Contribution by Indicator bar chart
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from chart_helpers import (
    add_indicator_threshold_lines,
    build_current_contributions_bar_figure,
    build_now_vs_crisis_vertical_figure,
)
from config import AppConfig, RunResults
from crisis_analysis import compute_indicator_crisis_similarity
from feature_builder import latest_value_and_date


def _simple_explainer(title: str, body: str) -> None:
    with st.expander(title):
        st.markdown(body)


def _badge(state: int) -> str:
    return {+1: "🟢 Bullish", 0: "🟡 Neutral", -1: "🔴 Bearish"}.get(int(state), "—")


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


UPDATE_SOURCE = {
    "REAL_YIELD_CPI": "REAL_YIELD_CPI",
    "CPI_YOY": "CPI_YOY",
    "USD_12M_CHG": "USD_TWEX_SPLICE",
    "CURVE_10Y_3M": "CURVE_10Y_3M",
    "DEFICIT_GDP": "DEFICIT_GDP",
    "REAL_YIELD_TIPS10": "REAL_YIELD_TIPS10",
    "HY_OAS": "HY_OAS",
    "QQQ_ABOVE_MA200": "QQQ_ABOVE_MA200",
    "QQQ_MA50_SLOPE_20D": "QQQ_MA50_SLOPE_20D",
    "MARKET_BREADTH_ABOVE_MA200": "MARKET_BREADTH_ABOVE_MA200",
    "GOLD_6M_RET": "GOLD_USD",
    "REALYIELD_3M_CHG": "REAL_YIELD_TIPS10",
    "STRESS_3M_CHG": "HY_OAS",
    "USD_3M_CHG": "USD_TWEX_SPLICE",
}

COMMON_MACRO_KEYS = ["REAL_YIELD_CPI", "USD_12M_CHG", "CURVE_10Y_3M"]
GOLD_ONLY_KEYS = ["CPI_YOY", "DEFICIT_GDP", "REAL_YIELD_TIPS10"]
MARKET_ONLY_KEYS = [
    "HY_OAS", "QQQ_ABOVE_MA200", "QQQ_MA50_SLOPE_20D", "MARKET_BREADTH_ABOVE_MA200"
]

CONTRIB_COLS = [
    "CONTRIB_REAL_YIELD", "CONTRIB_USD", "CONTRIB_INFL", "CONTRIB_STRESS",
    "CONTRIB_LIQ", "CONTRIB_ETF", "CONTRIB_CB", "CONTRIB_GEO", "Z_GOLD_MOM",
]
CONTRIB_LABELS = {
    "CONTRIB_REAL_YIELD": "Real Yield",
    "CONTRIB_USD": "USD",
    "CONTRIB_INFL": "Inflation",
    "CONTRIB_STRESS": "Stress",
    "CONTRIB_LIQ": "Liquidity",
    "CONTRIB_ETF": "ETF",
    "CONTRIB_CB": "Central Bank",
    "CONTRIB_GEO": "Geopolitics",
    "Z_GOLD_MOM": "Momentum",
}


# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────

def render_kpi_strip(results: RunResults, cfg: AppConfig) -> None:
    latest = results.res.iloc[-1]
    prev = results.res.iloc[-2] if len(results.res) > 1 else latest
    delta = float(latest["SIGNAL"] - prev["SIGNAL"])
    fair_gap_now = (
        float(results.res_base["FAIR_GOLD_GAP_PCT"].dropna().iloc[-1])
        if "FAIR_GOLD_GAP_PCT" in results.res_base.columns
        and results.res_base["FAIR_GOLD_GAP_PCT"].dropna().shape[0] > 0
        else np.nan
    )

    def _sl(col: str) -> float:
        if col in results.res_base.columns:
            d = results.res_base[col].dropna()
            return float(d.iloc[-1]) if not d.empty else np.nan
        return np.nan

    if cfg.mode == "Market Acceleration (fast)":
        rsi_col = "RSI_14_ASOF" if "RSI_14_ASOF" in results.res_base.columns else "RSI_14"
        rsi_last = _sl(rsi_col)
        rsi_z_last = _sl("RSI_Z_60")
        rsi_slope_last = _sl("RSI_SLOPE_3M")
        rsi_d = _sl("RSI_14D")
        rsi_d_z = _sl("RSI_14D_Z_1Y")
        rsi_d_slope = _sl("RSI_14D_SLOPE_1M")

        k1, k2, k3, k4, _ = st.columns([1.2, 1.2, 1, 1, 1])
        with k1:
            st.metric("Weighted Signal", f'{latest["SIGNAL"]:.2f}', f"{delta:+.2f}")
        with k2:
            st.metric("Regime", str(latest.get("REGIME", "—")))
        with k3:
            st.markdown("**RSI (Monthly)**")
            m1, m2, m3 = st.columns([1, 1.2, 1])
            m1.metric("RSI 14M", "—" if pd.isna(rsi_last) else f"{rsi_last:.0f}")
            m2.metric("z-score (5Y)", "—" if pd.isna(rsi_z_last) else f"{rsi_z_last:+.2f}")
            m3.metric("slope (3M)", "—" if pd.isna(rsi_slope_last) else f"{rsi_slope_last:+.0f}")
            st.markdown("**RSI (Daily)**")
            d1, d2, d3 = st.columns(3)
            d1.metric("RSI 14D", "—" if pd.isna(rsi_d) else f"{rsi_d:.0f}")
            d2.metric("vs 1Y (z)", "—" if pd.isna(rsi_d_z) else f"{rsi_d_z:+.2f}")
            d3.metric("change (1M)", "—" if pd.isna(rsi_d_slope) else f"{rsi_d_slope:+.0f}")
        with k4:
            st.metric("Fair Gold Gap", "—" if pd.isna(fair_gap_now) else f"{fair_gap_now:+.1f}%")
        st.caption(f"Core data through: {str(results.res.index.max().date())}")

        # RSI & Acceleration narrative
        regime_now = str(latest.get("REGIME", "—"))
        up_long = not pd.isna(rsi_last) and rsi_last >= 80
        up_short = not pd.isna(rsi_d) and rsi_d >= 70
        dn_long = not pd.isna(rsi_last) and rsi_last <= 30
        dn_short = not pd.isna(rsi_d) and rsi_d <= 30

        if "Bull" in regime_now or "Positive" in regime_now:
            if up_long and up_short:
                combo = "Acceleration is positive, but both RSIs are high. Momentum is strong, yet a short pause would be normal."
            elif up_long:
                combo = "Acceleration is positive. Long-term momentum is very strong, while short-term is not overheated."
            elif up_short:
                combo = "Acceleration is positive and short-term momentum is hot. The move may be fast in the short run."
            else:
                combo = "Acceleration is positive and RSI is not extreme. That's usually a healthy trend setup."
        elif "Headwind" in regime_now or "Vulnerable" in regime_now:
            if dn_long and dn_short:
                combo = "Acceleration is negative and both RSIs are very low. Selling may be exhausted, so a rebound is possible."
            elif dn_short:
                combo = "Acceleration is negative and short-term RSI is very low. That can happen near short-term bottoms."
            else:
                combo = "Acceleration is negative and RSI is not deeply oversold. Pressure may continue."
        else:
            if up_long and up_short:
                combo = "Acceleration is mixed, but RSI is high in both timeframes. This often cools off."
            elif dn_long and dn_short:
                combo = "Acceleration is mixed, but RSI is very low in both timeframes. This can happen near bottoms."
            else:
                combo = "Acceleration is mixed and RSI is not extreme."
        st.info(f"RSI & Acceleration: {combo}")

    elif cfg.mode == "Crisis Similarity (template)":
        from crisis_analysis import (
            build_dynamic_crisis_playbook,
            get_fit_color_label,
            get_historical_asset_playbook,
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Gold Signal", f'{latest["SIGNAL"]:.2f}', f"{delta:.2f}")
        c2.metric("Gold Regime", str(latest.get("REGIME", "—")))
        c3.metric("Market Regime", results.market_regime_now)
        fit_pct = results.crisis_fit.get("fit_pct", np.nan)
        coverage_used = results.crisis_fit.get("coverage_used", 0)
        coverage_total = results.crisis_fit.get("coverage_total", 0)
        fit_label = "—" if pd.isna(fit_pct) else get_fit_color_label(fit_pct)
        c4.metric("Crisis Fit", "—" if pd.isna(fit_pct) else f"{fit_pct:.0f}%", fit_label)
        c5.metric("Coverage", f"{coverage_used}/{coverage_total}")
        c6.metric("Core data through", str(results.res.index.max().date()))

        if not pd.isna(fit_pct):
            fit_text = (
                "strong resemblance to the selected historical crisis template." if fit_pct >= 80 else
                "moderate resemblance to the selected historical crisis template." if fit_pct >= 60 else
                "partial resemblance to the selected historical crisis template." if fit_pct >= 40 else
                "weak resemblance to the selected historical crisis template." if fit_pct >= 20 else
                "very different from the selected historical crisis template."
            )
            playbook_text = build_dynamic_crisis_playbook(
                crisis_name=cfg.crisis, fit_score=fit_pct,
                latest_row=latest, core_keys=results.core_keys, labels_map=results.labels,
            )
            asset_text = get_historical_asset_playbook(crisis_name=cfg.crisis, fit_score=fit_pct)
            st.info(
                "Crisis Fit interpretation: " + fit_text + "\n\n"
                "**Investor playbook (driver-based):**\n" + playbook_text + "\n\n"
                "**Historical asset playbook:** " + asset_text
            )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Gold Signal", f'{latest["SIGNAL"]:.2f}', f"{delta:.2f}")
        c2.metric("Gold Regime", str(latest.get("REGIME", "—")))
        c3.metric("Market Regime", results.market_regime_now)
        c4.metric("Preset Window", "—")
        c5.metric("Core data through", str(results.res.index.max().date()))


# ─────────────────────────────────────────────────────────────────────────────
# ACCELERATION CHARTS (Tab 1 body — Market Acceleration mode only)
# ─────────────────────────────────────────────────────────────────────────────

def render_acceleration_charts(results: RunResults, cfg: AppConfig) -> None:
    _simple_explainer("What this tab means in simple words", """
This tab is the **gold-focused dashboard**.
Combines slow macro signals, acceleration, RSI, and price charts.

**Fair Gold Gap guide:** >+15% = expensive | +5–15% = late bull | -5–+5% = fair value | -5–-15% = undervalued | <-15% = deep undervaluation
    """)

    rb = results.res_base
    rsi14d_daily = rb.attrs.get("RSI_14D_DAILY", None)
    gold_daily = rb.attrs.get("gold_daily", None)
    rsi14m_raw = rb.attrs.get("RSI_14_ASOF_RAW", None)

    if isinstance(rsi14m_raw, pd.Series) and not rsi14m_raw.dropna().empty:
        rsi14m = rsi14m_raw.dropna().sort_index()
        rsi14m_label, rsi14m_is_asof = "RSI (14M, as-of today)", True
    else:
        rsi14m = rb["RSI_14"].dropna().sort_index() if "RSI_14" in rb.columns else pd.Series(dtype=float)
        rsi14m_label, rsi14m_is_asof = "RSI (14M, month-end)", False

    rsi14d_full = rsi14d_daily.dropna().sort_index() if isinstance(rsi14d_daily, pd.Series) else pd.Series(dtype=float)
    gold_d_full = gold_daily.dropna().sort_index() if isinstance(gold_daily, pd.Series) else pd.Series(dtype=float)
    gold_m_full = rb["GOLD_USD"].dropna().sort_index() if "GOLD_USD" in rb.columns else pd.Series(dtype=float)

    # Determine window
    if cfg.lookback is None:
        start_dt = None
    else:
        ed = (
            gold_d_full.index.max() if not gold_d_full.empty else
            (rsi14d_full.index.max() if not rsi14d_full.empty else
             (rsi14m.index.max() if not rsi14m.empty else None))
        )
        start_dt = pd.Timestamp(ed) - pd.DateOffset(months=int(cfg.lookback)) if ed is not None else None

    def _w(s):
        if s.empty or start_dt is None:
            return s
        return s.loc[s.index >= start_dt]

    rsi14d_win = _w(rsi14d_full)
    gold_d_win = _w(gold_d_full)
    rsi14m_win = _w(rsi14m)
    gold_m_win = _w(gold_m_full)

    fair_d_full = rb.attrs.get("fair_gold_daily", pd.Series(dtype=float))
    if not isinstance(fair_d_full, pd.Series):
        fair_d_full = pd.Series(dtype=float)
    fair_m_full = rb["FAIR_GOLD"].dropna().sort_index() if "FAIR_GOLD" in rb.columns else pd.Series(dtype=float)
    fair_d_win = _w(fair_d_full.dropna().sort_index())
    fair_m_win = _w(fair_m_full)

    # Extend monthly series to today's daily close
    if not gold_d_win.empty and not gold_m_win.empty:
        gml = gold_m_win.copy()
        gml.loc[gold_d_win.index.max()] = float(gold_d_win.iloc[-1])
        gold_m_win = gml.sort_index()
    if not fair_d_win.empty and not fair_m_win.empty:
        fml = fair_m_win.copy()
        fml.loc[fair_d_win.index.max()] = float(fair_d_win.iloc[-1])
        fair_m_win = fml.sort_index()

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    left, mid, right = st.columns(3)

    with left:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        if not rsi14d_win.empty:
            ax.plot(rsi14d_win.index, rsi14d_win.values, linewidth=1.5, label="RSI (14D)")
        if not rsi14m_win.empty:
            ax.plot(rsi14m_win.index, rsi14m_win.values, linewidth=1.2,
                    linestyle="--" if rsi14m_is_asof else "-", label=rsi14m_label)
        ax.axhline(70, linestyle="--", linewidth=1.0, alpha=0.7, label="70")
        ax.axhline(30, linestyle="--", linewidth=1.0, alpha=0.7, label="30")
        ax.axhline(85, linestyle=":", linewidth=1.0, alpha=0.6, color="red", label="85")
        ax.axhline(90, linestyle=":", linewidth=1.0, alpha=0.6, color="red", label="90")
        ax.set_title("RSI (Daily + Monthly)", fontsize=11)
        ax.set_ylabel("RSI")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)

    with mid:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        if not gold_d_win.empty:
            ax.plot(gold_d_win.index, gold_d_win.values, linewidth=1.5, label="Gold (daily)")
        if not gold_m_win.empty:
            ax.plot(gold_m_win.index, gold_m_win.values, marker="o", markersize=3,
                    linewidth=1.2, label="Gold (month-end)")
        if not fair_d_win.empty:
            ax.plot(fair_d_win.index, fair_d_win.values, linewidth=1.3,
                    linestyle="--", color="tab:purple", label="Fair Gold (daily-aligned)")
        if not fair_m_win.empty:
            ax.plot(fair_m_win.index, fair_m_win.values, linewidth=1.0,
                    linestyle=":", color="tab:purple", label="Fair Gold (month-end)")
        ax.set_title("Gold Price (Daily + Month-end)", fontsize=11)
        ax.set_ylabel("USD")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)

    with right:
        gap_full = rb["FAIR_GOLD_GAP_PCT"].dropna().sort_index() if "FAIR_GOLD_GAP_PCT" in rb.columns else pd.Series(dtype=float)
        gap_win = _w(gap_full)
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        if not gap_win.empty:
            ax.plot(gap_win.index, gap_win.values, color="purple", linewidth=1.5, label="Fair Gold Gap %")
            ax.axhline(0, color="black", lw=1)
            ax.axhline(10, color="red", linestyle="--", linewidth=1)
            ax.axhline(-10, color="green", linestyle="--", linewidth=1)
            ymax, ymin = float(gap_win.max()), float(gap_win.min())
            ax.axhspan(10, max(10, ymax), color="red", alpha=0.08)
            ax.axhspan(min(-10, ymin), -10, color="green", alpha=0.08)
        ax.set_title("Gold vs Fair Value (%)")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)

    # ── Row 2: Fair Gold Driver Contributions ─────────────────────────────────
    avail_contrib = [c for c in CONTRIB_COLS if c in rb.columns]
    if avail_contrib:
        contrib_df = _w(rb[avail_contrib].copy()).dropna(how="all")
        if not contrib_df.empty:
            c_left, c_right = st.columns(2)
            plot_cols = [c for c in CONTRIB_COLS if c in contrib_df.columns]
            with c_left:
                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                ys = [contrib_df[c].fillna(0.0).values for c in plot_cols]
                lbls = [CONTRIB_LABELS.get(c, c) for c in plot_cols]
                ax.stackplot(contrib_df.index, ys, labels=lbls, alpha=0.85)
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.set_title("Fair Gold Driver Contributions", fontsize=11)
                ax.set_ylabel("Contribution to Fair Gold Z")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=7)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                fig.autofmt_xdate(rotation=30, ha="right")
                st.pyplot(fig, use_container_width=True)
            with c_right:
                lc = contrib_df.iloc[-1].fillna(0.0)
                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                vals = lc.values
                names = [CONTRIB_LABELS.get(c, c) for c in lc.index]
                ax.bar(names, vals)
                for i, v in enumerate(vals):
                    ax.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.2f}", ha="center", fontsize=8)
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.set_title("Latest Fair Gold Contribution Breakdown", fontsize=11)
                ax.set_ylabel("Contribution to Fair Gold Z")
                ax.grid(True, alpha=0.3, axis="y")
                plt.xticks(rotation=30, ha="right")
                st.pyplot(fig, use_container_width=True)

    # ── Row 3: Intraday RSI + Intraday Gold (optional) ─────────────────────────
    if cfg.show_intraday_rsi:
        rsi_intra = rb.attrs.get("RSI_14_INTRA", None)
        gold_intra = rb.attrs.get("gold_intraday", None)

        if not (isinstance(rsi_intra, pd.Series) and not rsi_intra.dropna().empty) or \
           not (isinstance(gold_intra, pd.Series) and not gold_intra.dropna().empty):
            st.info("Intraday view unavailable (market closed, rate-limited, or no intraday data).")
        else:
            rsi_intra = rsi_intra.dropna().sort_index()
            gold_intra = gold_intra.dropna().sort_index()
            end_i = min(rsi_intra.index.max(), gold_intra.index.max())
            start_i = end_i - pd.Timedelta(days=int(cfg.intraday_lookback_days))
            ri_win = rsi_intra.loc[(rsi_intra.index >= start_i) & (rsi_intra.index <= end_i)]
            gi_win = gold_intra.loc[(gold_intra.index >= start_i) & (gold_intra.index <= end_i)]

            i_left, i_right = st.columns(2)
            with i_left:
                last_rsi = float(ri_win.iloc[-1]) if not ri_win.empty else np.nan
                st.markdown(
                    f"### Intraday RSI (14) - Gold — {cfg.intraday_interval}  \n"
                    + (f"Latest: **{last_rsi:.1f}**" if not pd.isna(last_rsi) else "")
                )
                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                ax.plot(ri_win.index, ri_win.values, linewidth=1.2,
                        label=f"RSI (14, {cfg.intraday_interval})")
                raw_asof = rb.attrs.get("RSI_14_ASOF_RAW", None)
                if isinstance(raw_asof, pd.Series) and not raw_asof.dropna().empty:
                    rv = float(raw_asof.dropna().iloc[-1])
                    ax.axhline(rv, linestyle="--", linewidth=1.4, alpha=0.85,
                               color="tab:orange", label=f"Monthly RSI as-of-today ({rv:.1f})")
                ax.axhline(70, linestyle="--", linewidth=1.0, alpha=0.7, label="70")
                ax.axhline(30, linestyle="--", linewidth=1.0, alpha=0.7, label="30")
                ax.axhline(85, linestyle=":", linewidth=1.0, alpha=0.6, color="red", label="85")
                ax.axhline(90, linestyle=":", linewidth=1.0, alpha=0.6, color="red", label="90")
                ax.set_ylabel("RSI")
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
                fig.autofmt_xdate(rotation=30, ha="right")
                st.pyplot(fig, use_container_width=True)

            with i_right:
                last_px = float(gi_win.iloc[-1]) if not gi_win.empty else np.nan
                st.markdown(
                    f"### Intraday - Gold — {cfg.intraday_interval}  \n"
                    + (f"Latest: **{last_px:,.2f}**" if not pd.isna(last_px) else "")
                )
                fair_intra = rb.attrs.get("fair_gold_intraday", None)
                if isinstance(fair_intra, pd.Series) and not fair_intra.dropna().empty:
                    fi_win = fair_intra.dropna().sort_index()
                    fi_win = fi_win.loc[(fi_win.index >= start_i) & (fi_win.index <= end_i)]
                else:
                    fi_win = pd.Series(dtype=float)

                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                ax.plot(gi_win.index, gi_win.values, linewidth=1.2,
                        label=f"Gold ({cfg.intraday_interval})")
                if not fi_win.empty:
                    ax.plot(fi_win.index, fi_win.values, linewidth=1.2,
                            linestyle="--", color="tab:purple", label="Fair Gold")
                ax.set_ylabel("USD")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
                fig.autofmt_xdate(rotation=30, ha="right")
                st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR CARDS
# ─────────────────────────────────────────────────────────────────────────────

def render_indicator_cards(results: RunResults, cfg: AppConfig) -> tuple[bool, bool]:
    latest = results.res.iloc[-1]
    st.markdown("### Indicators (latest)")

    def render_group(group_title: str, keys_to_show: list) -> None:
        shown = [c for c in keys_to_show if c in results.labels]
        if not shown:
            return
        st.markdown(f"#### {group_title}")
        for row in _chunked(shown, 3):
            cards = st.columns(3)
            for i, col in enumerate(row):
                with cards[i]:
                    gold_scored = col in results.core_keys
                    market_scored = col in results.core_keys_market if results.core_keys_market else False
                    is_scored = gold_scored or market_scored

                    gold_state = int(latest.get(col + "_STATE", 0)) if gold_scored else None
                    gold_contrib = float(latest.get(col + "_CONTRIB", 0.0)) if gold_scored else None
                    market_state = (
                        int(results.latest_market.get(col + "_STATE", 0))
                        if (results.latest_market is not None and market_scored) else None
                    )
                    market_contrib = (
                        float(results.latest_market.get(col + "_CONTRIB", 0.0))
                        if (results.latest_market is not None and market_scored) else None
                    )

                    st.markdown(f"**{results.labels[col]}**")
                    if not is_scored:
                        st.caption("Context only (not scored in this mode)")

                    val = latest.get(col, np.nan)
                    src_col = UPDATE_SOURCE.get(col, col)
                    source_df = results.res_base if src_col in results.res_base.columns else results.res
                    if src_col not in source_df.columns:
                        last_val, last_dt = np.nan, "—"
                    else:
                        last_val, last_dt = latest_value_and_date(source_df, src_col)

                    if pd.isna(val):
                        st.write(f"Latest: **{last_val:.2f}**" if pd.notna(last_val) else "Latest: —")
                        st.caption(f"Updated: {last_dt} (stale)")
                    else:
                        if col == "QQQ_ABOVE_MA200":
                            st.write(f"Latest: **{'Yes' if float(val) >= 0.5 else 'No'}**")
                        else:
                            st.write(f"Latest: **{val:.2f}**")
                        st.caption(f"Updated: {last_dt}")

                    if col == "HY_OAS":
                        hyd = "enabled" if cfg.include_hy else "disabled"
                        st.caption(f"Used in: Market model & Gold model (optional — {hyd})")

                    if group_title == "Common Macro Indicators":
                        st.write(f"Gold State: **{_badge(gold_state)}**" if gold_state is not None else "Gold State: —")
                        st.write(f"Market State: **{_badge(market_state)}**" if market_state is not None else "Market State: —")
                        st.write(f"Gold Contrib: **{gold_contrib:+.2f}**" if gold_contrib is not None else "Gold Contrib: —")
                        st.write(f"Market Contrib: **{market_contrib:+.2f}**" if market_contrib is not None else "Market Contrib: —")
                    else:
                        ss = gold_state if gold_state is not None else market_state
                        sc = gold_contrib if gold_contrib is not None else market_contrib
                        st.write(f"State: **{_badge(ss)}**" if ss is not None else "State: — (not scored)")
                        st.write(f"Contrib: **{sc:+.2f}**" if sc is not None else "Contrib: —")

                    # Sparkline / crisis comparison chart
                    if col in results.res.columns:
                        fig = None
                        use_crisis = (
                            cfg.mode == "Crisis Similarity (template)"
                            and cfg.win_start is not None
                            and cfg.win_end is not None
                            and col in results.res_base.columns
                        )
                        if use_crisis:
                            full_series = results.res_base[col].dropna().sort_index()
                            current_window = full_series if cfg.lookback is None else full_series.tail(int(cfg.lookback))
                            crisis_window = full_series.loc[
                                pd.to_datetime(cfg.win_start):pd.to_datetime(cfg.win_end)
                            ].dropna()
                            sim_meta = compute_indicator_crisis_similarity(
                                df=results.res_base, current_row=latest,
                                win_start=cfg.win_start, win_end=cfg.win_end,
                                key=col, min_points=24,
                            )
                            fp = sim_meta.get("fit_pct", np.nan)
                            fl = sim_meta.get("label", "No data")
                            st.caption(
                                f"Similarity to {cfg.crisis}: {fp:.0f}% ({fl})"
                                if not pd.isna(fp)
                                else f"Similarity to {cfg.crisis}: unavailable ({fl})"
                            )
                            hib = results.dirs.get(col, None)
                            if current_window.dropna().shape[0] >= 3 and crisis_window.dropna().shape[0] >= 3:
                                fig = build_now_vs_crisis_vertical_figure(
                                    current_series=current_window, crisis_series=crisis_window,
                                    current_label="Now", crisis_label=f"{cfg.crisis}",
                                    title=results.labels.get(col, col),
                                    sim_meta=sim_meta, higher_is_bullish=hib,
                                )

                        if fig is None:
                            s = results.res[col]
                            spark = s if cfg.lookback is None else s.tail(cfg.lookback)
                            fig, ax = plt.subplots(figsize=(3.5, 3.2))
                            if col == "QQQ_ABOVE_MA200":
                                ax.step(spark.index, spark.values, where="post", linewidth=1.5)
                                ax.set_ylim(-0.1, 1.1)
                                ax.set_yticks([0, 1])
                                ax.set_yticklabels(["Below MA200", "Above MA200"])
                            else:
                                ax.plot(spark.index, spark.values, linewidth=1.2)
                            if cfg.show_indicator_thresholds and is_scored:
                                thr = (
                                    results.thresholds_market
                                    if (market_scored and not gold_scored)
                                    else results.thresholds
                                )
                                add_indicator_threshold_lines(ax, col, thr, cfg.mode)
                            ax.grid(True, linewidth=0.4, alpha=0.4)
                            plt.tight_layout()
                        st.pyplot(fig, clear_figure=True)

    render_group("Common Macro Indicators", COMMON_MACRO_KEYS)
    render_group("Gold-Specific Indicators", GOLD_ONLY_KEYS)
    render_group("Market-Specific Indicators", MARKET_ONLY_KEYS)

    # Triggers
    st.markdown("### Triggers (action layer)")
    sig = results.res["SIGNAL"]
    bull_now = sig.iloc[-cfg.persist:].ge(cfg.trig_hi).all()
    bear_now = sig.iloc[-cfg.persist:].le(cfg.trig_lo).all()
    tc1, tc2, tc3 = st.columns([1.2, 1.2, 2.2])
    tc1.metric("Bull trigger", "ON ✅" if bull_now else "OFF", f"{cfg.persist}m persistence")
    tc2.metric("Bear trigger", "ON ✅" if bear_now else "OFF", f"{cfg.persist}m persistence")
    if bull_now:
        tc3.success(f"Action: consider **increasing** gold exposure (Signal ≥ {cfg.trig_hi:.2f} for {cfg.persist} months).")
    elif bear_now:
        tc3.error(f"Action: consider **reducing** gold exposure (Signal ≤ {cfg.trig_lo:.2f} for {cfg.persist} months).")
    else:
        tc3.info("Action: no trigger. Maintain baseline / monitor drivers.")

    return bool(bull_now), bool(bear_now)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_history_panel(results: RunResults, cfg: AppConfig, theme) -> None:
    st.markdown("## History")
    _simple_explainer("What this section means in simple words", """
This section shows the model through time. Use it to see whether the signal was
strengthening, weakening, or changing direction, and whether gold moved in a similar way.
    """)

    res_plot = results.res.copy()
    try:
        end_dt = res_plot.index.max()
        if cfg.history_view == "Last 15y":
            res_plot = res_plot.loc[end_dt - pd.DateOffset(years=15):]
        elif cfg.history_view == "Last 5y":
            res_plot = res_plot.loc[end_dt - pd.DateOffset(years=5):]
        elif cfg.history_view == "Crisis window only" and cfg.win_start and cfg.win_end:
            res_plot = res_plot.loc[pd.to_datetime(cfg.win_start):pd.to_datetime(cfg.win_end)]
        elif cfg.history_view == "Crisis window ±5y" and cfg.win_start and cfg.win_end:
            s_dt = pd.to_datetime(cfg.win_start) - pd.DateOffset(years=5)
            e_dt = pd.to_datetime(cfg.win_end) + pd.DateOffset(years=5)
            res_plot = res_plot.loc[s_dt:e_dt]
    except Exception:
        pass

    # Row 1: Signal over time | Gold vs Signal (dual axis, with Market Signal)
    h1, h2 = st.columns(2)

    with h1:
        st.markdown("**Signal over time (with triggers)**")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(res_plot.index, res_plot["SIGNAL"].values, label="Gold Signal", color="tab:blue")
        if results.res_market is not None and not results.res_market.empty and "SIGNAL" in results.res_market.columns:
            mp = results.res_market.reindex(res_plot.index)
            ax.plot(mp.index, mp["SIGNAL"], label="Market Signal",
                    linestyle="--", color=theme.market_signal)
        ax.axhline(cfg.trig_hi, linestyle="--", color=theme.bull_color,
                   label=f"Bull trigger ({cfg.trig_hi:.2f})")
        ax.axhline(cfg.trig_lo, linestyle="--", color=theme.bear_color,
                   label=f"Bear trigger ({cfg.trig_lo:.2f})")
        ax.axhline(0.0, linewidth=0.8, color="black", label="Zero")
        ax.set_xlabel("Date")
        ax.set_ylabel("Signal (weighted sum)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with h2:
        st.markdown("**Gold vs Signal (if available)**")
        gold_col = next((c for c in ["GOLD_USD", "GC=F", "GLD"] if c in res_plot.columns), None)
        if not gold_col:
            st.warning("Gold series not available in this run.")
        else:
            plot_df = res_plot[[gold_col, "SIGNAL"]].dropna().sort_index()
            if plot_df.empty:
                st.warning("Gold vs Signal chart is empty.")
            else:
                fig, ax1 = plt.subplots(figsize=(7, 3.2))
                ax1.plot(plot_df.index, plot_df[gold_col].values,
                         linewidth=1.8, label="Gold (USD)", color="gold")
                ax2 = ax1.twinx()
                ax2.plot(plot_df.index, plot_df["SIGNAL"].values, linewidth=1.2,
                         linestyle="--", alpha=0.8, color="tab:blue", label="Gold Signal")
                # Add Market Signal on secondary axis
                if results.res_market is not None and not results.res_market.empty:
                    ms_r = results.res_market["SIGNAL"].reindex(plot_df.index)
                    if ms_r.dropna().shape[0] > 0:
                        ax2.plot(ms_r.index, ms_r.values, linewidth=1.0,
                                 linestyle=":", alpha=0.7,
                                 color=theme.market_signal, label="Market Signal")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Gold (USD)", color="goldenrod")
                ax2.set_ylabel("Signal", color="tab:blue")
                l1, lb1 = ax1.get_legend_handles_labels()
                l2, lb2 = ax2.get_legend_handles_labels()
                ax1.legend(l1 + l2, lb1 + lb2, loc="best", fontsize=8)
                ax1.grid(True, alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

    # Row 2: Contributions (stacked) | Current Indicator Influence (bar)
    h3, h4 = st.columns(2)
    contrib_cols_present = [c for c in res_plot.columns if c.endswith("_CONTRIB")]

    with h3:
        if contrib_cols_present:
            st.markdown("**Contributions (stacked by indicator)**")
            contrib = res_plot[contrib_cols_present].dropna(how="all")
            if not contrib.empty:
                fig, ax = plt.subplots(figsize=(7, 3.2))
                leg_labels = [
                    results.labels.get(c.replace("_CONTRIB", ""), c.replace("_CONTRIB", ""))
                    for c in contrib_cols_present
                ]
                ax.stackplot(contrib.index, contrib.T.values, labels=leg_labels, alpha=0.85)
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.set_xlabel("Date")
                ax.set_ylabel("Contribution to signal")
                ax.legend(loc="best", fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
        else:
            st.info("No contribution data available for this mode.")

    with h4:
        latest = results.res.iloc[-1]
        title = (
            "Current Indicator Influence (crisis score)"
            if cfg.mode == "Crisis Similarity (template)"
            else "Current Contribution by Indicator"
        )
        fig_bar = build_current_contributions_bar_figure(
            latest, results.core_keys, results.labels, title=title
        )
        if fig_bar is not None:
            st.markdown(f"**{title}**")
            st.pyplot(fig_bar, clear_figure=True)
        else:
            st.info("No contribution data available.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render(results: RunResults, cfg: AppConfig, theme) -> tuple[bool, bool]:
    """
    Main entry point for Tab 1.
    Returns (bull_now, bear_now) for use in the report section.
    """
    render_kpi_strip(results, cfg)

    # Acceleration-specific charts (only in Market Acceleration mode)
    if cfg.mode == "Market Acceleration (fast)":
        render_acceleration_charts(results, cfg)

    bull_now, bear_now = render_indicator_cards(results, cfg)
    render_history_panel(results, cfg, theme)
    return bull_now, bear_now
