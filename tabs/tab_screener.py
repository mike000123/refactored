"""
tabs/tab_screener.py
--------------------
Tab 2: Intraday RSI Screener
Tab 3: Monte Carlo Analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import intraday_screener
import mc_simulator as mc
from config import AppConfig, RunResults
from feature_builder import rsi, zscore
from strategy_engine import decide_trade


def _simple_explainer(title: str, body: str) -> None:
    with st.expander(title):
        st.markdown(body)


# ── Tab 2: Intraday RSI Screener ──────────────────────────────────────────────

def render_screener(results: RunResults, cfg: AppConfig, refresh_token: int) -> None:
    with st.expander("### How to interpret this tab"):
        st.markdown("""
This section answers the question: **Is there a short-term trading setup forming?**

It analyzes short-term momentum signals using RSI and volatility regimes.

- **BUY / SELL** → actionable short-term trade signal
- **WATCH** → setup forming but not yet confirmed
- **NEUTRAL** → no meaningful tactical signal

This tab focuses on **entry timing**, not medium-term position decisions.
        """)

    intraday_screener.render_intraday_rsi_screener_tab(
        rsi_func=rsi,
        zscore_func=zscore,
        tickers=cfg.tickers_top10,
        refresh_token=refresh_token,
    )


# ── Tab 3: Monte Carlo ────────────────────────────────────────────────────────

def render_monte_carlo(results: RunResults, cfg: AppConfig, refresh_token: int) -> None:
    if cfg.mode == "Crisis Similarity (template)":
        st.info("Monte Carlo disabled in Crisis Similarity mode.")
        return

    st.subheader("Monte Carlo — Analysis")
    _simple_explainer("What this section means in simple words", """
This section explores **what could happen next**, not just what happened so far.

It creates many possible future price paths based on similar past market states.
The goal is to show a reasonable range of outcomes — a typical path, a weaker path, and a stronger path.
    """)

    with st.expander("### How to interpret these charts"):
        st.markdown("""
**1) General Direction** — Is the ticker's price usually rising or falling in this environment?
p50 slopes upward: In similar conditions, price usually moved higher.

**2) Risk / Volatility regime**
Wide p10–p90 band = high historical volatility; narrow = calmer conditions.

**3) Downside risk (p10)** — A realistic bad outcome that has happened before in similar conditions.

**4) Upside potential (p90)** — A realistic strong outcome that has happened before.

**Simulation type:** Tactical block-bootstrap Monte Carlo
        """)

    left_col, right_col = st.columns([1, 3], gap="large")

    with left_col:
        overlay = st.checkbox("Overlay view (compare tickers)", value=True)
        mc_steps = st.slider("Simulation horizon (bars)", 20, 250, 60, 10)
        mc_n2 = st.slider("Simulations", 500, 5000, 2000, 500)

        universe = list(dict.fromkeys((cfg.tickers_top10 or []) + ["GLD", "SPY", "QQQ"]))
        universe = [t for t in universe if isinstance(t, str) and t.strip()]

        if "mc_selected" not in st.session_state:
            st.session_state["mc_selected"] = ["GLD", "SPY"]

        selected = []
        for t in universe:
            if st.checkbox(t, value=(t in st.session_state["mc_selected"]), key=f"mc_cb_{t}"):
                selected.append(t)
        st.session_state["mc_selected"] = selected

    with right_col:
        if not selected:
            st.info("Select tickers on the left.")
            return

        close_d = intraday_screener.yf_multi_close_fixed_period(
            selected, interval="1d", period="2y", refresh_token=refresh_token
        )

        if close_d is None or close_d.empty:
            st.warning("No price data available for selected tickers.")
            return

        # QQQ for relative strength
        try:
            qqq_df = intraday_screener.yf_multi_close_fixed_period(
                ["QQQ"], interval="1d", period="2y", refresh_token=refresh_token
            )
            qqq_px = (
                qqq_df["QQQ"].dropna()
                if (qqq_df is not None and not qqq_df.empty and "QQQ" in qqq_df.columns)
                else pd.Series(dtype=float)
            )
        except Exception:
            qqq_px = pd.Series(dtype=float)

        latest = results.res.iloc[-1] if not results.res.empty else pd.Series(dtype=float)
        p50_overlay: dict[str, pd.Series] = {}
        mc_summary_rows = []

        for t in selected:
            px = close_d[t].dropna() if t in close_d.columns else pd.Series(dtype=float)
            if px.empty or px.shape[0] < 60:
                st.warning(f"{t}: not enough daily history.")
                continue

            r = rsi(px, period=14).dropna().reindex(px.index, method="ffill")
            slope = r.diff(5)
            ma200 = px.rolling(200).mean()
            above_ma200 = px > ma200
            ma50 = px.rolling(50).mean()
            ma50_slope = ma50.pct_change(20)
            vol = px.pct_change().rolling(20).std() * np.sqrt(252)
            vol_q75 = vol.dropna().quantile(0.75) if not vol.dropna().empty else np.nan
            vol_regime = pd.Series(np.where(vol > vol_q75, "HIGH", "LOW"), index=px.index)

            if not qqq_px.empty:
                rs_vs_qqq = (px.pct_change(63) - qqq_px.pct_change(63)).reindex(px.index)
            else:
                rs_vs_qqq = pd.Series(index=px.index, dtype=float)

            state = pd.Series(
                [
                    mc.classify_accel_state(
                        r.iloc[i],
                        slope.iloc[i],
                        bool(above_ma200.iloc[i]) if i < len(above_ma200) else None,
                        rs_vs_qqq.iloc[i] if i < len(rs_vs_qqq) else None,
                        ma50_slope.iloc[i] if i < len(ma50_slope) else None,
                        vol_regime.iloc[i] if i < len(vol_regime) else None,
                    )
                    for i in range(len(px))
                ],
                index=px.index,
                name="STATE",
            )

            paths, state_now = mc.monte_carlo_paths_by_tactical_state_block(
                price=px, state=state, horizon_steps=mc_steps, n_sims=mc_n2, block_size=5, seed=7,
            )

            if paths is None:
                st.warning(f"{t}: not enough history in current tactical state ({state_now}).")
                continue

            bands = mc.mc_percentiles(paths)
            p50_overlay[t] = bands["p50"].rename(t)

            if not overlay:
                st.markdown(f"### {t} (state: `{state_now}`)")
                st.line_chart(bands)

            start_price = float(px.iloc[-1])
            p10_end = float(bands["p10"].iloc[-1])
            p50_end = float(bands["p50"].iloc[-1])
            p90_end = float(bands["p90"].iloc[-1])

            p10_ret = (p10_end / start_price - 1.0) * 100.0
            p50_ret = (p50_end / start_price - 1.0) * 100.0
            p90_ret = (p90_end / start_price - 1.0) * 100.0
            prob_up = float((paths[:, -1] > start_price).mean()) * 100.0

            structural_regime_for_ticker = (
                str(latest.get("REGIME", "—"))
                if t.upper() == "GLD"
                else results.market_regime_now
            )

            decision = decide_trade(
                ticker=t,
                structural_regime=structural_regime_for_ticker,
                tactical_state=state_now,
                mc_typical_pct=p50_ret,
                mc_prob_higher_pct=prob_up,
                above_ma200=bool(above_ma200.iloc[-1]) if len(above_ma200.dropna()) else None,
            )

            mc_summary_rows.append({
                "Ticker": t,
                "Structural Regime": structural_regime_for_ticker,
                "State": state_now,
                "Decision": decision.action,
                "Confidence (%)": round(decision.confidence * 100, 1),
                "Position Size (%)": round(decision.position_size_pct * 100, 1),
                "Current Price": round(start_price, 2),
                f"Downside (p10, {mc_steps} bars)": round(p10_end, 2),
                f"Typical (p50, {mc_steps} bars)": round(p50_end, 2),
                f"Upside (p90, {mc_steps} bars)": round(p90_end, 2),
                "Downside %": round(p10_ret, 2),
                "Typical %": round(p50_ret, 2),
                "Upside %": round(p90_ret, 2),
                "Prob. Finish Higher (%)": round(prob_up, 1),
                "Reason": decision.reason,
            })

        if overlay:
            if not p50_overlay:
                st.warning("Nothing to overlay — no ticker had enough state history.")
            else:
                st.markdown("### Overlay (Median path only)")
                st.caption("Each line = typical (p50) simulated path under each ticker's current tactical state.")
                st.line_chart(pd.concat(p50_overlay.values(), axis=1))

        if mc_summary_rows:
            st.markdown("### Monte Carlo Summary Table")
            _simple_explainer("What this table means in simple words", """
This table puts the main Monte Carlo results for several tickers in one place.
It helps you compare which assets look more or less favorable under their current tactical setup.
            """)
            mc_df = pd.DataFrame(mc_summary_rows).sort_values("Prob. Finish Higher (%)", ascending=False)
            st.caption(f"Top Monte Carlo probability: **{mc_df.iloc[0]['Ticker']}**")
            st.dataframe(mc_df, width="stretch", hide_index=True)
