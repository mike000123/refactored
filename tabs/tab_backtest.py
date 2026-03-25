"""
tabs/tab_backtest.py
--------------------
Tab 4: Walk-Forward Macro Backtest
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import (
    MacroBacktestConfig,
    build_portfolio_backtest,
    compute_backtest_analytics,
    run_macro_strategy_backtest,
)
from config import AppConfig, RunResults
from macro_fusion import build_macro_state
from macro_models import compute_accel_features
from model_runners import run_accel, run_market_structural, run_structural
from strategy_engine import decide_trade
from walkforward_mc_validation import (
    MCValidationConfig,
    run_walkforward_mc_validation,
    summarize_mc_validation,
)


def _simple_explainer(title: str, body: str) -> None:
    with st.expander(title):
        st.markdown(body)


# ── Decision rules ────────────────────────────────────────────────────────────

def _decision_from_rule(
    rule_name: str,
    ms,
    lg: pd.Series,
    lm: pd.Series,
    la: pd.Series,
    hist_base: pd.DataFrame,
    bt_price_col: str,
) -> dict:
    rule_name = str(rule_name)

    if rule_name == "Current fused strategy":
        if ms is None:
            return {"action": "HOLD", "confidence": 0.20, "position_size_pct": 0.0,
                    "reason": "Macro state unavailable in replay row"}
        dec = decide_trade(ticker=bt_price_col, macro_state=ms)
        pos_size = float(dec.position_size_pct)
        if pos_size <= 1.0:
            pos_size *= 100.0
        return {"action": dec.action, "confidence": dec.confidence,
                "position_size_pct": pos_size, "reason": dec.reason}

    if rule_name == "Structural-only":
        sig = pd.to_numeric(pd.Series([lg.get("SIGNAL")]), errors="coerce").iloc[0]
        reg = str(lg.get("REGIME") or "")
        if pd.notna(sig) and float(sig) >= 0.60:
            return {"action": "BUY", "confidence": 0.75, "position_size_pct": 100.0,
                    "reason": f"Structural regime bullish ({reg})"}
        if pd.notna(sig) and float(sig) >= 0.20:
            return {"action": "WATCH", "confidence": 0.60, "position_size_pct": 50.0,
                    "reason": f"Structural regime constructive ({reg})"}
        if pd.notna(sig) and float(sig) <= -0.60:
            return {"action": "SELL", "confidence": 0.75, "position_size_pct": 0.0,
                    "reason": f"Structural regime defensive ({reg})"}
        return {"action": "HOLD", "confidence": 0.45, "position_size_pct": 0.0,
                "reason": f"Structural regime mixed ({reg})"}

    if rule_name == "Acceleration-only":
        sig = pd.to_numeric(pd.Series([la.get("SIGNAL")]), errors="coerce").iloc[0]
        reg = str(la.get("REGIME") or "")
        if pd.notna(sig) and float(sig) >= 0.60:
            return {"action": "BUY", "confidence": 0.72, "position_size_pct": 100.0,
                    "reason": f"Acceleration bullish ({reg})"}
        if pd.notna(sig) and float(sig) >= 0.20:
            return {"action": "WATCH", "confidence": 0.58, "position_size_pct": 50.0,
                    "reason": f"Acceleration constructive ({reg})"}
        if pd.notna(sig) and float(sig) <= -0.60:
            return {"action": "SELL", "confidence": 0.72, "position_size_pct": 0.0,
                    "reason": f"Acceleration headwind ({reg})"}
        return {"action": "HOLD", "confidence": 0.45, "position_size_pct": 0.0,
                "reason": f"Acceleration mixed ({reg})"}

    if rule_name == "RSI-based gold rule":
        rsi_val = np.nan
        for c in ["RSI_14_ASOF", "RSI_14", "RSI_14D"]:
            if c in hist_base.columns and hist_base[c].dropna().shape[0]:
                rsi_val = float(hist_base[c].dropna().iloc[-1])
                break
        rsi_slope = float(hist_base["RSI_SLOPE_3M"].dropna().iloc[-1]) if (
            "RSI_SLOPE_3M" in hist_base.columns and hist_base["RSI_SLOPE_3M"].dropna().shape[0]
        ) else np.nan

        if pd.notna(rsi_val) and rsi_val <= 40:
            return {"action": "BUY", "confidence": 0.55, "position_size_pct": 100.0,
                    "reason": f"RSI oversold ({rsi_val:.1f})"}
        if pd.notna(rsi_val) and rsi_val >= 70:
            return {"action": "SELL", "confidence": 0.55, "position_size_pct": 0.0,
                    "reason": f"RSI overbought ({rsi_val:.1f})"}
        if pd.notna(rsi_val) and pd.notna(rsi_slope) and rsi_val < 50 and rsi_slope > 0:
            return {"action": "WATCH", "confidence": 0.50, "position_size_pct": 50.0,
                    "reason": f"RSI improving ({rsi_val:.1f})"}
        return {"action": "HOLD", "confidence": 0.40, "position_size_pct": 0.0, "reason": "RSI neutral"}

    if rule_name == "Always-long benchmark":
        return {"action": "BUY", "confidence": 1.00, "position_size_pct": 100.0,
                "reason": "Always invested benchmark"}

    return {"action": "HOLD", "confidence": 0.40, "position_size_pct": 0.0, "reason": "Fallback"}


def _prepare_bt_for_portfolio(bt_in: pd.DataFrame, trading_mode: str) -> pd.DataFrame:
    bt = bt_in.copy()
    if "signal_action" not in bt.columns and "decision" in bt.columns:
        bt["signal_action"] = bt["decision"]
    if "signal_confidence" not in bt.columns and "decision_confidence" in bt.columns:
        bt["signal_confidence"] = bt["decision_confidence"]
    bt["signal_action"] = bt["signal_action"].astype(str).str.upper().str.strip()
    if "position_size_pct" not in bt.columns:
        bt["position_size_pct"] = np.nan
    bt["position_size_pct"] = pd.to_numeric(bt["position_size_pct"], errors="coerce")
    mode = str(trading_mode).lower().strip()
    sell_mask = bt["signal_action"] == "SELL"
    if mode == "long_short":
        bt.loc[sell_mask & bt["position_size_pct"].isna(), "position_size_pct"] = 100.0
    else:
        bt.loc[sell_mask & bt["position_size_pct"].isna(), "position_size_pct"] = 0.0
    bt.loc[(bt["signal_action"] == "HOLD") & bt["position_size_pct"].isna(), "position_size_pct"] = 0.0
    bt.loc[(bt["signal_action"] == "WATCH") & bt["position_size_pct"].isna(), "position_size_pct"] = 0.0
    bt.loc[(bt["signal_action"] == "BUY") & bt["position_size_pct"].isna(), "position_size_pct"] = 100.0
    return bt


def _pct_from_start(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    first_valid = s.dropna()
    if first_valid.empty:
        return s * np.nan
    start_val = float(first_valid.iloc[0])
    if start_val == 0:
        return s * np.nan
    return (s / start_val - 1.0) * 100.0


# ── Main render ───────────────────────────────────────────────────────────────

def render(results: RunResults, cfg: AppConfig) -> None:
    st.subheader("Walk-Forward Macro Backtest")
    _simple_explainer("What this section means in simple words", """
This section is a **historical replay**. It takes the model back in time and asks:
"If I were living at that date, what would the platform have said?"

It is a **research tool**, not a fully realistic trading account simulation.
    """)

    # ── Settings UI ────────────────────────────────────────────────────────────
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        min_bt_date = pd.to_datetime(results.res_base.index.min()).date() if len(results.res_base.index) else datetime(1975, 1, 1).date()
        max_bt_date = pd.to_datetime(results.res_base.index.max()).date() if len(results.res_base.index) else datetime.today().date()
        default_bt_date = max(min_bt_date, datetime(2020, 1, 1).date())
        bt_start_dt = st.date_input("Start date", value=default_bt_date, min_value=min_bt_date, max_value=max_bt_date, key="bt_start_dt")
        bt_start = pd.Timestamp(bt_start_dt).date().isoformat()
    with b2:
        bt_step = int(st.slider("Replay step (months)", 1, 12, 1, 1, key="bt_step"))
    with b3:
        bt_fwd = int(st.slider("Forward horizon (months)", 1, 12, 12, 1, key="bt_fwd"))
    with b4:
        price_options = [c for c in ["GOLD_USD", "GLD"] if c in results.res_base.columns]
        if not price_options:
            st.warning("No gold price series available (GOLD_USD / GLD missing).")
            bt_price_col = None
        else:
            bt_price_col = st.selectbox("Price series", price_options, index=0, key="bt_price_col")
    with b5:
        bt_initial_capital = float(st.number_input("Initial investment", min_value=100.0, max_value=100_000_000.0, value=10_000.0, step=100.0, key="bt_initial_capital"))

    _, c2, c3, c4 = st.columns(4)
    with c4:
        annualize_by = st.selectbox("Analytics basis", ["Monthly", "Custom"], index=0, key="bt_annualize_by")
        periods_per_year = 12.0 if annualize_by == "Monthly" else max(1.0, 12.0 / max(float(bt_step), 1.0))

    st.markdown("### Strategy Rules")
    rule_options = ["Current fused strategy", "Structural-only", "Acceleration-only", "RSI-based gold rule", "Always-long benchmark"]
    sr1, sr2, sr3 = st.columns([1.2, 1.2, 1.0])
    with sr1:
        bt_primary_rule = st.selectbox("Primary strategy rule", rule_options, index=0, key="bt_primary_rule")
    with sr2:
        bt_compare_rule = st.selectbox("Comparison strategy rule", rule_options, index=3, key="bt_compare_rule")
    with sr3:
        compare_enabled = st.checkbox("Enable side-by-side rule comparison", value=True, key="bt_compare_enabled")

    st.markdown("#### Primary execution settings")
    p1, p2, p3 = st.columns(3)
    with p1:
        bt_trading_mode = st.selectbox("Primary trading mode", ["long_short", "long_flat"], index=0, key="bt_trading_mode")
    with p2:
        bt_tc_bps = float(st.number_input("Primary transaction cost (bps)", min_value=0.0, max_value=500.0, value=10.0, step=1.0, key="bt_tc_bps"))
    with p3:
        bt_slip_bps = float(st.number_input("Primary slippage (bps)", min_value=0.0, max_value=500.0, value=5.0, step=1.0, key="bt_slip_bps"))

    compare_trading_mode, compare_tc_bps, compare_slip_bps = bt_trading_mode, bt_tc_bps, bt_slip_bps
    if compare_enabled:
        st.markdown("#### Comparison execution settings")
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            compare_trading_mode = st.selectbox("Comparison trading mode", ["long_flat", "long_short"], index=0 if bt_trading_mode == "long_flat" else 1, key="compare_trading_mode")
        with cp2:
            compare_tc_bps = float(st.number_input("Comparison transaction cost (bps)", min_value=0.0, max_value=500.0, value=max(bt_tc_bps, 10.0), step=1.0, key="compare_tc_bps"))
        with cp3:
            compare_slip_bps = float(st.number_input("Comparison slippage (bps)", min_value=0.0, max_value=500.0, value=max(bt_slip_bps, 5.0), step=1.0, key="compare_slip_bps"))

    st.markdown("### Execution Realism")
    r1, r2, r3 = st.columns(3)
    bt_execution_lag = int(r1.number_input("Execution lag (steps)", 0, 6, 0, 1, key="bt_execution_lag"))
    bt_confirmation_steps = int(r2.number_input("Confirmation steps", 1, 6, 1, 1, key="bt_confirmation_steps"))
    bt_min_hold_steps = int(r3.number_input("Minimum hold (steps)", 0, 12, 0, 1, key="bt_min_hold_steps"))
    r4, r5, r6 = st.columns(3)
    bt_rebalance_mode = r4.selectbox("Rebalance mode", ["on_change", "every_signal", "threshold"], index=0, key="bt_rebalance_mode")
    bt_rebalance_threshold = float(r5.number_input("Rebalance threshold (%)", 0.0, 50.0, 2.5, 0.5, key="bt_rebalance_threshold"))
    bt_stop_loss_pct = float(r6.number_input("Stop-loss (%)", 0.0, 50.0, 5.0, 0.5, key="bt_stop_loss_pct"))
    r7, _, _ = st.columns(3)
    bt_take_profit_pct = float(r7.number_input("Take-profit (%)", 0.0, 100.0, 5.0, 0.5, key="bt_take_profit_pct"))

    st.markdown("### Historical Monte Carlo Validation")
    mv1, mv2, mv3, mv4 = st.columns(4)
    with mv1:
        wf_mc_enabled = st.checkbox("Enable MC validation", value=True, key="wf_mc_enabled")
    with mv2:
        wf_mc_hist_bars = int(st.number_input("MC history bars", 120, 2000, 500, 20, key="wf_mc_hist_bars"))
    with mv3:
        wf_mc_block = int(st.number_input("MC block size", 2, 20, 5, 1, key="wf_mc_block"))
    with mv4:
        wf_mc_sims = int(st.number_input("MC simulations", 200, 5000, 1000, 100, key="wf_mc_sims"))

    # ── Compute row factory ────────────────────────────────────────────────────
    def _compute_bt_row_factory(rule_name):
        def _compute_bt_row(hist_df: pd.DataFrame, dt: pd.Timestamp) -> dict:
            if hist_df.empty or len(hist_df.dropna(how="all")) < 36:
                return {"error": "Not enough history"}

            hist_accel = compute_accel_features(hist_df.copy())
            g_res, _, _, _, _ = run_structural(hist_df.copy(), results.weights_struct)
            m_res, _, _, _, _ = run_market_structural(hist_df.copy())
            a_res, _, _, _, _ = run_accel(hist_accel.copy(), cfg.accel_method, cfg.w_g, cfg.w_ry, cfg.w_st, cfg.w_u)

            lg = g_res.iloc[-1] if not g_res.empty else pd.Series(dtype=float)
            lm = m_res.iloc[-1] if not m_res.empty else pd.Series(dtype=float)
            la = a_res.iloc[-1] if not a_res.empty else pd.Series(dtype=float)

            ms = None
            macro_state_label = "—"
            macro_score_val = np.nan
            macro_error = None
            try:
                ms = build_macro_state(
                    asset=bt_price_col,
                    as_of_date=str(pd.Timestamp(dt).date()),
                    gold_signal=lg.get("SIGNAL"),
                    gold_regime=lg.get("REGIME"),
                    market_signal=lm.get("SIGNAL"),
                    market_regime=lm.get("REGIME"),
                    accel_signal=la.get("SIGNAL"),
                    accel_regime=la.get("REGIME"),
                )
                macro_state_label = getattr(ms, "fused_state", "—")
                macro_score_val = getattr(ms, "fused_score", np.nan)
            except Exception as e:
                macro_error = str(e)

            dec = _decision_from_rule(rule_name, ms, lg, lm, la, hist_df, bt_price_col)
            reason_text = dec["reason"]
            if macro_error:
                reason_text = f"{reason_text} | macro_state_error: {macro_error}"

            return {
                "gold_signal": lg.get("SIGNAL"), "gold_regime": lg.get("REGIME"),
                "market_signal": lm.get("SIGNAL"), "market_regime": lm.get("REGIME"),
                "accel_signal": la.get("SIGNAL"), "accel_regime": la.get("REGIME"),
                "macro_score": macro_score_val, "macro_state": macro_state_label,
                "decision": dec["action"], "decision_confidence": dec["confidence"],
                "position_size_pct": dec["position_size_pct"],
                "reason": reason_text, "strategy_rule": rule_name,
            }
        return _compute_bt_row

    # ── Run primary backtest ───────────────────────────────────────────────────
    bt_cfg = MacroBacktestConfig(
        ticker=bt_price_col, start_date=bt_start, step=bt_step, price_col=bt_price_col,
        forward_return_months=bt_fwd, trading_mode=bt_trading_mode,
        transaction_cost_bps=bt_tc_bps, slippage_bps=bt_slip_bps,
    )

    bt_df = run_macro_strategy_backtest(
        results.res_base, compute_one_date=_compute_bt_row_factory(bt_primary_rule), config=bt_cfg,
    )
    bt_df["signal_action"] = bt_df["decision"] if "decision" in bt_df.columns else np.nan
    bt_df["signal_confidence"] = bt_df["decision_confidence"] if "decision_confidence" in bt_df.columns else np.nan

    # ── MC validation ──────────────────────────────────────────────────────────
    mc_validation_df = pd.DataFrame()
    mc_validation_summary: dict = {}
    if wf_mc_enabled and bt_price_col is not None:
        mc_cfg = MCValidationConfig(horizon_steps=int(bt_fwd), n_sims=int(wf_mc_sims), block_size=int(wf_mc_block), seed=7)
        mc_validation_df = run_walkforward_mc_validation(
            replay_df=bt_df, full_price=results.res_base[bt_price_col],
            benchmark_price=results.res_base["QQQ"] if "QQQ" in results.res_base.columns else None,
            horizon_months=int(bt_fwd), cfg=mc_cfg,
        )
        if mc_validation_df is not None and not mc_validation_df.empty:
            bt_df = mc_validation_df
            mc_validation_summary = summarize_mc_validation(mc_validation_df)

    # ── Portfolio simulation ───────────────────────────────────────────────────
    bt_df_for_port = _prepare_bt_for_portfolio(bt_df, bt_trading_mode)
    portfolio_df = build_portfolio_backtest(
        bt_df_for_port, initial_capital=float(bt_initial_capital), trading_mode=bt_trading_mode,
        transaction_cost_bps=float(bt_tc_bps), slippage_bps=float(bt_slip_bps),
        execution_lag_steps=bt_execution_lag, min_hold_steps=bt_min_hold_steps,
        confirmation_steps=bt_confirmation_steps, rebalance_mode=bt_rebalance_mode,
        rebalance_threshold_pct=float(bt_rebalance_threshold),
        stop_loss_pct=float(bt_stop_loss_pct), take_profit_pct=float(bt_take_profit_pct),
    )
    bt_stats = compute_backtest_analytics(portfolio_df, periods_per_year=float(periods_per_year)) if (
        portfolio_df is not None and not portfolio_df.empty
    ) else {}

    # ── Comparison run ─────────────────────────────────────────────────────────
    compare_portfolio_df = None
    compare_stats: dict = {}
    if compare_enabled:
        compare_cfg = MacroBacktestConfig(
            ticker=bt_price_col, start_date=bt_start, step=bt_step, price_col=bt_price_col,
            forward_return_months=bt_fwd, trading_mode=compare_trading_mode,
            transaction_cost_bps=compare_tc_bps, slippage_bps=compare_slip_bps,
        )
        compare_bt_df = run_macro_strategy_backtest(
            results.res_base, compute_one_date=_compute_bt_row_factory(bt_compare_rule), config=compare_cfg,
        )
        compare_bt_df["signal_action"] = compare_bt_df.get("decision", np.nan)
        compare_bt_df["signal_confidence"] = compare_bt_df.get("decision_confidence", np.nan)
        compare_bt_df_for_port = _prepare_bt_for_portfolio(compare_bt_df, compare_trading_mode)
        compare_portfolio_df = build_portfolio_backtest(
            compare_bt_df_for_port, initial_capital=float(bt_initial_capital),
            trading_mode=compare_trading_mode, transaction_cost_bps=float(compare_tc_bps),
            slippage_bps=float(compare_slip_bps), execution_lag_steps=bt_execution_lag,
            min_hold_steps=bt_min_hold_steps, confirmation_steps=bt_confirmation_steps,
            rebalance_mode=bt_rebalance_mode, rebalance_threshold_pct=float(bt_rebalance_threshold),
            stop_loss_pct=float(bt_stop_loss_pct), take_profit_pct=float(bt_take_profit_pct),
        )
        compare_stats = compute_backtest_analytics(compare_portfolio_df, periods_per_year=float(periods_per_year)) if (
            compare_portfolio_df is not None and not compare_portfolio_df.empty
        ) else {}

    # ── KPI strip ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Replay rows", int(bt_stats.get("rows", 0)))
    m2.metric("BUY hit rate", f"{bt_stats.get('buy_hit_rate_pct', np.nan):.1f}%" if pd.notna(bt_stats.get("buy_hit_rate_pct", np.nan)) else "—")
    m3.metric("Strategy total return", f"{bt_stats.get('total_return_pct', np.nan):.1f}%" if pd.notna(bt_stats.get("total_return_pct", np.nan)) else "—")
    m4.metric("Max drawdown", f"{bt_stats.get('max_drawdown_pct', np.nan):.1f}%" if pd.notna(bt_stats.get("max_drawdown_pct", np.nan)) else "—")

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Confirmed rows", int(bt_stats.get("confirmed_rows", 0)))
    n2.metric("Stop events", int(bt_stats.get("stop_events", 0)))
    n3.metric("Take-profit events", int(bt_stats.get("take_profit_events", 0)))
    n4.metric("Avg turnover", f"{bt_stats.get('avg_turnover_pct', np.nan):.2f}%" if pd.notna(bt_stats.get("avg_turnover_pct", np.nan)) else "—")

    # ── Comparison summary ─────────────────────────────────────────────────────
    if compare_enabled:
        st.markdown("### Side-by-Side Strategy Comparison")
        comp_rows = [
            {"Setup": f"Primary — {bt_primary_rule}", "Trading Mode": bt_trading_mode, "TC (bps)": bt_tc_bps, "Slip (bps)": bt_slip_bps,
             "Total Return (%)": bt_stats.get("total_return_pct", np.nan), "CAGR (%)": bt_stats.get("cagr_pct", np.nan),
             "Volatility (%)": bt_stats.get("volatility_pct", np.nan), "Sharpe-like": bt_stats.get("sharpe_like", np.nan),
             "Max DD (%)": bt_stats.get("max_drawdown_pct", np.nan)},
            {"Setup": f"Comparison — {bt_compare_rule}", "Trading Mode": compare_trading_mode, "TC (bps)": compare_tc_bps, "Slip (bps)": compare_slip_bps,
             "Total Return (%)": compare_stats.get("total_return_pct", np.nan), "CAGR (%)": compare_stats.get("cagr_pct", np.nan),
             "Volatility (%)": compare_stats.get("volatility_pct", np.nan), "Sharpe-like": compare_stats.get("sharpe_like", np.nan),
             "Max DD (%)": compare_stats.get("max_drawdown_pct", np.nan)},
        ]
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    # ── Equity curve chart ─────────────────────────────────────────────────────
    if portfolio_df is not None and not portfolio_df.empty:
        curve_df = portfolio_df[["as_of_date", "equity_curve", "benchmark_curve", "price"]].dropna(how="all").copy()
        curve_df = curve_df.rename(columns={"equity_curve": "Primary Strategy",
                                             "benchmark_curve": "Benchmark Portfolio",
                                             "price": "Benchmark Raw Price"})
        if compare_enabled and compare_portfolio_df is not None and not compare_portfolio_df.empty:
            comp_curve = compare_portfolio_df[["as_of_date", "equity_curve"]].dropna().copy()
            comp_curve = comp_curve.rename(columns={"equity_curve": "Comparison Strategy"})
            curve_df = curve_df.merge(comp_curve, on="as_of_date", how="outer")
        curve_df = curve_df.set_index("as_of_date").sort_index()

        st.markdown("### Strategy vs Benchmark")
        fig_curve = go.Figure()
        color_map = {"Primary Strategy": "#F5A623", "Comparison Strategy": "#00C1D4", "Benchmark Portfolio": "#34D399"}
        for col in ["Primary Strategy", "Comparison Strategy", "Benchmark Portfolio"]:
            if col in curve_df.columns:
                pct_series = _pct_from_start(curve_df[col])
                fig_curve.add_trace(go.Scatter(
                    x=curve_df.index, y=pct_series.values, mode="lines", name=col,
                    line=dict(color=color_map.get(col, "#888888"), width=2),
                ))
        fig_curve.update_layout(
            title="Cumulative % Return from start", height=380,
            xaxis=dict(showgrid=False), yaxis=dict(title="% return since start"),
            hovermode="x unified", margin=dict(l=20, r=20, t=48, b=20),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Drawdown
        if "drawdown_pct" in portfolio_df.columns:
            st.markdown("### Drawdown")
            dd_df = portfolio_df[["as_of_date", "drawdown_pct"]].dropna().set_index("as_of_date").sort_index()
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_df.index, y=dd_df["drawdown_pct"].values, mode="lines",
                fill="tozeroy", line=dict(color="#FB7185", width=1.5), name="Drawdown %",
            ))
            fig_dd.update_layout(height=260, margin=dict(l=20, r=20, t=32, b=20),
                                  yaxis=dict(title="Drawdown (%)"), xaxis=dict(showgrid=False))
            st.plotly_chart(fig_dd, use_container_width=True)

        # ── Historical MC Validation Results (full section) ──────────────────────
        if mc_validation_summary and mc_validation_summary.get("rows", 0) > 0:
            st.markdown("### Historical Monte Carlo Validation Results")
            with st.expander("What these metrics mean"):
                st.markdown("""
**Inside p10–p90 (%)** — share of replay dates where the actual outcome landed inside the forecast cone.
Ideally around 80%. Higher = cone is well calibrated.

**Below p10 / Above p90** — how often the actual result was worse/better than the pessimistic/optimistic tail.

**Directional hit (%)** — how often the forecast direction (up vs down) matched the actual direction.

**Median abs error vs p50** — average absolute gap between forecast median and actual outcome (in %).
Lower is better.
                """)

            mc5 = st.columns(5)
            mc5[0].metric("Inside p10–p90",
                f"{mc_validation_summary.get('inside_p10_p90_pct', np.nan):.1f}%"
                if pd.notna(mc_validation_summary.get("inside_p10_p90_pct")) else "—")
            mc5[1].metric("Below p10",
                f"{mc_validation_summary.get('below_p10_pct', np.nan):.1f}%"
                if pd.notna(mc_validation_summary.get("below_p10_pct")) else "—")
            mc5[2].metric("Above p90",
                f"{mc_validation_summary.get('above_p90_pct', np.nan):.1f}%"
                if pd.notna(mc_validation_summary.get("above_p90_pct")) else "—")
            mc5[3].metric("Directional hit",
                f"{mc_validation_summary.get('direction_hit_pct', np.nan):.1f}%"
                if pd.notna(mc_validation_summary.get("direction_hit_pct")) else "—")
            mc5[4].metric("Median abs error vs p50",
                f"{mc_validation_summary.get('median_abs_err_pct', np.nan):.2f}%"
                if pd.notna(mc_validation_summary.get("median_abs_err_pct")) else "—")

            # Forecast Cone vs Actual | Historical Cone Replay charts
            import matplotlib.pyplot as _plt
            cone_left, cone_right = st.columns(2)

            # Left: Forecast cone p10/p50/p90 over time vs actual
            mc_p10_col = "mc_p10_ret_pct"
            mc_p50_col = "mc_p50_ret_pct"
            mc_p90_col = "mc_p90_ret_pct"
            mc_act_col = "mc_actual_ret_pct"
            mc_date_col = "as_of_date"

            if mc_validation_df is not None and not mc_validation_df.empty:
                vdf = mc_validation_df.copy()
                vdf[mc_date_col] = pd.to_datetime(vdf[mc_date_col], errors="coerce")
                vdf = vdf.sort_values(mc_date_col).reset_index(drop=True)

                with cone_left:
                    st.markdown("**Forecast Cone vs Actual Outcome**")
                    with st.expander("What this chart means"):
                        st.markdown("""
Shows how the MC forecast cone (p10/p50/p90 forward return) evolved over the replay period,
compared to what actually happened. If the black actual line stays mostly between
the green (p90) and red (p10) dashed lines, the cone is well-calibrated.
                        """)
                    fig, ax = _plt.subplots(figsize=(7, 3.8))
                    if mc_act_col in vdf.columns:
                        ax.plot(vdf[mc_date_col], pd.to_numeric(vdf[mc_act_col], errors="coerce"),
                                color="black", linewidth=1.5, label="Actual")
                    if mc_p90_col in vdf.columns:
                        ax.plot(vdf[mc_date_col], pd.to_numeric(vdf[mc_p90_col], errors="coerce"),
                                color="green", linestyle="--", linewidth=1.2, label="p90")
                    if mc_p50_col in vdf.columns:
                        ax.plot(vdf[mc_date_col], pd.to_numeric(vdf[mc_p50_col], errors="coerce"),
                                color="dodgerblue", linestyle="--", linewidth=1.2, label="p50")
                    if mc_p10_col in vdf.columns:
                        ax.plot(vdf[mc_date_col], pd.to_numeric(vdf[mc_p10_col], errors="coerce"),
                                color="red", linestyle="--", linewidth=1.2, label="p10")
                    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
                    ax.set_ylabel(f"Forward return over {bt_fwd} bars (%)")
                    ax.legend(loc="best", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    _plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)

                # Right: Historical cone replay from the first available anchor
                with cone_right:
                    st.markdown("**Historical Cone Replay from Start Date**")
                    with st.expander("What this chart means"):
                        st.markdown("""
Picks the earliest replay date and shows the full MC cone (p10/p50/p90 price path)
from that anchor alongside the actual price path that followed.
This gives a concrete before/after view for one historical moment.
                        """)
                    # Pick earliest valid row with p10/p50/p90 price columns
                    price_p10 = "mc_p10_price"
                    price_p50 = "mc_p50_price"
                    price_p90 = "mc_p90_price"
                    anchor_col = "mc_start_price"
                    state_col = "mc_state"

                    has_price_cols = all(c in vdf.columns for c in [price_p10, price_p50, price_p90, anchor_col])

                    if has_price_cols and bt_price_col is not None and bt_price_col in results.res_base.columns:
                        # Find first row with valid cone
                        valid_rows = vdf.dropna(subset=[price_p10, price_p50, price_p90, anchor_col])
                        if not valid_rows.empty:
                            anchor_row = valid_rows.iloc[0]
                            anchor_dt = pd.Timestamp(anchor_row[mc_date_col])
                            anchor_price = float(anchor_row[anchor_col])
                            state_used = str(anchor_row.get(state_col, "—"))

                            # Actual price from anchor forward (bt_fwd steps)
                            px_full = pd.to_numeric(results.res_base[bt_price_col], errors="coerce").dropna().sort_index()
                            px_actual = px_full.loc[px_full.index >= anchor_dt].head(bt_fwd + 2)

                            st.caption(f"Anchor used: {anchor_dt.date()} | MC tactical state: {state_used}")

                            fig, ax = _plt.subplots(figsize=(7, 3.8))
                            if not px_actual.empty:
                                ax.plot(range(len(px_actual)), px_actual.values,
                                        color="black", linewidth=1.5, marker="o", markersize=3, label="Actual price")
                            # Cone endpoints: draw as straight dashed lines from anchor
                            steps = np.arange(bt_fwd + 1)
                            p10_end = float(anchor_row[price_p10])
                            p50_end = float(anchor_row[price_p50])
                            p90_end = float(anchor_row[price_p90])
                            ax.plot([0, bt_fwd], [anchor_price, p90_end], color="green",
                                    linestyle="--", linewidth=1.2, label="p90")
                            ax.plot([0, bt_fwd], [anchor_price, p50_end], color="dodgerblue",
                                    linestyle="--", linewidth=1.2, label="p50")
                            ax.plot([0, bt_fwd], [anchor_price, p10_end], color="red",
                                    linestyle="--", linewidth=1.2, label="p10")
                            # Fill cone
                            ax.fill_between([0, bt_fwd], [anchor_price, p10_end],
                                            [anchor_price, p90_end], alpha=0.12, color="dodgerblue")
                            ax.scatter([0], [anchor_price], color="gray", s=30, zorder=5)
                            ax.set_ylabel("Price")
                            ax.legend(loc="best", fontsize=8)
                            ax.grid(True, alpha=0.3)
                            _plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                    else:
                        # Fallback: just show the p10/p50/p90 ret columns over time
                        st.info("Full price-level cone replay not available (requires mc_p10_price / mc_start_price columns in validation output).")

        # Action performance summary
        st.markdown("### Action Performance Summary")
        fwd_cols = [c for c in portfolio_df.columns if c.startswith("fwd_") and c.endswith("m_ret_pct")]
        if fwd_cols:
            fwd_col = fwd_cols[0]
            summary_rows = []
            tmp = portfolio_df.copy()
            tmp["decision"] = tmp["decision"].astype(str)
            for act, grp in tmp.groupby("decision", dropna=False):
                vals = pd.to_numeric(grp[fwd_col], errors="coerce").dropna()
                summary_rows.append({
                    "Action": act, "Rows": int(len(grp)),
                    "Avg forward return (%)": float(vals.mean()) if not vals.empty else np.nan,
                    "Median forward return (%)": float(vals.median()) if not vals.empty else np.nan,
                    "Hit rate (%)": float((vals > 0).mean() * 100.0) if not vals.empty else np.nan,
                })
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        # Replay table
        st.markdown("### Replay Table")
        show_cols = [
            "as_of_date", "strategy_rule", "gold_regime", "market_regime", "accel_regime",
            "macro_state", "macro_score", "decision", "decision_confidence", "signal_action",
            "executed_trade", "position_before_label", "position_before_pct", "position_after_label",
            "position_after_pct", "hold_steps_in_position", "stop_triggered", "take_profit_triggered",
            "asset_period_return_pct", "strategy_period_return_pct", "equity_curve", "benchmark_curve",
            "drawdown_pct", "reason",
        ]
        show_cols = [c for c in show_cols if c in portfolio_df.columns]
        st.dataframe(portfolio_df[show_cols], width="stretch", hide_index=True)
