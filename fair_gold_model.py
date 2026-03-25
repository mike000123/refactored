"""
fair_gold_model.py
------------------
Rolling OLS fair-value model for gold.
No Streamlit dependency. Debug output uses logging instead of print().
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from feature_builder import robust_zscore

logger = logging.getLogger(__name__)


def compute_fair_gold_monthly(
    gold_m: pd.Series,
    real_yield: pd.Series,
    usd_idx: pd.Series,
    infl_proxy: pd.Series,
    stress_proxy: pd.Series,
    liquidity_proxy: pd.Series,
    etf_proxy: pd.Series,
    cb_proxy: pd.Series,
    vix_series: pd.Series,
    *,
    z_window: int = 36,
    fit_window_months: int = 84,
    min_fit_obs: int = 24,
) -> pd.DataFrame:
    """
    Rolling fitted Fair Gold model.

    For each month t:
      1) compute z-scored macro drivers
      2) fit WLS on trailing fit_window_months sample
      3) use fitted coefficients at t to estimate Fair Gold_t

    Returns a DataFrame with FAIR_GOLD, FAIR_GOLD_GAP_PCT, driver contributions,
    and rolling betas.
    """
    tmp = pd.concat(
        [
            pd.to_numeric(gold_m, errors="coerce").rename("GOLD_USD"),
            pd.to_numeric(real_yield, errors="coerce").rename("REAL_YIELD_INPUT"),
            pd.to_numeric(usd_idx, errors="coerce").rename("USD_INPUT"),
            pd.to_numeric(infl_proxy, errors="coerce").rename("INFL_INPUT"),
            pd.to_numeric(stress_proxy, errors="coerce").rename("STRESS_INPUT"),
            pd.to_numeric(liquidity_proxy, errors="coerce").rename("LIQ_INPUT"),
            pd.to_numeric(etf_proxy, errors="coerce").rename("ETF_INPUT"),
            pd.to_numeric(cb_proxy, errors="coerce").rename("CB_INPUT"),
        ],
        axis=1,
    ).sort_index()

    # ── Z-score drivers ──────────────────────────────────────────────────────
    tmp["Z_REAL_YIELD"] = robust_zscore(tmp["REAL_YIELD_INPUT"], window=z_window)
    tmp["Z_USD"] = robust_zscore(tmp["USD_INPUT"], window=z_window)
    infl_residual = tmp["INFL_INPUT"] - tmp["INFL_INPUT"].rolling(12, min_periods=6).mean()
    tmp["Z_INFL"] = robust_zscore(infl_residual, window=z_window)
    tmp["Z_STRESS"] = robust_zscore(tmp["STRESS_INPUT"], window=z_window)
    tmp["Z_LIQ"] = robust_zscore(tmp["LIQ_INPUT"], window=z_window)
    tmp["Z_ETF"] = robust_zscore(tmp["ETF_INPUT"], window=z_window)
    tmp["Z_CB"] = robust_zscore(tmp["CB_INPUT"], window=z_window)

    # ── Gold momentum ────────────────────────────────────────────────────────
    tmp["GOLD_RET_12M"] = tmp["GOLD_USD"].pct_change(12)
    tmp["Z_GOLD_MOM"] = robust_zscore(tmp["GOLD_RET_12M"], window=z_window).clip(-3, 3)

    # ── Orthogonalise momentum ───────────────────────────────────────────────
    X_orth_cols = [c for c in ["Z_REAL_YIELD", "Z_USD", "Z_INFL", "Z_STRESS",
                                "Z_LIQ", "Z_ETF", "Z_CB"] if c in tmp.columns]
    mom = tmp["Z_GOLD_MOM"]
    X_orth = tmp[X_orth_cols].copy()
    common_idx = mom.dropna().index.intersection(X_orth.dropna().index)
    if len(common_idx) > 30:
        X_mat = np.column_stack([np.ones(len(common_idx)),
                                  X_orth.loc[common_idx].values])
        y_mat = mom.loc[common_idx].values
        beta_orth, *_ = np.linalg.lstsq(X_mat, y_mat, rcond=None)
        mom_hat = X_mat @ beta_orth
        mom_resid = y_mat - mom_hat
        tmp.loc[common_idx, "Z_GOLD_MOM"] = mom_resid
        logger.debug("Momentum orthogonalized. Residual std: %.4f", np.std(mom_resid))

    # ── VIX / Geopolitics ────────────────────────────────────────────────────
    tmp["VIX"] = pd.to_numeric(vix_series, errors="coerce").reindex(tmp.index)
    tmp["GEO_LEVEL"] = tmp["VIX"].rolling(3, min_periods=1).mean()
    tmp["GEO_SHOCK"] = tmp["VIX"] - tmp["VIX"].rolling(6, min_periods=3).mean()
    tmp["GEO_INPUT"] = 0.6 * tmp["GEO_LEVEL"] + 0.4 * tmp["GEO_SHOCK"]
    tmp["Z_GEO"] = robust_zscore(tmp["GEO_INPUT"], window=z_window)

    # ── CB adjustments ───────────────────────────────────────────────────────
    tmp["Z_CB"] = tmp["Z_CB"] * 1.3
    tmp["CB_TREND"] = tmp["CB_INPUT"].rolling(6, min_periods=1).mean()
    tmp["Z_CB_TREND"] = robust_zscore(tmp["CB_TREND"], window=z_window)
    tmp["Z_LIQ_STRESS"] = tmp["Z_LIQ"] * tmp["Z_STRESS"]
    tmp["Z_CB_STRESS"] = tmp["Z_CB"] * np.maximum(tmp["Z_STRESS"], 0)
    tmp["Z_GEO_STRESS"] = tmp["Z_GEO"] * np.maximum(tmp["Z_STRESS"], 0)

    # ── Regression column sets ────────────────────────────────────────────────
    core_reg_cols = ["Z_REAL_YIELD", "Z_USD", "Z_INFL", "Z_STRESS", "Z_LIQ"]
    extra_reg_cols = ["Z_ETF", "Z_CB", "Z_CB_TREND", "Z_GEO", "Z_GEO_STRESS", "Z_GOLD_MOM"]
    regression_cols = core_reg_cols + extra_reg_cols

    # ── Initialise output columns ─────────────────────────────────────────────
    for col in ["FAIR_GOLD_Z", "FAIR_GOLD_LOG", "FAIR_GOLD",
                "CONTRIB_REAL_YIELD", "CONTRIB_USD", "CONTRIB_INFL", "CONTRIB_STRESS",
                "CONTRIB_LIQ", "CONTRIB_ETF", "CONTRIB_CB", "CONTRIB_GEO",
                "BETA_CB_TREND", "BETA_INTERCEPT", "BETA_RY", "BETA_USD",
                "BETA_INFL", "BETA_STRESS", "BETA_LIQ", "BETA_ETF", "BETA_CB",
                "BETA_GOLD_MOM", "ETF_INPUT_LATEST", "CB_INPUT_LATEST", "B0_PREV"]:
        tmp[col] = np.nan

    valid = tmp[["GOLD_USD"] + regression_cols].copy()

    # ── Rolling regression loop ───────────────────────────────────────────────
    for i in range(len(tmp)):
        end_idx = tmp.index[i]
        start_pos = max(0, i - fit_window_months + 1)
        cur = tmp.loc[end_idx, regression_cols]

        # Build window and require core columns
        win_full = valid.iloc[start_pos : i + 1].copy()
        regression_cols_available = [c for c in regression_cols if c in win_full.columns]
        base_cols = list(dict.fromkeys(
            ["GOLD_USD"] + regression_cols_available
        ))
        win2 = win_full[base_cols].dropna(subset=["GOLD_USD"] + regression_cols_available)

        if win2.shape[0] < min_fit_obs:
            continue

        X_cols = [c for c in regression_cols_available if c in win2.columns]
        y = np.log(win2["GOLD_USD"].astype(float).values)
        X = np.column_stack(
            [np.ones(len(win2))] + [win2[c].fillna(0.0).astype(float).values for c in X_cols]
        )

        # Exponential time-weighting (recent data matters more)
        n = len(y)
        weights_t = np.exp(np.linspace(-2.0, 0.0, n))
        sqrt_w = np.sqrt(weights_t)
        beta, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)

        # Smooth intercept
        b0_raw = beta[0]
        if i > 0:
            prev_idx = tmp.index[i - 1]
            prev_b0 = tmp.loc[prev_idx, "B0_PREV"] if "B0_PREV" in tmp.columns else np.nan
            b0 = 0.7 * prev_b0 + 0.3 * b0_raw if pd.notna(prev_b0) else b0_raw
        else:
            b0 = b0_raw
        tmp.loc[end_idx, "B0_PREV"] = b0

        betas = dict(zip(X_cols, beta[1:]))

        # Regime-aware stress activation
        if "Z_STRESS" in betas:
            z_stress = cur.get("Z_STRESS", 0.0)
            beta_stress = max(betas["Z_STRESS"], 0.0)
            if abs(z_stress) < 0.75:
                beta_stress *= 0.2
            elif abs(z_stress) < 1.5:
                beta_stress *= 0.6
            else:
                beta_stress *= 1.2
            betas["Z_STRESS"] = beta_stress

        # Momentum activation
        if "Z_GOLD_MOM" in betas:
            z_mom = cur.get("Z_GOLD_MOM", 0.0)
            beta_mom = betas["Z_GOLD_MOM"]
            if abs(z_mom) < 0.5:
                beta_mom *= 0.2
            elif abs(z_mom) < 1.0:
                beta_mom *= 0.6
            beta_mom = np.clip(beta_mom, -0.7, 0.7)
            betas["Z_GOLD_MOM"] = beta_mom

        fair_z = sum(betas[c] * cur.get(c, 0.0) for c in X_cols if c in betas)
        fair_log = b0 + fair_z

        tmp.loc[end_idx, "FAIR_GOLD_Z"] = fair_z
        tmp.loc[end_idx, "FAIR_GOLD_LOG"] = fair_log
        tmp.loc[end_idx, "FAIR_GOLD"] = np.exp(fair_log)

        # Per-driver contributions
        for driver, out_col in [
            ("Z_REAL_YIELD", "CONTRIB_REAL_YIELD"),
            ("Z_USD", "CONTRIB_USD"),
            ("Z_INFL", "CONTRIB_INFL"),
            ("Z_STRESS", "CONTRIB_STRESS"),
            ("Z_LIQ", "CONTRIB_LIQ"),
            ("Z_ETF", "CONTRIB_ETF"),
            ("Z_GEO", "CONTRIB_GEO"),
            ("Z_GOLD_MOM", "CONTRIB_GOLD_MOM"),
        ]:
            tmp.loc[end_idx, out_col] = betas.get(driver, 0.0) * cur.get(driver, np.nan)

        tmp.loc[end_idx, "CONTRIB_CB"] = (
            betas.get("Z_CB", 0.0) * cur.get("Z_CB", 0.0)
            + betas.get("Z_CB_TREND", 0.0) * cur.get("Z_CB_TREND", 0.0)
        )

        # Rolling betas
        tmp.loc[end_idx, "BETA_INTERCEPT"] = b0
        for beta_key, beta_col in [
            ("Z_REAL_YIELD", "BETA_RY"), ("Z_USD", "BETA_USD"), ("Z_INFL", "BETA_INFL"),
            ("Z_STRESS", "BETA_STRESS"), ("Z_LIQ", "BETA_LIQ"), ("Z_ETF", "BETA_ETF"),
            ("Z_CB", "BETA_CB"), ("Z_CB_TREND", "BETA_CB_TREND"), ("Z_GOLD_MOM", "BETA_GOLD_MOM"),
        ]:
            if beta_key in betas:
                tmp.loc[end_idx, beta_col] = betas[beta_key]

        tmp.loc[end_idx, "ETF_INPUT_LATEST"] = cur.get("Z_ETF", np.nan)
        tmp.loc[end_idx, "CB_INPUT_LATEST"] = cur.get("Z_CB", np.nan)

    # ── Post-loop cleanup ─────────────────────────────────────────────────────
    for col in ["Z_ETF", "Z_CB", "CONTRIB_ETF", "CONTRIB_CB"]:
        if col in tmp.columns:
            tmp[col] = tmp[col].fillna(0.0)

    tmp["FAIR_GOLD_GAP_PCT"] = np.where(
        (tmp["FAIR_GOLD"] > 0) & tmp["GOLD_USD"].notna(),
        (tmp["GOLD_USD"] / tmp["FAIR_GOLD"] - 1.0) * 100.0,
        np.nan,
    )

    # Debug summary via logging
    logger.debug("FAIR_GOLD non-null: %d", int(tmp["FAIR_GOLD"].notna().sum()))
    logger.debug("FAIR_GOLD_GAP_PCT non-null: %d", int(tmp["FAIR_GOLD_GAP_PCT"].notna().sum()))

    output_cols = [
        "FAIR_GOLD_Z", "FAIR_GOLD", "FAIR_GOLD_GAP_PCT",
        "BETA_INTERCEPT", "BETA_RY", "BETA_USD", "BETA_INFL",
        "BETA_STRESS", "BETA_LIQ", "BETA_ETF", "BETA_CB",
        "Z_REAL_YIELD", "Z_USD", "Z_INFL", "Z_STRESS", "Z_LIQ", "Z_ETF", "Z_CB",
        "CONTRIB_REAL_YIELD", "CONTRIB_USD", "CONTRIB_INFL", "CONTRIB_STRESS",
        "CONTRIB_LIQ", "CONTRIB_ETF", "CONTRIB_CB",
        "CB_INPUT_LATEST", "ETF_INPUT_LATEST",
        "CONTRIB_GEO", "BETA_CB_TREND", "Z_GOLD_MOM",
    ]
    existing = [c for c in output_cols if c in tmp.columns]
    return tmp[existing]
