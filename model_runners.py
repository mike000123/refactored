"""
model_runners.py
----------------
High-level model dispatch functions.
These call macro_models.py (canonical computation) and crisis_analysis.py.
No Streamlit dependency — return error strings instead of calling st.error/stop.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from macro_models import (
    compute_accel_features,
    compute_accel_thresholds_quantiles,
    compute_signal,
    get_accel_dirs,
    get_market_dirs,
    get_market_weights,
    market_structural_thresholds,
    run_accel as _mm_run_accel,
    run_market_structural as _mm_run_market_structural,
    run_structural as _mm_run_structural,
    structural_thresholds,
    accel_thresholds_fixed,
    bucket_accel,
    label_signal,
)
from crisis_analysis import (
    compute_crisis_fit_score,
    compute_thresholds_from_window,
    crisis_conditioned_gold_stats,
)

logger = logging.getLogger(__name__)

# ── Direction / label dicts ────────────────────────────────────────────────────

DIRS_STRUCT = {
    "REAL_YIELD_CPI": False,
    "CPI_YOY": True,
    "USD_12M_CHG": False,
    "CURVE_10Y_3M": False,
    "DEFICIT_GDP": False,
    "REAL_YIELD_TIPS10": False,
    "HY_OAS": True,
}

DIRS_MARKET = get_market_dirs()

DIRS_CRISIS = {
    "REAL_YIELD_CPI": False,
    "CPI_YOY": True,
    "USD_12M_CHG": False,
    "CURVE_10Y_3M": False,
    "REAL_YIELD_TIPS10": False,
    "HY_OAS": True,
}

DIRS_ACCEL = get_accel_dirs()

LABELS = {
    "REAL_YIELD_CPI": "Real yield (10Y − CPI YoY)",
    "CPI_YOY": "Inflation (CPI YoY)",
    "USD_12M_CHG": "USD 12M % change (TWEX)",
    "CURVE_10Y_3M": "Curve (10Y–3M)",
    "DEFICIT_GDP": "Deficit % GDP",
    "REAL_YIELD_TIPS10": "Real yield (10Y TIPS, DFII10, 20D MA)",
    "HY_OAS": "High Yield OAS (20D MA)",
    "QQQ_ABOVE_MA200": "QQQ above MA200",
    "QQQ_MA50_SLOPE_20D": "QQQ MA50 slope (20d, %)",
    "MARKET_BREADTH_ABOVE_MA200": "Market breadth (% above MA200)",
}

LABELS_ACCEL = {
    "GOLD_6M_RET": "Gold 6M return (%)",
    "REALYIELD_3M_CHG": "Real yield 3M change (pp, TIPS preferred)",
    "STRESS_3M_CHG": "Stress 3M change (pp, HY OAS)",
    "USD_3M_CHG": "USD 3M change (%) (TWEX)",
}

LABELS_MARKET = {
    "REAL_YIELD_CPI": "Real yield (10Y − CPI YoY)",
    "USD_12M_CHG": "USD 12M % change (TWEX)",
    "CURVE_10Y_3M": "Curve (10Y–3M)",
    "HY_OAS": "High Yield OAS (20D MA)",
    "QQQ_ABOVE_MA200": "QQQ above MA200",
    "QQQ_MA50_SLOPE_20D": "QQQ MA50 slope (20d, %)",
    "MARKET_BREADTH_ABOVE_MA200": "Market breadth (% above MA200)",
}

LEGEND_MAP = {
    "REAL_YIELD_CPI": "Real yield (10Y − CPI YoY)",
    "CPI_YOY": "Inflation (CPI YoY)",
    "USD_12M_CHG": "USD 12M % change (TWEX)",
    "CURVE_10Y_3M": "Curve (10Y–3M)",
    "DEFICIT_GDP": "Deficit % GDP",
    "REAL_YIELD_TIPS10": "Real yield (10Y TIPS, 20D MA)",
    "HY_OAS": "High Yield OAS (20D MA)",
}

CRISIS_PRESET_WINDOWS: dict[str, tuple[str, str]] = {
    "1929 Great Depression": ("1929-01-01", "1933-12-31"),
    "1974 Oil Shock": ("1973-01-01", "1975-12-31"),
    "1980 Volcker Shock": ("1979-01-01", "1983-12-31"),
    "2011 Euro Crisis": ("2010-01-01", "2012-12-31"),
    "2020 Pandemic": ("2020-01-01", "2021-06-30"),
}


# ── Model runners ──────────────────────────────────────────────────────────────

def run_structural(
    res_base: pd.DataFrame,
    weights_struct: dict,
    dirs_struct: dict | None = None,
) -> tuple[pd.DataFrame, list, dict, dict, dict]:
    """Delegate to macro_models.run_structural (canonical version)."""
    return _mm_run_structural(res_base, weights_struct, dirs_struct or DIRS_STRUCT)


def run_market_structural(
    res_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list, dict, dict, dict]:
    """Delegate to macro_models.run_market_structural (canonical version)."""
    return _mm_run_market_structural(res_base)


def run_accel(
    res_accel_base: pd.DataFrame,
    accel_method: str,
    w_g: float,
    w_ry: float,
    w_st: float,
    w_u: float,
) -> tuple[pd.DataFrame, list, dict, dict, dict]:
    """Delegate to macro_models.run_accel (canonical version)."""
    return _mm_run_accel(res_accel_base, accel_method, w_g, w_ry, w_st, w_u)


def run_crisis(
    res_base: pd.DataFrame,
    weights_crisis: dict,
    dirs_crisis: dict | None = None,
    win_start: str = "",
    win_end: str = "",
) -> tuple[pd.DataFrame, list, dict, dict, dict, dict, dict, Optional[str]]:
    """
    Run crisis similarity model.

    Returns:
        (res, core_keys, thresholds, dirs, weights, gold_stats, crisis_fit, error_message)
    where error_message is None on success or a string the UI layer should display.
    """
    _dirs = dirs_crisis or DIRS_CRISIS
    candidate_keys = [k for k in _dirs.keys() if k in weights_crisis]

    thresholds = compute_thresholds_from_window(
        res_base, win_start, win_end, keys=candidate_keys, dirs=_dirs
    )

    _empty = {
        "fit_pct": np.nan, "coverage_used": 0, "coverage_total": 0, "by_indicator": {}
    }
    _empty_gold = {"3m": {"ok": False, "reason": "no-overlap"}, "6m": {"ok": False, "reason": "no-overlap"}}

    if not thresholds:
        avail_start = res_base.index.min()
        avail_end = res_base.index.max()
        err = (
            f"No data available in the selected crisis window ({win_start} → {win_end}).\n\n"
            f"Available dataset range: {avail_start.date()} → {avail_end.date()}.\n\n"
            "Pick a later crisis window or switch to Custom and use an overlapping range."
        )
        res = res_base.copy()
        res["SIGNAL"] = np.nan
        res["REGIME"] = "No data"
        return res, [], {}, {}, {}, _empty_gold, _empty, err

    core_keys = list(thresholds.keys())
    warning: Optional[str] = None
    if len(core_keys) < 2:
        warning = (
            "Crisis window has very limited usable indicator history. "
            "Similarity score may be unstable (too few indicators available)."
        )

    dirs = {k: _dirs[k] for k in core_keys}
    weights: dict = {k: weights_crisis.get(k, 0.0) for k in core_keys}
    tot = sum(weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}

    res = compute_signal(res_base, thresholds, weights, dirs)

    latest_row = res.iloc[-1] if not res.empty else pd.Series(dtype=float)
    crisis_fit = compute_crisis_fit_score(
        df=res_base, current_row=latest_row,
        win_start=win_start, win_end=win_end,
        keys=core_keys, weights=weights, min_points=24,
    )

    gold_stats_6m = crisis_conditioned_gold_stats(
        res_base, win_start, win_end, thresholds, weights, dirs, horizon_months=6, band=0.15
    )
    gold_stats_3m = crisis_conditioned_gold_stats(
        res_base, win_start, win_end, thresholds, weights, dirs, horizon_months=3, band=0.15
    )
    gold_stats = {"3m": gold_stats_3m, "6m": gold_stats_6m}

    return res, core_keys, thresholds, dirs, weights, gold_stats, crisis_fit, warning


def build_weight_dicts(
    w_real: float,
    w_infl: float,
    w_usd: float,
    w_curve: float,
    w_fisc: float,
    w_tips: float,
    w_hy: float,
    include_tips: bool,
    include_hy: bool,
    res_base: pd.DataFrame,
    mode: str,
) -> tuple[dict, dict]:
    """Construct weights_struct and weights_crisis from sidebar slider values."""
    raw_struct = {
        "REAL_YIELD_CPI": w_real,
        "CPI_YOY": w_infl,
        "USD_12M_CHG": w_usd,
        "CURVE_10Y_3M": w_curve,
        "DEFICIT_GDP": w_fisc if mode == "Structural Regime (today)" else 0.0,
        "REAL_YIELD_TIPS10": w_tips if include_tips else 0.0,
        "HY_OAS": w_hy if include_hy else 0.0,
    }

    # Avoid double-counting real yields when TIPS is enabled and has data
    if (
        include_tips
        and "REAL_YIELD_TIPS10" in res_base.columns
        and res_base["REAL_YIELD_TIPS10"].dropna().size > 0
    ):
        raw_struct["REAL_YIELD_CPI"] = 0.0

    active = {k: v for k, v in raw_struct.items() if v > 0}
    tot = sum(active.values()) or 1.0
    weights_struct = {k: v / tot for k, v in active.items()}

    raw_crisis = {
        "REAL_YIELD_CPI": w_real,
        "CPI_YOY": w_infl,
        "USD_12M_CHG": w_usd,
        "CURVE_10Y_3M": w_curve,
        "DEFICIT_GDP": 0.0,
        "REAL_YIELD_TIPS10": 0.0,
        "HY_OAS": 0.0,
    }
    active_c = {k: v for k, v in raw_crisis.items() if v > 0}
    tot_c = sum(active_c.values()) or 1.0
    weights_crisis = {k: v / tot_c for k, v in active_c.items()}

    return weights_struct, weights_crisis
