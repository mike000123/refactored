"""
feature_builder.py
------------------
Transforms raw data series into the modelling DataFrame.
No Streamlit dependency.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import (
    load_gold_series_fast,
    load_yf_panel_fast,
    load_yf_series_fast,
    safe_fred,
)

logger = logging.getLogger(__name__)


# ── Basic transforms ─────────────────────────────────────────────────────────

def to_monthly(s: pd.Series, method: str = "avg") -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return s.resample("ME").last() if method == "eom" else s.resample("ME").mean()


def splice_index(old: pd.Series, new: pd.Series, anchor_date: str = "2006-01-31") -> pd.Series:
    """Scale *old* to match *new* at *anchor_date*, then concatenate."""
    old_m = old.copy()
    new_m = new.copy()
    old_m.index = pd.to_datetime(old_m.index)
    new_m.index = pd.to_datetime(new_m.index)

    anchor = pd.to_datetime(anchor_date)
    old_anchor = old_m.loc[:anchor].dropna().iloc[-1] if not old_m.loc[:anchor].dropna().empty else np.nan
    new_anchor = new_m.loc[:anchor].dropna().iloc[-1] if not new_m.loc[:anchor].dropna().empty else np.nan

    if not (np.isfinite(old_anchor) and np.isfinite(new_anchor) and old_anchor != 0):
        combined = pd.concat([old_m, new_m]).sort_index()
        return combined[~combined.index.duplicated(keep="last")]

    scale = new_anchor / old_anchor
    old_scaled = old_m * scale
    combined = pd.concat([old_scaled, new_m]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def pct_slope(series: pd.Series, window: int) -> pd.Series:
    """Percentage change of a rolling mean over *window* bars."""
    return series.pct_change(window) * 100.0


# ── Technical indicators ─────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Returns RSI in [0, 100]."""
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    s = series.astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def robust_zscore(series: pd.Series, window: int = 60, min_periods: int = 24) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


# ── Intraday helpers ──────────────────────────────────────────────────────────

def compute_intraday_rsi_pack(intraday_close: pd.Series, rsi_period: int = 14) -> dict:
    """Returns dict with RSI series and supplementary diagnostics."""
    if not isinstance(intraday_close, pd.Series) or intraday_close.dropna().empty:
        return {}
    s = intraday_close.dropna().astype(float).sort_index()
    r = rsi(s, period=rsi_period).dropna()
    if r.empty:
        return {}
    return {
        "RSI_14_INTRA": r,
        "RSI_14_INTRA_Z_2W": zscore(r, window=14 * 10),
        "RSI_14_INTRA_SLOPE_3H": r.diff(36),
    }


def classify_rsi_overlay(rsi_val: float, rsi_z: float, rsi_slope_3m: float) -> str:
    if pd.isna(rsi_val):
        return "RSI unavailable"
    if not pd.isna(rsi_z):
        if rsi_z >= 1.5:
            stretch = "Stretched (high)"
        elif rsi_z >= 1.0:
            stretch = "Elevated"
        elif rsi_z <= -1.5:
            stretch = "Stretched (low)"
        elif rsi_z <= -1.0:
            stretch = "Depressed"
        else:
            stretch = "Normal"
    else:
        if rsi_val >= 70:
            stretch = "Overbought"
        elif rsi_val <= 30:
            stretch = "Oversold"
        else:
            stretch = "Neutral"

    if pd.isna(rsi_slope_3m):
        mom = "Momentum unknown"
    elif rsi_slope_3m >= 5:
        mom = "Momentum strengthening"
    elif rsi_slope_3m <= -5:
        mom = "Momentum cooling"
    else:
        mom = "Momentum steady"
    return f"{stretch} | {mom}"


# ── Utility ───────────────────────────────────────────────────────────────────

def last_update_date(df: pd.DataFrame, col: str) -> str:
    s = df[col].dropna()
    return "—" if s.empty else s.index.max().date().isoformat()


def latest_value_and_date(df: pd.DataFrame, col: str) -> tuple[float, str]:
    s = df[col].dropna()
    if s.empty:
        return np.nan, "—"
    return float(s.iloc[-1]), s.index.max().date().isoformat()


# ── Main feature construction ─────────────────────────────────────────────────

def build_features(monthly_method: str = "avg") -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch all series, align to a monthly index, and return the model DataFrame.
    Returns (df, warnings_list).
    Warnings are plain strings the UI layer can forward to st.warning().
    """
    warnings: list[str] = []

    def _w(msg: str | None):
        if msg:
            warnings.append(msg)

    # ── Rates ──────────────────────────────────────────────────────────────────
    dgs10_raw, w = safe_fred("DGS10");  _w(w)
    dgs10 = to_monthly(dgs10_raw, monthly_method)

    tb3ms_raw, w = safe_fred("TB3MS");  _w(w)
    tb3ms = to_monthly(tb3ms_raw, "avg")

    # ── Inflation ──────────────────────────────────────────────────────────────
    cpi_raw, w = safe_fred("CPIAUCSL");  _w(w)
    cpi = to_monthly(cpi_raw, "avg")
    cpi_yoy = 100 * (cpi / cpi.shift(12) - 1.0)
    cpi_yoy.name = "CPI_YOY"

    real_yield_cpi = dgs10 - cpi_yoy
    real_yield_cpi.name = "REAL_YIELD_CPI"

    t10yie_raw, w = safe_fred("T10YIE");  _w(w)
    t10yie = to_monthly(t10yie_raw, monthly_method)
    t10yie.name = "T10YIE"

    infl_exp = t10yie.combine_first(cpi_yoy)
    infl_exp.name = "INFL_EXP_PROXY"

    real_yield_proxy = dgs10 - infl_exp
    real_yield_proxy.name = "REAL_YIELD_PROXY"

    # ── Dollar ──────────────────────────────────────────────────────────────────
    twexbmth_raw, w = safe_fred("TWEXBMTH");  _w(w)
    twexbmth = to_monthly(twexbmth_raw, "avg")

    twexbgsmth_raw, w = safe_fred("TWEXBGSMTH");  _w(w)
    twexbgsmth = to_monthly(twexbgsmth_raw, "avg")

    usd_idx = splice_index(twexbmth, twexbgsmth)
    usd_idx.name = "USD_TWEX_SPLICE"

    usd_12m_chg = 100 * (usd_idx / usd_idx.shift(12) - 1.0)
    usd_12m_chg.name = "USD_12M_CHG"

    # ── Curve ───────────────────────────────────────────────────────────────────
    curve = dgs10 - tb3ms
    curve.name = "CURVE_10Y_3M"

    # ── Deficit ─────────────────────────────────────────────────────────────────
    deficit_gdp_raw, w = safe_fred("FYFSGDA188S");  _w(w)
    if not deficit_gdp_raw.empty:
        deficit_gdp = deficit_gdp_raw.resample("ME").ffill()
    else:
        deficit_gdp = pd.Series(
            index=pd.DatetimeIndex([], name="DATE"), dtype=float, name="DEFICIT_GDP"
        )
    deficit_gdp.name = "DEFICIT_GDP"

    # ── TIPS real yield ──────────────────────────────────────────────────────────
    dfii10_raw, w = safe_fred("DFII10");  _w(w)
    real_yield_tips10 = dfii10_raw.rolling(20).mean().dropna().resample("ME").last()
    real_yield_tips10.name = "REAL_YIELD_TIPS10"

    # ── HY OAS ───────────────────────────────────────────────────────────────────
    hy_oas_raw, w = safe_fred("BAMLH0A0HYM2");  _w(w)
    hy_oas_m = to_monthly(hy_oas_raw.rolling(20).mean(), monthly_method)
    hy_oas_m.name = "HY_OAS"

    # ── Liquidity (Fed balance sheet) ─────────────────────────────────────────
    walcl_raw, w = safe_fred("WALCL");  _w(w)
    walcl_m = to_monthly(walcl_raw, monthly_method)
    liq_12m_chg = 100.0 * (walcl_m / walcl_m.shift(12) - 1.0)
    liq_12m_chg.name = "LIQ_12M_CHG"

    # ── Combine core series ───────────────────────────────────────────────────
    df = pd.concat(
        [real_yield_cpi, real_yield_proxy, infl_exp, cpi_yoy, usd_idx,
         usd_12m_chg, curve, deficit_gdp, real_yield_tips10, hy_oas_m, liq_12m_chg],
        axis=1,
    )

    for col in ["DEFICIT_GDP", "REAL_YIELD_TIPS10", "HY_OAS"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ── Gold ──────────────────────────────────────────────────────────────────
    gold, gold_daily, gold_source, gold_err = load_gold_series_fast(start="1970-01-01")
    if gold_err:
        warnings.append(f"Gold price load failed: {gold_err}")

    if gold is not None:
        if isinstance(gold, pd.DataFrame):
            for c in ["Close", "Adj Close", "GOLD_USD"]:
                if c in gold.columns:
                    gold = gold[c]
                    break
            else:
                gold = gold.iloc[:, 0]
        gold = gold.rename("GOLD_USD")
        df = pd.concat([df, gold], axis=1)

    # ── QQQ ───────────────────────────────────────────────────────────────────
    qqq_ser, _, qqq_err = load_yf_series_fast("QQQ", start="2000-01-01")
    if qqq_ser is not None and not qqq_ser.empty:
        qqq_d = qqq_ser.sort_index()
        qqq_d.index = pd.to_datetime(qqq_d.index, errors="coerce")
        qqq_m = qqq_d.resample("ME").last()
        qqq_ma200 = qqq_d.rolling(200, min_periods=200).mean().resample("ME").last()
        qqq_ma50 = qqq_d.rolling(50, min_periods=50).mean()
        qqq_ma50_slope = pct_slope(qqq_ma50, 20).resample("ME").last()
        df["QQQ_ABOVE_MA200"] = (qqq_m > qqq_ma200).astype(float).reindex(df.index)
        df["QQQ_MA50_SLOPE_20D"] = qqq_ma50_slope.reindex(df.index)
    else:
        df["QQQ_ABOVE_MA200"] = np.nan
        df["QQQ_MA50_SLOPE_20D"] = np.nan
        if qqq_err:
            warnings.append(f"QQQ load failed: {qqq_err}")

    # ── Market breadth ────────────────────────────────────────────────────────
    breadth_tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD", "NFLX",
        "COST", "WMT", "PLTR", "CSCO", "INTC", "AMAT", "LRCX", "MU", "ASML", "PEP",
    ]
    breadth_px, breadth_failed = load_yf_panel_fast(breadth_tickers, start="2000-01-01")
    if not breadth_px.empty:
        above_ma200 = pd.DataFrame({
            c: (breadth_px[c] > breadth_px[c].rolling(200, min_periods=200).mean()).astype(float)
            for c in breadth_px.columns
        })
        breadth_monthly = (above_ma200.mean(axis=1) * 100.0).resample("ME").last()
        breadth_monthly.name = "MARKET_BREADTH_ABOVE_MA200"
        df = df.join(breadth_monthly, how="left")
    else:
        df["MARKET_BREADTH_ABOVE_MA200"] = np.nan
        if breadth_failed:
            warnings.append(f"Breadth load failed for {len(breadth_failed)} tickers.")

    # ── RSI overlays (monthly) ─────────────────────────────────────────────────
    if "GOLD_USD" in df.columns:
        df["RSI_14"] = rsi(df["GOLD_USD"], period=14)
        df["RSI_Z_60"] = zscore(df["RSI_14"], window=60)
        df["RSI_SLOPE_3M"] = df["RSI_14"] - df["RSI_14"].shift(3)

    if "GOLD_USD" in df.columns and isinstance(gold_daily, pd.Series) and not gold_daily.dropna().empty:
        monthly = df["GOLD_USD"].dropna()
        gd = gold_daily.dropna()
        today_close = float(gd.iloc[-1])
        today_dt = gd.index[-1]
        monthly_like = monthly.copy()
        monthly_like.loc[today_dt] = today_close
        monthly_like = monthly_like.sort_index()
        rsi_asof_raw = rsi(monthly_like, period=14)
        df["RSI_14_ASOF"] = rsi_asof_raw.reindex(df.index, method="ffill")
        df.attrs["RSI_14_ASOF_RAW"] = rsi_asof_raw

    # ── RSI overlays (daily) ──────────────────────────────────────────────────
    if isinstance(gold_daily, pd.Series) and not gold_daily.dropna().empty:
        rsi_d_raw = rsi(gold_daily, period=14)
        df["RSI_14D"] = rsi_d_raw.reindex(df.index, method="ffill")
        df["RSI_14D_Z_1Y"] = zscore(rsi_d_raw, window=252).reindex(df.index, method="ffill")
        df["RSI_14D_SLOPE_1M"] = rsi_d_raw.diff(20).reindex(df.index, method="ffill")
        df.attrs["RSI_14D_DAILY"] = rsi_d_raw
        df.attrs["RSI_14D_Z_1Y_DAILY"] = zscore(rsi_d_raw, window=252)
        df.attrs["RSI_14D_SLOPE_1M_DAILY"] = rsi_d_raw.diff(20)

    # ── Store daily gold in attrs for chart use ────────────────────────────────
    df.attrs["gold_error"] = gold_err
    if isinstance(gold_daily, pd.Series) and not gold_daily.empty:
        gold_daily_clean = gold_daily.copy()
        gold_daily_clean.index = pd.to_datetime(gold_daily_clean.index, errors="coerce")
        gold_daily_clean = gold_daily_clean[~gold_daily_clean.index.isna()].sort_index()
        df.attrs["gold_daily"] = gold_daily_clean

    logger.debug("build_features complete: %d rows, %d columns", len(df), len(df.columns))
    return df, warnings
