"""
data_loader.py
--------------
All remote data fetching for FRED series and Yahoo Finance tickers.
No Streamlit imports — call st.warning/st.error at the *call site* in the UI layer
by inspecting the optional warning string returned by each function.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Optional

import pandas as pd
import requests

from cache_store import (
    FRED_CACHE_DIR,
    YF_DAILY_CACHE_DIR,
    load_series_cache,
    merge_series_keep_latest,
    save_series_cache,
    tail_start_date_from_cache,
)

logger = logging.getLogger(__name__)

# ── Shared HTTP session ────────────────────────────────────────────────────────
_FRED_SESSION = requests.Session()

# ── Module-level refresh flag ─────────────────────────────────────────────────
# Set to True before calling any loader to force a live refresh.
# In Streamlit, drive this via st.session_state["force_refresh_data"].
refresh_remote_data: bool = False


# ── FRED ──────────────────────────────────────────────────────────────────────

def fred_csv_live(series_id: str) -> pd.Series:
    """Live FRED fetch only. Raises RuntimeError on failure."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    last_err = None
    for attempt in range(2):
        try:
            r = _FRED_SESSION.get(url, timeout=12)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            date_col, val_col = df.columns[0], df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            s = df.set_index(date_col)[val_col].sort_index()
            s.name = series_id
            return s
        except Exception as e:
            last_err = e
            if attempt < 1:
                time.sleep(1.5)
    raise RuntimeError(f"FRED live download failed for {series_id}: {last_err}")


def fred_csv(series_id: str, refresh_remote: bool = False) -> pd.Series:
    """
    Cache-first FRED loader.
    Returns cached data immediately unless *refresh_remote* is True,
    in which case it attempts a live fetch and merges the result.
    Falls back to cache on network errors.
    """
    cached = load_series_cache(FRED_CACHE_DIR, series_id)
    if cached is not None:
        cached.name = series_id

    if cached is not None and not cached.empty and not refresh_remote:
        return cached

    try:
        live = fred_csv_live(series_id)
        merged = merge_series_keep_latest(cached, live, name=series_id)
        save_series_cache(FRED_CACHE_DIR, series_id, merged)
        return merged
    except Exception as e:
        if cached is not None and not cached.empty:
            logger.warning("FRED refresh failed for %s, using cache: %s", series_id, e)
            return cached
        raise RuntimeError(f"FRED download failed for {series_id}: {e}")


def safe_fred(series_id: str) -> tuple[pd.Series, Optional[str]]:
    """
    Wrapper that never raises. Returns (series, warning_message | None).
    The UI layer is responsible for displaying warnings.
    """
    try:
        return fred_csv(series_id, refresh_remote=refresh_remote_data), None
    except Exception as e:
        empty = pd.Series(
            index=pd.DatetimeIndex([], name="DATE"), dtype=float, name=series_id
        )
        return empty, f"FRED series {series_id} failed: {e}"


# ── Yahoo Finance – Gold ───────────────────────────────────────────────────────

def _yf_close_series(ticker: str, start: str, period: str | None = None) -> pd.Series:
    """Download a single Close series from yfinance. Raises on failure."""
    import yfinance as yf

    kwargs = dict(progress=False, auto_adjust=True, threads=False)
    if period:
        raw = yf.download(ticker, period=period, **kwargs)
    else:
        raw = yf.download(ticker, start=start, **kwargs)

    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: empty download")

    col = "Close" if "Close" in raw.columns else (
        "Adj Close" if "Adj Close" in raw.columns else None
    )
    if col is None:
        raise ValueError(f"{ticker}: no Close column")

    obj = raw[col]
    ser = obj.iloc[:, 0].dropna() if isinstance(obj, pd.DataFrame) else obj.dropna()
    if ser.empty:
        raise ValueError(f"{ticker}: all NaN after dropna")

    ser.name = ticker
    return ser


def load_gold_series_live(start: str = "1970-01-01") -> tuple[pd.Series | None, str | None, str | None]:
    """Live Yahoo gold fetch only. Tries GC=F → GLD → IAU in order."""
    last_err = None
    for ticker in ["GC=F", "GLD", "IAU"]:
        try:
            ser = _yf_close_series(ticker, start=start)
            return ser, f"yfinance:{ticker}", None
        except Exception as e:
            last_err = f"{ticker}: {type(e).__name__} — {e}"
    return None, None, last_err


def load_gold_series_fast(
    start: str = "1970-01-01",
) -> tuple[pd.Series | None, pd.Series | None, str | None, str | None]:
    """
    Cache-first gold loader.
    Returns (gold_monthly, gold_daily, source_label, warning_message).
    """
    cached_series: pd.Series | None = None
    cached_ticker: str | None = None

    for ticker in ["GC=F", "GLD", "IAU"]:
        cached = load_series_cache(YF_DAILY_CACHE_DIR, ticker)
        if cached is not None and not cached.empty:
            cached_series = cached
            cached_ticker = ticker
            break

    if cached_series is not None and not refresh_remote_data:
        daily = cached_series.copy()
        daily.name = "GOLD_USD_D"
        monthly = cached_series.resample("ME").last()
        monthly.name = "GOLD_USD"
        return monthly, daily, f"cache:{cached_ticker}", None

    live_start = tail_start_date_from_cache(cached_series, overlap_days=10, default_start=start)
    live_ser, live_source, live_err = load_gold_series_live(start=live_start)

    if live_ser is not None and not live_ser.empty:
        live_ticker = live_source.split(":", 1)[1]
        base_cached = load_series_cache(YF_DAILY_CACHE_DIR, live_ticker)
        merged = merge_series_keep_latest(base_cached, live_ser, name=live_ticker)
        save_series_cache(YF_DAILY_CACHE_DIR, live_ticker, merged)
        daily = merged.copy()
        daily.name = "GOLD_USD_D"
        monthly = merged.resample("ME").last()
        monthly.name = "GOLD_USD"
        return monthly, daily, live_source, None

    if cached_series is not None and not cached_series.empty:
        daily = cached_series.copy()
        daily.name = "GOLD_USD_D"
        monthly = cached_series.resample("ME").last()
        monthly.name = "GOLD_USD"
        return monthly, daily, f"cache:{cached_ticker}", live_err

    return None, None, None, live_err


# ── Yahoo Finance – Generic ────────────────────────────────────────────────────

def load_yf_series_fast(
    ticker: str, start: str = "1970-01-01"
) -> tuple[pd.Series | None, str | None, str | None]:
    """
    Cache-first Yahoo daily loader for a single ticker.
    Returns (series | None, source_label | None, error | None).
    """
    cached = load_series_cache(YF_DAILY_CACHE_DIR, ticker)

    if cached is not None and not cached.empty and not refresh_remote_data:
        cached.name = ticker
        return cached, f"cache:{ticker}", None

    live_start = tail_start_date_from_cache(cached, overlap_days=10, default_start=start)

    try:
        ser = _yf_close_series(ticker, start=live_start)
        base_cached = load_series_cache(YF_DAILY_CACHE_DIR, ticker)
        merged = merge_series_keep_latest(base_cached, ser, name=ticker)
        save_series_cache(YF_DAILY_CACHE_DIR, ticker, merged)
        return merged, f"yfinance:{ticker}", None
    except Exception as e:
        err = f"{ticker}: {type(e).__name__} — {e}"
        if cached is not None and not cached.empty:
            cached.name = ticker
            return cached, f"cache:{ticker}", err
        return (
            pd.Series(index=pd.DatetimeIndex([], name="date"), dtype=float, name=ticker),
            None,
            err,
        )


def load_yf_panel_fast(
    tickers: list[str], start: str = "2000-01-01"
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load multiple tickers. Returns (panel_DataFrame, failed_dict).
    failed_dict maps ticker → error message.
    """
    out: dict[str, pd.Series] = {}
    failed: dict[str, str] = {}

    for t in tickers:
        s, _, err = load_yf_series_fast(t, start=start)
        if s is not None and not s.empty:
            out[t] = s
        else:
            failed[t] = err or "unknown error"

    panel = pd.DataFrame(out).sort_index() if out else pd.DataFrame()
    return panel, failed


# ── Yahoo Finance – Intraday ───────────────────────────────────────────────────

def yf_intraday_close(
    tickers: list[str],
    interval: str = "15m",
    lookback_days: int = 10,
) -> pd.Series:
    """
    Best-effort intraday close via yfinance (tz-naive).
    Not cached — callers should apply @st.cache_data if needed.
    """
    try:
        import yfinance as yf
    except Exception:
        return pd.Series(dtype=float)

    period_map = {1: "1d", 2: "2d", 3: "5d", 5: "5d", 7: "5d", 10: "10d", 15: "1mo", 30: "1mo"}
    period = "3mo"
    for threshold, p in sorted(period_map.items()):
        if lookback_days <= threshold:
            period = p
            break

    for t in tickers:
        try:
            df_i = yf.download(
                t, period=period, interval=interval, progress=False,
                auto_adjust=True, prepost=False,
            )
            if df_i is None or df_i.empty:
                continue
            col = "Close" if "Close" in df_i.columns else (
                "Adj Close" if "Adj Close" in df_i.columns else None
            )
            if col is None:
                continue
            s = df_i[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna()
            if s.empty:
                continue
            try:
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_convert(None)
            except Exception:
                pass
            s.name = "GOLD_INTRA"
            return s.sort_index()
        except Exception:
            continue

    return pd.Series(dtype=float)
