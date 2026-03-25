"""
cache_store.py
--------------
Low-level parquet cache for FRED and Yahoo Finance series.
No Streamlit dependency — safe to use in any context.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Directory layout ─────────────────────────────────────────────────────────
CACHE_ROOT = Path("data_cache")
FRED_CACHE_DIR = CACHE_ROOT / "fred"
YF_DAILY_CACHE_DIR = CACHE_ROOT / "yahoo_daily"

for _p in [CACHE_ROOT, FRED_CACHE_DIR, YF_DAILY_CACHE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


# ── Key helpers ───────────────────────────────────────────────────────────────

def _safe_key(key: str) -> str:
    """Sanitise a series id/ticker so it is safe as a filename."""
    for ch in r"/\:=?* ":
        key = key.replace(ch, "_")
    return key


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{_safe_key(key)}.parquet"


# ── Read / write ──────────────────────────────────────────────────────────────

def load_series_cache(cache_dir: Path, key: str) -> pd.Series | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty or "date" not in df.columns or "value" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"])
        s = df.set_index("date")["value"].sort_index()
        s.name = key
        return s
    except Exception:
        return None


def save_series_cache(cache_dir: Path, key: str, s: pd.Series) -> None:
    if s is None or len(s) == 0:
        return
    try:
        ser = s.copy()
        ser.index = pd.to_datetime(ser.index, errors="coerce")
        ser = pd.to_numeric(ser, errors="coerce")
        ser = ser[~ser.index.isna()]
        if len(ser) == 0:
            return
        df = pd.DataFrame({"date": ser.index, "value": ser.values})
        df.to_parquet(_cache_path(cache_dir, key), index=False)
    except Exception:
        pass


def merge_series_keep_latest(
    old_s: pd.Series | None,
    new_s: pd.Series | None,
    name: str | None = None,
) -> pd.Series:
    """Merge two series, keeping the newest value on duplicate dates."""
    if old_s is None or len(old_s) == 0:
        out = new_s.copy() if new_s is not None else pd.Series(dtype=float)
    elif new_s is None or len(new_s) == 0:
        out = old_s.copy()
    else:
        out = pd.concat([old_s, new_s])
        out = out[~out.index.duplicated(keep="last")].sort_index()
    if name is not None:
        out.name = name
    return out


def tail_start_date_from_cache(
    cached: pd.Series | None,
    overlap_days: int = 10,
    default_start: str = "1970-01-01",
) -> str:
    """Return an ISO date string to use as the incremental download start."""
    if cached is None or len(cached) == 0:
        return default_start
    last_dt = pd.to_datetime(cached.index.max(), errors="coerce")
    if pd.isna(last_dt):
        return default_start
    return str((last_dt - pd.Timedelta(days=overlap_days)).date())


def clear_cache_dir(cache_dir: Path) -> int:
    """Delete all parquet files in *cache_dir*. Returns the number removed."""
    removed = 0
    for p in cache_dir.glob("*.parquet"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    return removed
