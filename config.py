"""
config.py
---------
Typed configuration and result containers.
Replaces the 15-argument function signatures that proliferated in the old monolith.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AppConfig:
    """All user-controlled settings, populated by sidebar.py."""
    mode: str = "Market Acceleration (fast)"
    monthly_method: str = "avg"
    lookback: Optional[int] = 240
    history_view: str = "Full history"

    # Triggers
    trig_hi: float = 0.60
    trig_lo: float = -0.60
    persist: int = 2

    # Monte Carlo
    mc_horizon: int = 12
    mc_n_sims: int = 2000

    # Comparison view
    enable_compare: bool = False
    compare_modes: Optional[list[str]] = None

    # Crisis settings
    crisis: Optional[str] = None
    win_start: Optional[str] = None
    win_end: Optional[str] = None

    # Indicator weights (structural mode)
    w_real: float = 0.30
    w_infl: float = 0.20
    w_usd: float = 0.20
    w_curve: float = 0.15
    w_fisc: float = 0.15
    w_tips: float = 0.10
    w_hy: float = 0.10
    include_tips: bool = False
    include_hy: bool = False

    # Acceleration weights
    accel_method: str = "Fixed (recommended)"
    w_g: float = 0.35
    w_ry: float = 0.25
    w_st: float = 0.25
    w_u: float = 0.15

    # Intraday
    show_intraday_rsi: bool = False
    intraday_interval: str = "15m"
    intraday_lookback_days: int = 10

    # Live updates
    live_updates: bool = False
    refresh_seconds: int = 60

    # Screener tickers
    tickers_top10: list[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "AVGO", "COST", "AMD",
    ])

    # UI options
    show_indicator_thresholds: bool = True


@dataclass
class RunResults:
    """All computed results for the current run — passed to tab renderers."""
    # Primary model output
    res: pd.DataFrame = field(default_factory=pd.DataFrame)
    res_base: pd.DataFrame = field(default_factory=pd.DataFrame)
    res_accel_base: pd.DataFrame = field(default_factory=pd.DataFrame)
    core_keys: list[str] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    dirs: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)

    # Mode metadata
    gold_stats: Optional[dict] = None
    crisis_fit: dict = field(default_factory=lambda: {
        "fit_pct": np.nan, "coverage_used": 0, "coverage_total": 0, "by_indicator": {}
    })

    # Market structural (always computed)
    res_market: pd.DataFrame = field(default_factory=pd.DataFrame)
    core_keys_market: list[str] = field(default_factory=list)
    thresholds_market: dict = field(default_factory=dict)
    dirs_market: dict = field(default_factory=dict)
    weights_market: dict = field(default_factory=dict)
    latest_market: Optional[pd.Series] = None
    market_regime_now: str = "—"
    market_signal_now: float = np.nan

    # Convenience
    weights_struct: dict = field(default_factory=dict)
    weights_crisis: dict = field(default_factory=dict)
