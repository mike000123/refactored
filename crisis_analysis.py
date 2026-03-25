"""
crisis_analysis.py
------------------
Crisis similarity scoring, conditioned gold stats, and investor playbooks.
No Streamlit dependency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from macro_models import compute_signal, compute_thresholds_from_window


# ── Forward return helper ─────────────────────────────────────────────────────

def fwd_return_months(gold: pd.Series, months: int) -> pd.Series:
    """Forward % return over *months* months, aligned to the start month."""
    g = gold.dropna()
    return 100.0 * (g.shift(-months) / g - 1.0)


# ── Crisis-conditioned gold stats ─────────────────────────────────────────────

def crisis_conditioned_gold_stats(
    df: pd.DataFrame,
    win_start: str,
    win_end: str,
    thresholds: dict,
    weights: dict,
    dirs: dict,
    horizon_months: int = 6,
    band: float = 0.15,
) -> dict:
    """
    Gold forward-return stats inside the crisis window for months whose SIGNAL
    is within ±*band* of today's SIGNAL.
    """
    if "GOLD_USD" not in df.columns:
        return {"ok": False, "reason": "GOLD_USD not available"}

    scored = compute_signal(df, thresholds, weights, dirs)
    today_sig = float(scored["SIGNAL"].dropna().iloc[-1])
    w = scored.loc[pd.to_datetime(win_start):pd.to_datetime(win_end)].copy()

    if w.empty:
        return {"ok": False, "reason": "No data in crisis window"}

    w["GOLD_FWD_RET"] = fwd_return_months(w["GOLD_USD"], horizon_months)
    sel = w[(w["SIGNAL"] >= today_sig - band) & (w["SIGNAL"] <= today_sig + band)].copy()
    sel = sel.dropna(subset=["GOLD_FWD_RET"])

    if sel.empty or sel["GOLD_FWD_RET"].dropna().shape[0] < 6:
        return {
            "ok": False,
            "reason": "Too few similar months with forward gold returns",
            "today_signal": today_sig,
            "n_samples": int(sel.shape[0]),
        }

    s = sel["GOLD_FWD_RET"].dropna()
    return {
        "ok": True,
        "today_signal": today_sig,
        "n_samples": int(s.shape[0]),
        "horizon_m": int(horizon_months),
        "mean_ret": float(s.mean()),
        "median_ret": float(s.median()),
        "p_pos": float((s > 0).mean()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    }


# ── Crisis fit score ──────────────────────────────────────────────────────────

def compute_crisis_fit_score(
    df: pd.DataFrame,
    current_row: pd.Series,
    win_start: str,
    win_end: str,
    keys: list[str],
    weights: dict,
    min_points: int = 24,
) -> dict:
    """
    Weighted similarity score (0–100) between today's macro profile and the
    selected crisis window.
    """
    start_dt = pd.to_datetime(win_start)
    end_dt = pd.to_datetime(win_end)
    w = df.loc[start_dt:end_dt].copy()

    if w.empty or not keys:
        return {"fit_pct": np.nan, "coverage_used": 0, "coverage_total": len(keys), "by_indicator": {}}

    sims: dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    used = 0

    for k in keys:
        if k not in w.columns or k not in current_row.index or k not in weights:
            continue
        hist = pd.to_numeric(w[k], errors="coerce").dropna()
        cur = pd.to_numeric(pd.Series([current_row.get(k, np.nan)]), errors="coerce").iloc[0]

        if pd.isna(cur) or len(hist) < min_points:
            continue

        med = float(hist.median())
        q25 = float(hist.quantile(0.25))
        q75 = float(hist.quantile(0.75))
        iqr = q75 - q25

        if not np.isfinite(iqr) or iqr <= 0:
            sd = float(hist.std(ddof=0))
            scale = sd if np.isfinite(sd) and sd > 0 else None
        else:
            scale = iqr

        if scale is None:
            continue

        dist = abs(float(cur) - med) / scale
        sim = float(np.exp(-dist))
        sims[k] = sim
        weighted_sum += float(weights[k]) * sim
        weight_sum += float(weights[k])
        used += 1

    fit_pct = 100.0 * (weighted_sum / weight_sum) if weight_sum > 0 else np.nan
    return {"fit_pct": fit_pct, "coverage_used": used, "coverage_total": len(keys), "by_indicator": sims}


def compute_indicator_crisis_similarity(
    df: pd.DataFrame,
    current_row: pd.Series,
    win_start: str,
    win_end: str,
    key: str,
    min_points: int = 24,
) -> dict:
    """Per-indicator similarity metadata (median, IQR, current_percentile, fit_pct, label)."""
    start_dt = pd.to_datetime(win_start)
    end_dt = pd.to_datetime(win_end)
    w = df.loc[start_dt:end_dt]

    if key not in w.columns:
        return {"fit_pct": np.nan, "label": "No data"}

    hist = pd.to_numeric(w[key], errors="coerce").dropna()
    if len(hist) < min_points:
        return {"fit_pct": np.nan, "label": "Insufficient history", "median": np.nan, "q33": np.nan, "q67": np.nan}

    cur_val = pd.to_numeric(pd.Series([current_row.get(key, np.nan)]), errors="coerce").iloc[0]
    med = float(hist.median())
    q25 = float(hist.quantile(0.25))
    q75 = float(hist.quantile(0.75))
    q33 = float(hist.quantile(0.33))
    q67 = float(hist.quantile(0.67))
    iqr = q75 - q25

    if pd.isna(cur_val):
        return {"fit_pct": np.nan, "label": "No current value", "median": med, "q33": q33, "q67": q67}

    scale = iqr if (np.isfinite(iqr) and iqr > 0) else float(hist.std(ddof=0))
    if not (np.isfinite(scale) and scale > 0):
        return {"fit_pct": np.nan, "label": "Zero variance", "median": med, "q33": q33, "q67": q67}

    dist = abs(float(cur_val) - med) / scale
    sim = float(np.exp(-dist)) * 100.0

    cur_pctile = float((hist < cur_val).mean() * 100.0)

    label = (
        "Very similar" if sim >= 80 else
        "Similar" if sim >= 60 else
        "Partial match" if sim >= 40 else
        "Dissimilar"
    )
    return {
        "fit_pct": sim,
        "label": label,
        "median": med,
        "q33": q33,
        "q67": q67,
        "current_percentile": cur_pctile,
    }


# ── Playbooks ─────────────────────────────────────────────────────────────────

def get_fit_color_label(score: float) -> str:
    if score >= 80:
        return "🔴 Strong Match"
    elif score >= 60:
        return "🟠 High Similarity"
    elif score >= 40:
        return "🟡 Moderate Similarity"
    elif score >= 20:
        return "🔵 Weak Similarity"
    return "⚪ Low / No Similarity"


def get_crisis_investor_playbook(crisis_name: str, fit_score: float) -> str:
    if pd.isna(fit_score):
        return "Historical asset playbook unavailable."

    intensity = "strongly" if fit_score >= 80 else "moderately" if fit_score >= 60 else "somewhat"
    base_note = f"Similarity strength: {fit_score:.0f}% ({intensity}). "

    playbooks = {
        "1929": """
• Risk regime: systemic deflation / liquidity collapse
• Typical winners: cash, high-quality bonds (early), gold (later phase)
• Typical losers: equities, cyclicals, leverage

Suggested positioning:
- Reduce risk assets exposure
- Increase liquidity / optionality
- Gradually accumulate gold on stress spikes
""",
        "2008": """
• Risk regime: credit stress / liquidity crunch
• Typical winners: gold, USD (early), later equities rebound
• Typical losers: financials, high leverage assets

Suggested positioning:
- Favor gold and defensive assets
- Avoid high leverage / credit-sensitive sectors
- Watch for policy pivot → risk rebound opportunity
""",
        "1980": """
• Risk regime: inflation shock / aggressive tightening
• Typical winners: commodities (early), USD (late), real yields
• Typical losers: bonds, gold (after peak), growth equities

Suggested positioning:
- Avoid duration (bonds)
- Watch for real yield spikes → gold downside risk
- Favor cash / short-term instruments
""",
        "2020": """
• Risk regime: liquidity shock → massive stimulus
• Typical winners: gold, tech, liquidity-driven assets
• Typical losers: real economy sectors (initially)

Suggested positioning:
- Follow liquidity expansion signals
- Risk-on after policy response confirmed
- Gold benefits from monetary expansion
""",
    }

    for key, text in playbooks.items():
        if key in (crisis_name or ""):
            return base_note + text

    return base_note + """
• Mixed or unclear regime
Suggested positioning:
- Stay diversified
- Use signals from individual indicators (rates, USD, liquidity)
"""


def get_historical_asset_playbook(crisis_name: str, fit_score: float) -> str:
    if pd.isna(fit_score):
        return "Historical asset playbook unavailable."

    strength = (
        "strongly" if fit_score >= 80 else
        "meaningfully" if fit_score >= 60 else
        "partly" if fit_score >= 40 else
        "weakly"
    )

    texts = {
        "1929 Great Depression": (
            f"The current setup {strength} resembles a deflationary-deleveraging regime. "
            "Historically, the best defensive posture was cash, reduce equity risk, "
            "favor high-quality sovereign bonds early, and treat gold as a later-stage hedge."
        ),
        "1974 Oil Shock": (
            f"The current setup {strength} resembles an inflation / commodity shock regime. "
            "Historically, hard assets, energy, and gold did well; long-duration bonds and broad equities lagged."
        ),
        "1980 Volcker Shock": (
            f"The current setup {strength} resembles a high-inflation / aggressive-tightening regime. "
            "Historically, cash and short-duration USD assets outperformed; avoid long bonds and gold after inflation peaks."
        ),
        "2011 Euro Crisis": (
            f"The current setup {strength} resembles a sovereign / policy stress regime. "
            "Historically, USD strength, high-quality government bonds, and selective gold exposure worked well."
        ),
        "2020 Pandemic": (
            f"The current setup {strength} resembles a liquidity-shock-then-stimulus regime. "
            "Historically, hold cash early, then rotate toward gold and liquidity-supported risk assets once policy response is clear."
        ),
    }
    return texts.get(
        crisis_name,
        f"The current setup {strength} resembles the selected crisis only partially. "
        "Stay diversified and let the strongest current drivers determine asset allocation.",
    )


def get_top_crisis_contributors(
    latest_row: pd.Series, core_keys: list[str], labels_map: dict, top_n: int = 3
) -> list[dict]:
    items = []
    for k in core_keys:
        v = latest_row.get(f"{k}_CONTRIB", np.nan)
        if pd.notna(v):
            items.append((k, float(v)))
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "key": k,
            "label": labels_map.get(k, k),
            "value": v,
            "direction": "supportive" if v > 0 else ("negative" if v < 0 else "neutral"),
        }
        for k, v in items[:top_n]
    ]


def build_dynamic_crisis_playbook(
    crisis_name: str,
    fit_score: float,
    latest_row: pd.Series,
    core_keys: list[str],
    labels_map: dict,
) -> str:
    top = get_top_crisis_contributors(latest_row, core_keys, labels_map, top_n=3)
    base = get_crisis_investor_playbook(crisis_name, fit_score)

    if not top:
        return base + "\n\nDriver-based note: no current contribution breakdown is available."

    supportive = [x for x in top if x["value"] > 0]
    negative = [x for x in top if x["value"] < 0]

    def fmt_driver(x: dict) -> str:
        return f"{x['label']} ({x['value']:+.2f})"

    lines = ["Driver-based reading of the current setup:"]
    if supportive:
        lines.append("- Main crisis-matching drivers now: " + ", ".join(fmt_driver(x) for x in supportive[:3]) + ".")
    if negative:
        lines.append("- Main offsets / differences vs the crisis template: " + ", ".join(fmt_driver(x) for x in negative[:3]) + ".")

    if pd.notna(fit_score) and fit_score >= 70:
        lines.append("- Similarity is high; treat the top supportive drivers as the main regime clues.")
    elif pd.notna(fit_score) and fit_score >= 45:
        lines.append("- Similarity is moderate; use drivers as a guide but avoid assuming the full crisis path repeats.")
    else:
        lines.append("- Similarity is weak; keep the historical analogy as reference only.")

    if supportive and not negative:
        lines.append("- Positioning idea: tilt toward assets that historically benefited in this regime, scaling in gradually.")
    elif supportive and negative:
        lines.append("- Positioning idea: balanced stance. The crisis analogy is partly present but some drivers are not aligned.")
    elif negative and not supportive:
        lines.append("- Positioning idea: stay defensive. The current setup is not strongly confirming the historical regime.")

    return base + "\n\n" + "\n".join(lines)
