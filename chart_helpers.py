"""
chart_helpers.py
----------------
Matplotlib figure builders for reports and the Streamlit UI.
No Streamlit dependency.
"""
from __future__ import annotations

import tempfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Utility ───────────────────────────────────────────────────────────────────

def save_fig_to_tempfile(fig) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


# ── Threshold helpers ─────────────────────────────────────────────────────────

def add_indicator_threshold_lines(
    ax,
    ind_key: str,
    thresholds: dict,
    mode: str,
    color_bull: str = "green",
    color_bear: str = "red",
) -> None:
    if thresholds is None or ind_key not in thresholds:
        return
    thr = thresholds[ind_key]
    if isinstance(thr, dict):
        bull = thr.get("bull")
        bear = thr.get("bear")
    else:
        bull, bear = thr[0], thr[1]

    if bull is not None and pd.notna(bull):
        ax.axhline(float(bull), linestyle="--", linewidth=1.0, color=color_bull, alpha=0.85, label="Bull threshold")
    if bear is not None and pd.notna(bear):
        ax.axhline(float(bear), linestyle="--", linewidth=1.0, color=color_bear, alpha=0.85, label="Bear threshold")


# ── Indicator figures ─────────────────────────────────────────────────────────

def build_indicator_figure(
    res_plot: pd.DataFrame,
    ind_key: str,
    thresholds: dict,
    mode: str,
    labels_dict: dict,
):
    if ind_key not in res_plot.columns:
        return None
    s = res_plot[ind_key].dropna()
    if s.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(s.index, s.values, linewidth=1.5, label=labels_dict.get(ind_key, ind_key))
    try:
        add_indicator_threshold_lines(ax, ind_key, thresholds, mode)
    except Exception:
        pass
    ax.set_xlabel("Date")
    ax.set_ylabel(labels_dict.get(ind_key, ind_key))
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig


def build_indicator_history_figure(
    res_plot: pd.DataFrame,
    key: str,
    label: str,
    thresholds: dict,
    mode: str,
):
    if key not in res_plot.columns:
        return None
    s = res_plot[key].dropna()
    if s.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(s.index, s.values, label=label)
    if thresholds and key in thresholds:
        add_indicator_threshold_lines(ax, key, thresholds, mode)
    ax.set_title(label)
    ax.set_xlabel("Date")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)
    return fig


# ── Signal / Gold / Contribution charts ──────────────────────────────────────

def build_signal_figure(res_plot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(res_plot.index, res_plot["SIGNAL"].values, label="Signal", color="tab:blue")
    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_xlabel("Date")
    ax.set_ylabel("Signal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig


def build_gold_signal_figure(res_plot: pd.DataFrame):
    gold_col = next((c for c in ["GOLD_USD", "GC=F", "GLD"] if c in res_plot.columns), None)
    if gold_col is None:
        return None

    plot_df = res_plot[[gold_col, "SIGNAL"]].dropna()
    if plot_df.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(8, 3.0))
    ax1.plot(plot_df.index, plot_df[gold_col].values, color="gold", linewidth=1.8, label="Gold (USD)")
    ax2 = ax1.twinx()
    ax2.plot(plot_df.index, plot_df["SIGNAL"].values, color="tab:blue", linewidth=1.2,
             linestyle="--", alpha=0.8, label="Signal")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Gold (USD)", color="goldenrod")
    ax2.set_ylabel("Signal", color="tab:blue")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="best", fontsize=8)
    ax1.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig


def build_contrib_figure(res_plot: pd.DataFrame):
    contrib_cols = [c for c in res_plot.columns if c.endswith("_CONTRIB")]
    if not contrib_cols:
        return None

    data = res_plot[contrib_cols].dropna(how="all")
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 3.0))
    pos = data.clip(lower=0)
    neg = data.clip(upper=0)
    bottom_pos = np.zeros(len(data))
    bottom_neg = np.zeros(len(data))

    for col in contrib_cols:
        label = col.replace("_CONTRIB", "")
        ax.bar(data.index, pos[col].values, bottom=bottom_pos, label=label, width=25)
        ax.bar(data.index, neg[col].values, bottom=bottom_neg, width=25)
        bottom_pos += pos[col].fillna(0).values
        bottom_neg += neg[col].fillna(0).values

    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_xlabel("Date")
    ax.set_ylabel("Contribution")
    ax.legend(loc="best", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def build_current_contributions_bar_figure(
    latest_row: pd.Series,
    core_keys: list,
    labels_map: dict,
    title: str = "Current Contribution by Indicator",
):
    items = []
    for k in core_keys:
        c = latest_row.get(f"{k}_CONTRIB", np.nan)
        if pd.notna(c):
            items.append((labels_map.get(k, k), float(c)))

    if not items:
        return None

    items.sort(key=lambda x: abs(x[1]), reverse=True)
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = ["green" if v > 0 else "red" for v in vals]

    fig, ax = plt.subplots(figsize=(6, max(3.0, len(names) * 0.45)))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, linewidth=0.8, color="black")
    ax.set_xlabel("Contribution")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig


# ── Crisis comparison figure ──────────────────────────────────────────────────

def build_now_vs_crisis_vertical_figure(
    current_series: pd.Series,
    crisis_series: pd.Series,
    current_label: str,
    crisis_label: str,
    title: str,
    sim_meta: dict,
    higher_is_bullish: Optional[bool] = None,
):
    cur = pd.to_numeric(current_series, errors="coerce").dropna().copy()
    cri = pd.to_numeric(crisis_series, errors="coerce").dropna().copy()

    if cur.empty or cri.empty:
        return None

    median = sim_meta.get("median", np.nan)
    q33 = sim_meta.get("q33", np.nan)
    q67 = sim_meta.get("q67", np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharey=True)
    plt.subplots_adjust(hspace=0.25)

    all_vals = pd.concat([cur, cri])
    y_min = float(all_vals.min())
    y_max = float(all_vals.max())
    pad = 0.08 * (y_max - y_min) if y_max > y_min else 1.0

    def add_zones(ax):
        if pd.isna(q33) or pd.isna(q67):
            return
        if higher_is_bullish is True:
            ax.axhspan(ax.get_ylim()[0], q33, alpha=0.08, color="red")
            ax.axhspan(q33, q67, alpha=0.08, color="gold")
            ax.axhspan(q67, ax.get_ylim()[1], alpha=0.08, color="green")
        elif higher_is_bullish is False:
            ax.axhspan(ax.get_ylim()[0], q33, alpha=0.08, color="green")
            ax.axhspan(q33, q67, alpha=0.08, color="gold")
            ax.axhspan(q67, ax.get_ylim()[1], alpha=0.08, color="red")
        else:
            ax.axhspan(q33, q67, alpha=0.08, color="gold")

    for ax, series, label, linestyle in [
        (ax1, cur, current_label, "solid"),
        (ax2, cri, crisis_label, "dashed"),
    ]:
        ax.plot(series.index, series.values, linewidth=1.4, linestyle=linestyle, label=label)
        ax.scatter(series.index[-1], series.values[-1], s=25, zorder=3)
        ax.set_ylim(y_min - pad, y_max + pad)
        add_zones(ax)
        if pd.notna(median):
            ax.axhline(median, linestyle="--", linewidth=1.0, label="Crisis median")
        ax.set_title("Current window" if ax is ax1 else "Historical crisis window", fontsize=11)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig
