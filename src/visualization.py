"""
visualization.py
=================
Reusable exploratory-data-analysis (EDA) plotting helpers built on
matplotlib/seaborn: distribution plots (box/violin/histogram), bivariate
relationships (scatter/joint/pairplot), and a one-call summary report.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_boxplot(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                  palette: str = "Blues", ax=None):
    ax = ax or plt.gca()
    sns.boxplot(x=x, y=y, hue=hue, data=df, palette=palette, ax=ax)
    ax.set_title(f"{y} by {x}")
    return ax


def plot_violin(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, ax=None):
    ax = ax or plt.gca()
    sns.violinplot(x=x, y=y, hue=hue, data=df, ax=ax)
    ax.set_title(f"{y} distribution by {x}")
    return ax


def plot_histogram(series: pd.Series, bins: int = 20, cumulative: bool = False, ax=None):
    ax = ax or plt.gca()
    ax.hist(series.dropna(), bins=bins, cumulative=cumulative)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count" if not cumulative else "Cumulative count")
    return ax


def plot_scatter_with_fit(df: pd.DataFrame, x: str, y: str, fit_reg: bool = True, ax=None):
    ax = ax or plt.gca()
    sns.regplot(x=df[x], y=df[y], fit_reg=fit_reg, ax=ax)
    return ax


def plot_pairwise(df: pd.DataFrame, hue: Optional[str] = None):
    """Full pairwise correlogram (own figure; returns the seaborn PairGrid)."""
    return sns.pairplot(df, hue=hue)


def eda_report(df: pd.DataFrame, numeric_cols: Optional[list[str]] = None,
               category_col: Optional[str] = None):
    """Produce a compact multi-panel EDA figure: histograms for each
    numeric column, plus a boxplot grouped by ``category_col`` if given.
    Returns the matplotlib Figure.
    """
    numeric_cols = numeric_cols or df.select_dtypes("number").columns.tolist()
    n = len(numeric_cols) + (1 if category_col else 0)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        plot_histogram(df[col], ax=axes[i])
        axes[i].set_title(col)

    if category_col:
        plot_boxplot(df, x=category_col, y=numeric_cols[0], ax=axes[len(numeric_cols)])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig
