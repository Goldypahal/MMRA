"""
generate_plots.py — Publication-Quality Comparison Graphs Generator for MMRA.

Generates 6 IEEE/ACM paper-ready visual charts:
1. Figure 1: Accuracy by Category × Condition (Grouped Bar Chart)
2. Figure 2: Token Efficiency Pareto Frontier (Accuracy vs Token Cost)
3. Figure 3: Difficulty Interaction (C4-C1 Accuracy Gap by Difficulty)
4. Figure 4: Model Contribution Win-Rate in C4 Debate
5. Figure 5: Failure Mode Taxonomy Breakdown across Conditions (Table II)
6. Figure 6: Core Condition Pair Comparisons (Gain % & Cohen's d Effect Sizes)

Outputs saved as high-DPI PNGs in results/
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.config import RESULTS_DB, CATEGORIES, CONDITIONS, MODELS
from src.database import load_results
from src.analysis import (
    accuracy_table, paired_ttests, all_condition_pair_tests,
    complexity_interaction, token_efficiency, model_contribution,
    failure_mode_breakdown,
)

# Apply sleek modern publication theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
})

COLORS = {
    "C1": "#3182bd",  # Single Model (Blue)
    "C2": "#31a354",  # Self-Consistency (Green)
    "C3": "#e6550d",  # Parallel Vote (Orange)
    "C4": "#756bb1",  # Multi-Agent Debate (Purple)
}

OUT_DIR = HERE / "results"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_fig1_accuracy_by_category(df: pd.DataFrame):
    """Figure 1: Accuracy by Category × Condition (Grouped Bar Chart)"""
    acc_df = accuracy_table(df)
    if acc_df.empty:
        return

    # Exclude AVERAGE row for plotting individual categories
    cats = [c for c in acc_df.index if c != "AVERAGE"]
    cat_names = [CATEGORIES[c].name if c in CATEGORIES else c for c in cats]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cats))
    width = 0.2

    for i, cond in enumerate(CONDITIONS):
        vals = [acc_df.loc[c, cond] * 100 if cond in acc_df.columns else 0 for c in cats]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=f"{cond}: {CONDITIONS[cond].name}", color=COLORS[cond], edgecolor='white', linewidth=1)
        # Value labels above bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Figure 1: Accuracy by Benchmark Category across 4 Conditions')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=15, ha='right')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    path = OUT_DIR / "fig1_accuracy_by_category.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved Figure 1 → {path}")


def plot_fig2_token_efficiency(df: pd.DataFrame):
    """Figure 2: Token Efficiency Pareto Frontier"""
    effs = token_efficiency(df)
    if not effs:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for e in effs:
        cond = e.condition
        color = COLORS.get(cond, "#333333")
        acc = e.avg_accuracy * 100
        tok = e.avg_tokens

        ax.scatter(tok, acc, color=color, s=200, label=f"{cond} ({e.accuracy_per_1k_tokens:.2f} acc/1k tok)", zorder=5)
        ax.annotate(f" {cond}\n ({acc:.1f}%, {tok:,.0f} tok)",
                    xy=(tok, acc), xytext=(10, -5),
                    textcoords="offset points", fontweight='bold', fontsize=10, color=color)

    # Connect Pareto curve
    sorted_effs = sorted(effs, key=lambda x: x.avg_tokens)
    x_coords = [e.avg_tokens for e in sorted_effs]
    y_coords = [e.avg_accuracy * 100 for e in sorted_effs]
    ax.plot(x_coords, y_coords, linestyle='--', color='gray', alpha=0.7, zorder=3)

    ax.set_xlabel('Average Tokens per Task')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Figure 2: Token Efficiency Pareto Curve (Accuracy vs Cost)')
    ax.set_ylim(min(y_coords) - 10, max(y_coords) + 15)
    ax.set_xlim(0, max(x_coords) * 1.15)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    path = OUT_DIR / "fig2_token_efficiency_pareto.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved Figure 2 → {path}")


def plot_fig3_difficulty_interaction(df: pd.DataFrame):
    """Figure 3: Difficulty Interaction (C4-C1 Accuracy Gap)"""
    cx_df = complexity_interaction(df)
    if cx_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    diffs = list(cx_df.index)
    
    if "C1" in cx_df.columns and "C4" in cx_df.columns:
        x = np.arange(len(diffs))
        width = 0.35
        
        c1_vals = [cx_df.loc[d, "C1"] * 100 for d in diffs]
        c4_vals = [cx_df.loc[d, "C4"] * 100 for d in diffs]

        ax.bar(x - width/2, c1_vals, width, label='C1 Single Model', color=COLORS["C1"])
        ax.bar(x + width/2, c4_vals, width, label='C4 Multi-Agent Debate', color=COLORS["C4"])

        # Annotate gap
        for i, d in enumerate(diffs):
            gap = (cx_df.loc[d, "C4"] - cx_df.loc[d, "C1"]) * 100
            ax.annotate(f"Gap: +{gap:.1f}%",
                        xy=(i, max(c1_vals[i], c4_vals[i]) + 3),
                        ha='center', fontweight='bold', color='#756bb1', fontsize=10)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Figure 3: Accuracy Progression by Task Difficulty (C1 vs C4)')
        ax.set_xticks(x)
        ax.set_xticklabels(diffs)
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        path = OUT_DIR / "fig3_difficulty_interaction.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✓ Saved Figure 3 → {path}")


def plot_fig4_model_contribution(df: pd.DataFrame):
    """Figure 4: Model Contribution in C4 Debate"""
    contrib = model_contribution(df)
    if contrib.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    models = contrib["model_name"]
    rates = contrib["win_rate"] * 100

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][:len(models)]
    bars = ax.bar(models, rates, color=colors, edgecolor='white', linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Consensus Win Rate (%)')
    ax.set_title('Figure 4: Model Contribution Win Rate in C4 Debate Rounds')
    ax.set_ylim(0, 115)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    path = OUT_DIR / "fig4_model_contribution.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved Figure 4 → {path}")


def plot_fig5_failure_modes(df: pd.DataFrame):
    """Figure 5: Failure Mode Reduction Breakdown (Table II)"""
    fm_df = failure_mode_breakdown(df)
    if fm_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Plot stacked horizontal bar chart
    fm_df.plot(kind='bar', stacked=True, ax=ax, colormap='Blues_r', edgecolor='white')

    ax.set_ylabel('Failure Mode (%)')
    ax.set_title('Figure 5: Failure Mode Distribution Across Conditions (Table II Taxonomy)')
    ax.set_ylim(0, 115)
    ax.legend(title='Failure Mode', loc='upper right', frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = OUT_DIR / "fig5_failure_mode_breakdown.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved Figure 5 → {path}")


def plot_fig6_condition_pair_comparisons(df: pd.DataFrame):
    """Figure 6: Core Condition Pair Comparisons (Accuracy Gains & Effect Sizes)"""
    pairs = all_condition_pair_tests(df)
    if not pairs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = [p.comparison for p in pairs]
    gains = [p.gain * 100 for p in pairs]
    effect_sizes = [p.cohens_d for p in pairs]

    # Bar 1: Gain %
    bar_colors = ["#3182bd" if g > 0 else "#e6550d" for g in gains]
    bars1 = ax1.barh(names, gains, color=bar_colors, edgecolor='white')
    ax1.set_xlabel('Accuracy Gain (%)')
    ax1.set_title('Core Condition Comparison Gains (%)')
    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f'{w:+.1f}%',
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5 if w >= 0 else -25, 0), textcoords="offset points",
                     va='center', fontweight='bold', fontsize=9)

    # Bar 2: Cohen's d Effect Size
    bars2 = ax2.barh(names, effect_sizes, color='#756bb1', edgecolor='white')
    ax2.set_xlabel("Cohen's d Effect Size")
    ax2.set_title("Effect Size (Cohen's d)")
    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f'{w:.2f}',
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     va='center', fontweight='bold', fontsize=9)

    fig.suptitle('Figure 6: Core Paper Condition Pair Hypotheses Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / "fig6_condition_pair_gaps.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ Saved Figure 6 → {path}")


def main():
    print("Generating publication-ready comparison graphs for MMRA...")
    df = load_results()
    if df.empty:
        print("[!] No results found in results.db. Run experiment first.")
        return

    plot_fig1_accuracy_by_category(df)
    plot_fig2_token_efficiency(df)
    plot_fig3_difficulty_interaction(df)
    plot_fig4_model_contribution(df)
    plot_fig5_failure_modes(df)
    plot_fig6_condition_pair_comparisons(df)
    print("\n✓ All 6 comparison graphs generated and saved in results/!")


if __name__ == "__main__":
    main()
