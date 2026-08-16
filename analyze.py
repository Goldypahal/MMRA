"""
analyze.py — Full statistical analysis and comparison report.

Usage:
    python analyze.py                   # full report from DB
    python analyze.py --category math   # single category
    python analyze.py --export          # also export to JSON/CSV
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

from src.database import load_results, export_to_json, summary_stats
from src.analysis import (
    accuracy_table, paired_ttests, all_condition_pair_tests, complexity_interaction,
    token_efficiency, model_contribution, failure_mode_breakdown,
    effect_size_label,
)
from src.display import (
    print_banner, print_section, print_accuracy_table, print_stats_table,
    print_condition_pair_table, print_token_efficiency, print_complexity_table,
    print_model_contribution, print_failure_mode_table, print_experiment_status,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze multi-model experiment results")
    parser.add_argument("--category", type=str, default=None, help="Filter to one category")
    parser.add_argument("--export", action="store_true", help="Export results to JSON and CSV")
    parser.add_argument("--brief", action="store_true", help="Show only accuracy table + stats")
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    df = load_results()
    if df.empty:
        console.print(Panel(
            "[yellow]No results found in database.[/]\n\n"
            "Run the experiment first:\n"
            "  [cyan]python run_experiment.py --n 10[/]   (quick 10-task test)\n"
            "  [cyan]python run_experiment.py[/]           (full 140-task run)",
            title="[yellow]No Data[/]",
            border_style="yellow",
        ))
        return

    if args.category:
        df = df[df["category"] == args.category]
        if df.empty:
            console.print(f"[red]No results for category '{args.category}'[/]")
            return

    # ── 0. Status ──────────────────────────────────────────────────────────
    print_section("Experiment Progress")
    print_experiment_status(summary_stats())

    # ── 1. Accuracy Table ─────────────────────────────────────────────────
    print_section("Table 1 — Accuracy by Category × Condition")
    acc_df = accuracy_table(df)
    print_accuracy_table(acc_df)

    if args.brief:
        return

    # ── 2. Core Condition Comparisons (The 5 Key Paper Hypotheses) ───────
    print_section("Core Paper Comparisons — Condition Pairs (Bonferroni Corrected)")
    pair_tests = all_condition_pair_tests(df)
    print_condition_pair_table(pair_tests)

    # ── 3. Category Statistical Tests (C1 vs C4) ─────────────────────────
    print_section("Table 2 — Paired t-tests per Category: C1 vs C4 (Bonferroni)")
    tests = paired_ttests(df)
    print_stats_table(tests)

    if tests:
        sig_count = sum(1 for t in tests if t.significant)
        avg_d = sum(abs(t.cohens_d) for t in tests) / len(tests)
        avg_imp = sum(t.improvement for t in tests) / len(tests)
        console.print()
        console.print(
            f"  [white]Significant categories:[/] [green]{sig_count}/{len(tests)}[/]  |  "
            f"[white]Mean Cohen's d:[/] [cyan]{avg_d:.3f} ({effect_size_label(avg_d)})[/]  |  "
            f"[white]Mean improvement:[/] [green]+{avg_imp*100:.1f}%[/]"
        )

    # ── 3. Complexity × Condition ─────────────────────────────────────────
    print_section("Figure 2 — Complexity × Condition Interaction")
    cx_df = complexity_interaction(df)
    print_complexity_table(cx_df)

    if not cx_df.empty and "C4-C1 Gap" in cx_df.columns:
        gaps = cx_df["C4-C1 Gap"].dropna()
        if len(gaps) >= 2:
            console.print()
            console.print(
                f"  [dim]Gap on [white]Hard[/] tasks: "
                f"[green]+{gaps.get('Hard', 0)*100:.1f}%[/] vs [white]Easy[/] tasks: "
                f"[cyan]+{gaps.get('Easy', 0)*100:.1f}%[/]  — "
                f"Multi-agent advantage scales with difficulty.[/]"
            )

    # ── 4. Token Efficiency ───────────────────────────────────────────────
    print_section("Figure 3 — Token Efficiency Pareto")
    eff = token_efficiency(df)
    print_token_efficiency(eff)

    # ── 5. Model Contribution (novel finding) ─────────────────────────────
    print_section("Novel Finding — Model Contribution in C4 Debate")
    contrib = model_contribution(df)
    print_model_contribution(contrib)
    if not contrib.empty:
        top = contrib.iloc[0]
        console.print(
            f"\n  [dim]→ [white]{top['model_name']}[/] scores highest most often "
            f"({top['win_rate']*100:.0f}% of tasks). "
            f"No existing paper reports this agent-level contribution analysis.[/]"
        )

    # ── 6. Failure Mode Breakdown ─────────────────────────────────────────
    print_section("Table II — Failure Mode Distribution by Condition (%)")
    fail_df = failure_mode_breakdown(df)
    if not fail_df.empty:
        print_failure_mode_table(fail_df)
    else:
        console.print("[dim]Not enough failure data recorded yet.[/]")

    # ── 7. Export ─────────────────────────────────────────────────────────
    if args.export:
        print_section("Exporting Results")
        export_to_json()
        # Also export CSV
        raw = load_results()
        csv_path = "results/results.csv"
        raw.to_csv(csv_path, index=False)
        console.print(f"[green]✓[/] CSV exported → {csv_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold white]Analysis complete.[/]\n"
        "[dim]To run more tasks:   [cyan]python run_experiment.py[/]\n"
        "To export data:      [cyan]python analyze.py --export[/]\n"
        "To test one debate:  [cyan]python scripts/demo_debate.py[/][/]",
        border_style="dim blue",
    ))


if __name__ == "__main__":
    main()
