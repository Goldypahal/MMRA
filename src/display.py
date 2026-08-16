"""
display.py — Rich terminal rendering for all outputs.
Tables, progress bars, debate logs, statistical summaries.
"""

import sys
import io
import os
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TextColumn
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box

from src.config import MODELS, CONDITIONS, CATEGORIES
from src.conditions import TaskResult
from src.analysis import PairedTTest, TokenEfficiency

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False)

COND_STYLES = {
    "C1": "bold blue",
    "C2": "bold cyan",
    "C3": "bold yellow",
    "C4": "bold green",
}

MODEL_STYLES = {
    "A": "bold blue",
    "B": "bold green",
    "C": "bold yellow",
    "D": "bold red",
}


# ─────────────────────────────────────────────────────────────────────────────
# Banners
# ─────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    console.print(Panel.fit(
        "[bold white]Multi-Agent vs Single-Model Research Framework[/]\n"
        "[dim]Research Paper 1 — Empirical Study[/]\n"
        "[dim]DeepSeek-R1 · Gemma-4 · Qwen3 · Llama-4  |  4 Conditions × 7 Categories × 140 Tasks[/]",
        border_style="blue",
        title="[bold cyan]MultiModel[/]",
        subtitle="[dim]OpenRouter Free Tier[/]",
    ))


def print_section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold white]{title}[/]", style="dim blue"))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Live task result
# ─────────────────────────────────────────────────────────────────────────────

def print_task_result(result: TaskResult, task_text: str = "") -> None:
    cond_style = COND_STYLES.get(result.condition, "white")
    score_color = "green" if result.score >= 0.8 else "yellow" if result.score >= 0.5 else "red"

    lines = []
    if task_text:
        short_q = task_text[:80] + ("..." if len(task_text) > 80 else "")
        lines.append(f"[dim]Q: {short_q}[/]")
    lines.append(
        f"[{cond_style}]{result.condition}[/] [{score_color}]Score: {result.score:.2f}[/]  "
        f"[dim]Tokens: {result.tokens_total:,}  Latency: {result.latency_ms:.0f}ms[/]"
    )
    lines.append(
        f"[dim]Answer: {result.final_answer[:100]}[/]"
    )
    console.print("\n".join(lines))


def print_debate_round(task_id: int, round_num: int, model_id: str, response: str) -> None:
    model = MODELS[model_id]
    style = MODEL_STYLES.get(model_id, "white")
    short_resp = response[:200] + ("..." if len(response) > 200 else "")
    console.print(
        f"  [{style}]{model.short}[/] [dim](R{round_num})[/]: {short_resp}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar factory
# ─────────────────────────────────────────────────────────────────────────────

def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy table
# ─────────────────────────────────────────────────────────────────────────────

def print_accuracy_table(acc_df: pd.DataFrame) -> None:
    if acc_df.empty:
        console.print("[yellow]No results yet.[/]")
        return

    table = Table(
        title="[bold white]Accuracy by Category × Condition[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Category", style="bold white", min_width=20)
    for cond_id, cond in CONDITIONS.items():
        table.add_column(f"{cond_id}\n{cond.name[:10]}", style=COND_STYLES[cond_id],
                         justify="center", min_width=12)
    if "C4-C1 Gap" in acc_df.columns or "C4" in acc_df.columns:
        table.add_column("C4-C1", style="bold green", justify="center", min_width=8)

    cat_display = {c: CATEGORIES[c].name if c in CATEGORIES else c for c in acc_df.index}

    for idx, row in acc_df.iterrows():
        cells = [cat_display.get(idx, str(idx))]
        for cond_id in CONDITIONS:
            val = row.get(cond_id, float("nan"))
            if pd.isna(val):
                cells.append("[dim]—[/]")
            else:
                pct = f"{val*100:.1f}%"
                if val >= 0.80:
                    cells.append(f"[bold green]{pct}[/]")
                elif val >= 0.60:
                    cells.append(f"[yellow]{pct}[/]")
                else:
                    cells.append(f"[red]{pct}[/]")
        # Gap
        c1 = row.get("C1", float("nan"))
        c4 = row.get("C4", float("nan"))
        if not pd.isna(c1) and not pd.isna(c4):
            gap = (c4 - c1) * 100
            gstr = f"+{gap:.1f}%" if gap >= 0 else f"{gap:.1f}%"
            cells.append(f"[{'green' if gap > 0 else 'red'}]{gstr}[/]")
        else:
            cells.append("[dim]—[/]")

        style = "on grey23" if str(idx) == "AVERAGE" else ""
        table.add_row(*cells, style=style)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests table
# ─────────────────────────────────────────────────────────────────────────────

def print_stats_table(tests: list[PairedTTest]) -> None:
    if not tests:
        console.print("[yellow]No statistical tests computed yet.[/]")
        return

    table = Table(
        title="[bold white]Paired t-test: C1 vs C4 (Bonferroni corrected)[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Category",     style="white",     min_width=18)
    table.add_column("C1 Mean",      justify="right",   min_width=8)
    table.add_column("C4 Mean",      justify="right",   min_width=8)
    table.add_column("Gain",         justify="right",   min_width=8, style="bold green")
    table.add_column("t-stat",       justify="right",   min_width=8)
    table.add_column("p (Bonf.)",    justify="right",   min_width=10)
    table.add_column("Cohen's d",    justify="right",   min_width=10)
    table.add_column("Effect",       justify="center",  min_width=10)
    table.add_column("Sig.",         justify="center",  min_width=6)

    for t in tests:
        from src.analysis import effect_size_label
        eff = effect_size_label(t.cohens_d)
        eff_color = {"Large": "green", "Medium": "yellow", "Small": "cyan", "Negligible": "dim"}.get(eff, "white")
        sig_str = "[bold green]***[/]" if t.p_bonferroni < 0.001 else \
                  "[green]**[/]"  if t.p_bonferroni < 0.01  else \
                  "[yellow]*[/]"  if t.p_bonferroni < 0.05  else \
                  "[dim]ns[/]"
        delta = f"+{t.improvement*100:.1f}%" if t.improvement >= 0 else f"{t.improvement*100:.1f}%"
        table.add_row(
            CATEGORIES[t.category].name if t.category in CATEGORIES else t.category,
            f"{t.c1_mean*100:.1f}%",
            f"{t.c4_mean*100:.1f}%",
            delta,
            f"{t.t_stat:.3f}",
            f"{t.p_bonferroni:.4f}",
            f"{t.cohens_d:.3f}",
            f"[{eff_color}]{eff}[/]",
            sig_str,
        )
    console.print(table)
    console.print("[dim]  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns=not significant[/]")
    console.print(f"[dim]  Bonferroni α = 0.05 / {len(tests)} = {0.05/len(tests):.4f} per test[/]")


def print_condition_pair_table(pair_tests) -> None:
    if not pair_tests:
        return

    table = Table(
        title="[bold white]Core Condition Comparisons (Paper Hypotheses)[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Comparison",      style="bold white", min_width=12)
    table.add_column("Research Hypothesis / Isolation", style="dim white", min_width=32)
    table.add_column("Cond A",          justify="right",   min_width=8)
    table.add_column("Cond B",          justify="right",   min_width=8)
    table.add_column("Gain",            justify="right",   min_width=8, style="bold green")
    table.add_column("t-stat",          justify="right",   min_width=8)
    table.add_column("p (Bonf.)",       justify="right",   min_width=10)
    table.add_column("Cohen's d",       justify="right",   min_width=10)
    table.add_column("Effect",          justify="center",  min_width=10)
    table.add_column("Sig.",            justify="center",  min_width=6)

    for p in pair_tests:
        from src.analysis import effect_size_label
        eff = effect_size_label(p.cohens_d)
        eff_color = {"Large": "green", "Medium": "yellow", "Small": "cyan", "Negligible": "dim"}.get(eff, "white")
        sig_str = "[bold green]***[/]" if p.p_bonferroni < 0.001 else \
                  "[green]**[/]"  if p.p_bonferroni < 0.01  else \
                  "[yellow]*[/]"  if p.p_bonferroni < 0.05  else \
                  "[dim]ns[/]"
        delta = f"+{p.gain*100:.1f}%" if p.gain >= 0 else f"{p.gain*100:.1f}%"
        table.add_row(
            p.comparison,
            p.description,
            f"{p.mean_a*100:.1f}%",
            f"{p.mean_b*100:.1f}%",
            delta,
            f"{p.t_stat:.3f}",
            f"{p.p_bonferroni:.4f}",
            f"{p.cohens_d:.3f}",
            f"[{eff_color}]{eff}[/]",
            sig_str,
        )
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Token efficiency table
# ─────────────────────────────────────────────────────────────────────────────

def print_token_efficiency(efficiencies: list[TokenEfficiency]) -> None:
    if not efficiencies:
        return

    table = Table(
        title="[bold white]Token Efficiency — Accuracy per 1,000 Tokens[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
    )
    table.add_column("Condition",       style="white",    min_width=20)
    table.add_column("Avg Accuracy",    justify="right",  min_width=13)
    table.add_column("Avg Tokens",      justify="right",  min_width=12)
    table.add_column("Acc/1K Tokens",   justify="right",  min_width=13, style="bold cyan")
    table.add_column("vs C1 Tokens",    justify="right",  min_width=13)

    baseline_tokens = next((e.avg_tokens for e in efficiencies if e.condition == "C1"), 1.0)

    for e in sorted(efficiencies, key=lambda x: x.accuracy_per_1k_tokens, reverse=True):
        cond = CONDITIONS[e.condition]
        ratio = e.avg_tokens / baseline_tokens if baseline_tokens else 1
        ratio_str = f"{ratio:.1f}×"
        table.add_row(
            f"[{COND_STYLES[e.condition]}]{e.condition}[/] {cond.name}",
            f"{e.avg_accuracy*100:.1f}%",
            f"{e.avg_tokens:,.0f}",
            f"{e.accuracy_per_1k_tokens:.4f}",
            ratio_str,
        )
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Complexity interaction table
# ─────────────────────────────────────────────────────────────────────────────

def print_complexity_table(cx_df: pd.DataFrame) -> None:
    if cx_df.empty:
        return

    table = Table(
        title="[bold white]Difficulty × Condition Accuracy[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
    )
    table.add_column("Difficulty", style="bold white", min_width=10)
    for col in cx_df.columns:
        style = COND_STYLES.get(col, "cyan") if col != "C4-C1 Gap" else "bold green"
        table.add_column(col, justify="center", min_width=10, style=style)

    for diff, row in cx_df.iterrows():
        cells = [str(diff)]
        for col in cx_df.columns:
            val = row.get(col, float("nan"))
            if pd.isna(val):
                cells.append("—")
            elif col == "C4-C1 Gap":
                sign = "+" if val >= 0 else ""
                cells.append(f"[{'green' if val >= 0 else 'red'}]{sign}{val*100:.1f}%[/]")
            else:
                cells.append(f"{val*100:.1f}%")
        table.add_row(*cells)
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Progress summary (experiment status)
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_status(stats: dict) -> None:
    if not stats:
        console.print("[yellow]No results in database yet.[/]")
        return
    table = Table(title="[bold white]Experiment Progress[/]", box=box.SIMPLE_HEAD, border_style="dim")
    table.add_column("Condition", style="white", min_width=20)
    table.add_column("Completed", justify="right", min_width=10)
    table.add_column("Avg Score", justify="right", min_width=10)
    table.add_column("Total Tokens", justify="right", min_width=14)
    for cond_id, s in stats.items():
        cond = CONDITIONS.get(cond_id, None)
        name = f"[{COND_STYLES.get(cond_id,'white')}]{cond_id}[/] {cond.name if cond else ''}"
        table.add_row(name, str(s["n"]), f"{s['avg_score']*100:.1f}%", f"{s['total_tokens']:,}")
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Model contribution
# ─────────────────────────────────────────────────────────────────────────────

def print_model_contribution(contrib_df: pd.DataFrame) -> None:
    if contrib_df.empty:
        return
    table = Table(
        title="[bold white]Model Contribution in C4 (Who Scores Best Most Often?)[/]",
        box=box.SIMPLE_HEAD, border_style="dim",
    )
    table.add_column("Rank", justify="center", min_width=6)
    table.add_column("Model", style="white", min_width=18)
    table.add_column("Wins", justify="right", min_width=8)
    table.add_column("Win Rate", justify="right", min_width=10)

    medals = ["🥇", "🥈", "🥉", "4️⃣ "]
    for i, row in contrib_df.iterrows():
        mid = row["model_id"]
        style = MODEL_STYLES.get(mid, "white")
        table.add_row(
            medals[i] if i < 4 else str(i + 1),
            f"[{style}]{row['model_name']}[/]",
            str(row["wins"]),
            f"{row['win_rate']*100:.1f}%",
        )
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode table (Table II in Paper 1)
# ─────────────────────────────────────────────────────────────────────────────

def print_failure_mode_table(fm_df: pd.DataFrame) -> None:
    if fm_df.empty:
        return
    table = Table(
        title="[bold white]Table II — Failure Mode Distribution by Condition (%)[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
    )
    table.add_column("Condition", style="bold white", min_width=12)
    for col in fm_df.columns:
        table.add_column(col, justify="right", min_width=15)

    for cond, row in fm_df.iterrows():
        style = COND_STYLES.get(cond, "white")
        cells = [f"[{style}]{cond}[/]"]
        for col in fm_df.columns:
            val = row.get(col, 0.0)
            cells.append(f"{val:.1f}%")
        table.add_row(*cells)
    console.print(table)
