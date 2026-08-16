"""
run_experiment.py — Main experiment runner.

Usage:
    python run_experiment.py                        # all 140 tasks, all 4 conditions
    python run_experiment.py --conditions C1 C4     # specific conditions
    python run_experiment.py --n 10                 # first 10 tasks only (quick test)
    python run_experiment.py --category math        # single category
    python run_experiment.py --resume               # skip already completed
    python run_experiment.py --model A              # for C1/C2: which model to use
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

from src.config import CONDITIONS, CATEGORIES, MODELS
from src.tasks import ALL_TASKS, get_tasks_by_category, get_task_subset, get_dataset
from src.conditions import run_condition, TaskResult
from src.database import init_db, save_result, get_completed_combos, summary_stats, clear_db
from src.display import (
    print_banner, print_section, print_task_result,
    make_progress, print_experiment_status,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False)

# Semaphore limits concurrent API calls (avoid rate-limits)
CONCURRENCY = 3


async def run_tasks(
    tasks,
    conditions: list[str],
    model_id: str = "A",
    resume: bool = True,
    show_detail: bool = True,
) -> list[TaskResult]:
    """Run all (task × condition) combos with concurrency control."""

    if not resume:
        clear_db()
        completed = set()
    else:
        completed = get_completed_combos()
    init_db()

    todo = [
        (task, cond)
        for task in tasks
        for cond in conditions
        if (task.id, cond) not in completed
    ]

    total = len(todo)
    if total == 0:
        console.print("[green]✓ All tasks already completed. Use --no-resume to re-run.[/]")
        return []

    console.print(f"[cyan]Tasks to run:[/] {total}  "
                  f"[dim](skipped {len(tasks)*len(conditions) - total} already completed)[/]")

    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[TaskResult] = []
    failed = 0

    async def run_one(task, cond):
        nonlocal failed
        async with sem:
            try:
                result = await run_condition(task, cond, model_id)
                save_result(result)
                results.append(result)
                return result
            except Exception as e:
                failed += 1
                console.print(f"  [red]✗ Task {task.id} {cond}: {e}[/]")
                return None

    with make_progress() as prog:
        main_task = prog.add_task(
            f"[cyan]Running {len(conditions)} condition(s) on {len(tasks)} tasks...",
            total=total,
        )

        coroutines = [run_one(task, cond) for task, cond in todo]

        for coro in asyncio.as_completed(coroutines):
            result = await coro
            prog.advance(main_task)
            if result and show_detail:
                task_obj = next((t for t in tasks if t.id == result.task_id), None)
                print_task_result(result, task_obj.text if task_obj else "")

    console.print(f"\n[green]✓ Done.[/] {len(results)} results saved. "
                  f"{'[red]' + str(failed) + ' failed.[/]' if failed else ''}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent vs Single-Model Research Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py                        Run full experiment (140 tasks × 4 conditions)
  python run_experiment.py --n 5                  Quick smoke-test (5 tasks, all conditions)
  python run_experiment.py --conditions C1 C4     Only baseline + debate
  python run_experiment.py --category math        Math tasks only
  python run_experiment.py --resume               Skip already-done tasks (default)
  python run_experiment.py --no-resume            Re-run everything
        """,
    )
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()),
                        help="Which conditions to run (default: all)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N tasks (for quick testing)")
    parser.add_argument("--dataset", type=str, default="standard",
                        choices=["standard", "extended", "all"],
                        help="Which dataset to run: 'standard' (140 tasks), 'extended' (70 adversarial tasks), or 'all' (210 tasks)")
    parser.add_argument("--category", type=str, default=None,
                        choices=list(CATEGORIES.keys()),
                        help="Run only tasks from this category")
    parser.add_argument("--model", type=str, default="A",
                        choices=list(MODELS.keys()),
                        help="Model to use for C1/C2 conditions (default: A = DeepSeek-R1)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-completed tasks (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-run all tasks even if already done")
    parser.add_argument("--quiet", action="store_true",
                        help="Don't print individual task results")
    return parser.parse_args()


async def main():
    print_banner()
    args = parse_args()

    # Validate API key
    from src.config import OPENROUTER_API_KEY
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_key_here":
        console.print(Panel(
            "[bold red]OPENROUTER_API_KEY not set![/]\n\n"
            "1. Copy [cyan].env.example[/] → [cyan].env[/]\n"
            "2. Get a free key from [link=https://openrouter.ai]https://openrouter.ai[/link]\n"
            "3. Add it to [cyan].env[/]",
            title="[red]Configuration Error[/]",
            border_style="red",
        ))
        sys.exit(1)

    # Select tasks
    if args.category:
        tasks = get_tasks_by_category(args.category, dataset=args.dataset)
        if args.n:
            tasks = tasks[:args.n]
    else:
        if args.n:
            tasks = get_task_subset(args.n, dataset=args.dataset)
        else:
            tasks = get_dataset(args.dataset)

    # Show plan
    print_section("Experiment Plan")
    console.print(f"  [white]Tasks:[/]      {len(tasks)} tasks")
    console.print(f"  [white]Conditions:[/] {', '.join(args.conditions)}")
    console.print(f"  [white]C1/C2 Model:[/] [{MODELS[args.model].color}]{MODELS[args.model].name}[/]")
    console.print(f"  [white]Resume:[/]     {'Yes (skipping completed)' if args.resume else 'No (re-running all)'}")
    console.print(f"  [white]API Calls:[/]  ~{estimate_calls(len(tasks), args.conditions):,} calls")
    console.print()

    # Show current status
    status = summary_stats()
    if status:
        print_section("Current Database Status")
        print_experiment_status(status)

    # Run
    print_section("Running Experiment")
    t0 = time.perf_counter()
    await run_tasks(
        tasks=tasks,
        conditions=args.conditions,
        model_id=args.model,
        resume=args.resume,
        show_detail=not args.quiet,
    )
    elapsed = time.perf_counter() - t0

    # Final status
    print_section("Final Status")
    print_experiment_status(summary_stats())
    console.print(f"\n[dim]Total wall time: {elapsed/60:.1f} min[/]")
    console.print("[dim]Run [bold]python analyze.py[/] to see full statistical analysis.[/]")


def estimate_calls(n_tasks: int, conditions: list[str]) -> int:
    """Rough API call count estimate."""
    mapping = {"C1": 1, "C2": 3, "C3": 4, "C4": 8}
    return sum(n_tasks * mapping.get(c, 1) for c in conditions)


if __name__ == "__main__":
    asyncio.run(main())
