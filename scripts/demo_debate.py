"""
scripts/demo_debate.py — Watch a single C4 debate live in the terminal.

Usage:
    python scripts/demo_debate.py                       # random task
    python scripts/demo_debate.py --task_id 15          # specific task
    python scripts/demo_debate.py --category math       # random math task
"""

import asyncio
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.config import MODELS
from src.tasks import COMBINED_TASKS, ALL_TASKS, get_tasks_by_category, get_dataset
from src.client import call_model, call_all_models
from src.graders import extract_final_answer, score_response
from src.display import print_banner, MODEL_STYLES

console = Console()

DEBATE_SYSTEM = (
    "You are a precise reasoning agent in a collaborative debate. "
    "Think step by step, state your reasoning, and end with 'Final answer: <answer>'."
)


async def watch_debate(task_id: int):
    task = next((t for t in COMBINED_TASKS if t.id == task_id), None)
    if not task:
        console.print(f"[red]Task {task_id} not found.[/]")
        return

    print_banner()
    console.print()
    console.print(Panel(
        f"[bold white]{task.text}[/]\n\n"
        f"[dim]Category: {task.category}  |  Difficulty: {task.difficulty}  |  "
        f"Ground Truth: {task.answer}[/]",
        title=f"[cyan]Task #{task.id}[/]",
        border_style="cyan",
    ))

    # ── Round 1 ────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold yellow]🔵 Round 1 — Independent Answers[/]", style="yellow"))
    console.print()

    prompt_r1 = f"Question: {task.text}\n\nAnswer step by step. End with 'Final answer: <answer>'."

    r1_tasks = {
        mid: call_model(mid, prompt_r1, temperature=0.0, system_prompt=DEBATE_SYSTEM)
        for mid in MODELS
    }

    round1 = {}
    console.print("[dim]Calling all 4 models in parallel...[/]\n")
    r1_resps = await asyncio.gather(*r1_tasks.values(), return_exceptions=True)
    r1_results = dict(zip(r1_tasks.keys(), r1_resps))

    for mid, resp in r1_results.items():
        model = MODELS[mid]
        style = MODEL_STYLES.get(mid, "white")
        if hasattr(resp, "success") and resp.success:
            ans = extract_final_answer(resp.text)
            round1[mid] = resp.text
            graded = await score_response(task, ans)
            score_color = "green" if graded["score"] >= 0.8 else "yellow" if graded["score"] >= 0.5 else "red"
            console.print(Panel(
                resp.text,
                title=f"[{style}]{model.name}[/]  [{score_color}]Score: {graded['score']:.2f}[/]  [dim]{resp.latency_ms:.0f}ms · {resp.tokens_total} tokens[/]",
                border_style=style.split()[-1] if " " in style else style,
                expand=False,
            ))
        else:
            err = getattr(resp, "error", str(resp))
            console.print(f"[red]✗ {model.name}: {err}[/]")
        console.print()

    if len(round1) < 2:
        console.print("[red]Not enough models succeeded for debate.[/]")
        return

    # ── Round 2 ────────────────────────────────────────────────────────────
    console.print(Rule("[bold green]🟢 Round 2 — Debate & Revision[/]", style="green"))
    console.print()
    console.print("[dim]Each model now sees all other Round 1 answers and revises...[/]\n")

    REVISION_PROMPT = """\
Task: {task}

Other agents' Round 1 answers:
{others}

Your Round 1 answer was:
{mine}

Revise your answer if others raised valid points you missed.
If you still believe you are correct, defend your answer.
End with: "Final answer: <your revised answer>"
"""

    r2_calls = {}
    for mid, my_text in round1.items():
        others_str = "\n---\n".join(
            f"[{MODELS[oid].short}]: {extract_final_answer(txt)}"
            for oid, txt in round1.items() if oid != mid
        )
        r2_prompt = REVISION_PROMPT.format(
            task=task.text,
            others=others_str,
            mine=extract_final_answer(my_text),
        )
        r2_calls[mid] = call_model(mid, r2_prompt, temperature=0.0, system_prompt=DEBATE_SYSTEM)

    r2_resps_list = await asyncio.gather(*r2_calls.values(), return_exceptions=True)
    r2_results = dict(zip(r2_calls.keys(), r2_resps_list))

    round2_answers = {}
    for mid, resp in r2_results.items():
        model = MODELS[mid]
        style = MODEL_STYLES.get(mid, "white")
        if hasattr(resp, "success") and resp.success:
            ans = extract_final_answer(resp.text)
            round2_answers[mid] = ans
            graded = await score_response(task, ans)
            score_color = "green" if graded["score"] >= 0.8 else "yellow" if graded["score"] >= 0.5 else "red"

            # Did they change their answer?
            r1_ans = extract_final_answer(round1.get(mid, ""))
            changed = "(revised ✏️)" if ans.lower()[:30] != r1_ans.lower()[:30] else "(unchanged)"

            console.print(Panel(
                resp.text,
                title=f"[{style}]{model.name}[/]  [{score_color}]Score: {graded['score']:.2f}[/]  [dim]{changed}[/]",
                border_style=style.split()[-1] if " " in style else style,
                expand=False,
            ))
        else:
            err = getattr(resp, "error", str(resp))
            console.print(f"[red]✗ {model.name} R2: {err}[/]")
        console.print()

    # ── Verdict ────────────────────────────────────────────────────────────
    console.print(Rule("[bold white]📋 Final Verdict[/]", style="white"))
    console.print()

    from src.graders import majority_vote
    final = majority_vote(list(round2_answers.values()))
    final_graded = await score_response(task, final)
    score_color = "green" if final_graded["score"] >= 0.8 else "yellow" if final_graded["score"] >= 0.5 else "red"

    console.print(f"  [white]Ground truth:[/]  [bold]{task.answer}[/]")
    console.print(f"  [white]C4 final answer:[/] [bold {score_color}]{final}[/]")
    console.print(f"  [white]C4 score:[/]       [{score_color}]{final_graded['score']:.2f}[/]")
    console.print()

    # Compare with C1 (just one model's R1 answer)
    best_r1 = max(round1, key=lambda m: len(round1[m]))  # pick longest (proxy)
    r1_ans = extract_final_answer(round1[best_r1])
    r1_graded = await score_response(task, r1_ans)
    r1_color = "green" if r1_graded["score"] >= 0.8 else "yellow" if r1_graded["score"] >= 0.5 else "red"
    console.print(
        f"  [dim]Best C1 (single model) score: [{r1_color}]{r1_graded['score']:.2f}[/]  "
        f"C4 delta: [{score_color}]{(final_graded['score'] - r1_graded['score']):+.2f}[/][/]"
    )
    console.print()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a single C4 multi-agent debate in the terminal")
    parser.add_argument("--task_id", type=int, default=None, help="Specific task ID (1-140)")
    parser.add_argument("--category", type=str, default=None, help="Pick a random task from this category")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.task_id:
        task_id = args.task_id
    elif args.category:
        pool = get_tasks_by_category(args.category)
        task_id = random.choice(pool).id if pool else 1
    else:
        task_id = random.choice(ALL_TASKS).id

    asyncio.run(watch_debate(task_id))
