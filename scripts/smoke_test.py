"""
scripts/smoke_test.py — Validate API keys and model connectivity.

Usage:
    python scripts/smoke_test.py
"""

import asyncio
import sys
import os

# Set UTF-8 before any imports that open stdout
os.environ["PYTHONUTF8"] = "1"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config import MODELS, OPENROUTER_API_KEY
from src.client import call_model
from src.display import print_banner, MODEL_STYLES

console = Console()

TEST_PROMPT = (
    "Reply with a valid JSON object containing exactly two fields:\n"
    '{"status": "ok", "model": "<your model name>"}\n'
    "Output only the JSON, nothing else."
)


async def test_model(mid: str) -> dict:
    import time, json
    t0 = time.perf_counter()
    resp = await call_model(mid, TEST_PROMPT, temperature=0.0)
    elapsed = (time.perf_counter() - t0) * 1000

    result = {
        "model_id": mid,
        "model_name": MODELS[mid].name,
        "success": resp.success,
        "latency_ms": elapsed,
        "tokens": resp.tokens_total,
        "error": resp.error,
        "valid_json": False,
    }

    if resp.success:
        try:
            text = resp.text.strip()
            if "```" in text:
                import re
                text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
            parsed = json.loads(text)
            result["valid_json"] = parsed.get("status") == "ok"
        except Exception:
            result["valid_json"] = False

    return result


async def main():
    print_banner()

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_key_here":
        console.print(Panel(
            "[red]OPENROUTER_API_KEY not set.[/]\n\n"
            "1. Copy [cyan].env.example[/] to [cyan].env[/]\n"
            "2. Get free key: https://openrouter.ai\n"
            "3. Paste your key in [cyan].env[/]",
            title="[red]Missing API Key[/]",
            border_style="red",
        ))
        sys.exit(1)

    console.print(f"\n[cyan]Testing {len(MODELS)} models via OpenRouter...[/]\n")

    tasks = [test_model(mid) for mid in MODELS]
    results = await asyncio.gather(*tasks)

    table = Table(
        title="[bold white]Model Connectivity Test[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
    )
    table.add_column("Model",       style="white",  min_width=20)
    table.add_column("Status",      justify="center", min_width=10)
    table.add_column("Latency",     justify="right",  min_width=10)
    table.add_column("Tokens",      justify="right",  min_width=8)
    table.add_column("JSON Valid",  justify="center", min_width=10)
    table.add_column("API Model ID", style="dim",    min_width=38)

    all_ok = True
    for r in results:
        mid = r["model_id"]
        style = MODEL_STYLES.get(mid, "white")
        status = "[green]OK[/]" if r["success"] else "[red]FAIL[/]"
        json_ok = "[green]YES[/]" if r["valid_json"] else "[red]NO[/]" if r["success"] else "[dim]-[/]"
        if not r["success"]:
            all_ok = False
        table.add_row(
            f"[{style}]{r['model_name']}[/]",
            status,
            f"{r['latency_ms']:.0f}ms",
            str(r["tokens"]),
            json_ok,
            MODELS[mid].api_key,
        )
        if not r["success"] and r["error"]:
            short_err = str(r["error"])[:90]
            table.add_row(f"  [dim red]{short_err}[/]", "", "", "", "", "")

    console.print(table)

    if all_ok:
        console.print(Panel(
            "[green]All models reachable.[/] Ready to run the experiment.\n\n"
            "Quick test (10 tasks):  [cyan]python run_experiment.py --n 10[/]\n"
            "Full run (140 tasks):   [cyan]python run_experiment.py[/]\n"
            "Watch a debate:         [cyan]python scripts/demo_debate.py[/]",
            title="[green]Ready[/]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[yellow]Some models failed.[/]\n"
            "Free-tier models rotate on OpenRouter.\n"
            "Check https://openrouter.ai/models?q=free for current availability.\n"
            "Update src/config.py MODELS dict with working model IDs.",
            title="[yellow]Partial Failure[/]",
            border_style="yellow",
        ))


if __name__ == "__main__":
    asyncio.run(main())
