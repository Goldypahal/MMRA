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

    console.print(f"\n[cyan]Testing {len(MODELS)} models via direct native provider APIs...[/]\n")

    tasks = [test_model(mid) for mid in MODELS]
    results = await asyncio.gather(*tasks)

    table = Table(
        title="[bold white]Model Connectivity Test[/]",
        box=box.SIMPLE_HEAD,
        border_style="dim",
    )
    table.add_column("Model",          style="white",  min_width=18)
    table.add_column("Provider",       style="cyan",   min_width=12)
    table.add_column("Status",         justify="center", min_width=10)
    table.add_column("Latency",        justify="right",  min_width=10)
    table.add_column("Tokens",         justify="right",  min_width=8)
    table.add_column("JSON Valid",     justify="center", min_width=10)
    table.add_column("Model Identifier", style="dim",    min_width=25)

    all_ok = True
    for r in results:
        mid = r["model_id"]
        m_cfg = MODELS[mid]
        style = MODEL_STYLES.get(mid, "white")
        status = "[green]OK[/]" if r["success"] else "[red]FAIL[/]"
        json_ok = "[green]YES[/]" if r["valid_json"] else "[red]NO[/]" if r["success"] else "[dim]-[/]"
        if not r["success"]:
            all_ok = False
        table.add_row(
            f"[{style}]{r['model_name']}[/]",
            m_cfg.provider.upper(),
            status,
            f"{r['latency_ms']:.0f}ms",
            str(r["tokens"]),
            json_ok,
            m_cfg.api_key,
        )
        if not r["success"] and r["error"]:
            short_err = str(r["error"])[:90]
            table.add_row(f"  [dim red]{short_err}[/]", "", "", "", "", "", "")

    console.print(table)

    if all_ok:
        console.print(Panel(
            "[green]All models reachable via direct provider APIs.[/] Ready to run the experiment.\n\n"
            "Quick test (10 tasks):  [cyan]python run_experiment.py --n 10[/]\n"
            "Full run (140 tasks):   [cyan]python run_experiment.py[/]\n"
            "Watch a debate:         [cyan]python scripts/demo_debate.py[/]",
            title="[green]Ready[/]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[yellow]Some models failed or hit provider limits.[/]\n"
            "Check provider API keys or endpoint rate limits in your .env file.\n"
            "Offline mock mode is available by setting MMRA_MOCK_MODE=1.",
            title="[yellow]Connectivity Warning[/]",
            border_style="yellow",
        ))


if __name__ == "__main__":
    asyncio.run(main())
