#!/usr/bin/env python3
"""
check_listed.py — Utility to check if specified model IDs (or config models)
are currently listed and available on OpenRouter's API.

Usage:
    python check_listed.py
    python check_listed.py nvidia/nemotron-3-ultra-550b-a55b:free google/gemma-4-31b-it:free z-ai/glm-4.5-air:free openai/gpt-oss-20b:free
"""

import sys
import json
import io
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

# Force UTF-8 output encoding for Windows terminal compatibility
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Try importing config models if available
try:
    from src.config import MODELS, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
except ImportError:
    MODELS = {}
    OPENROUTER_API_KEY = ""
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def fetch_openrouter_models() -> List[Dict[str, Any]]:
    """Fetch live list of models from OpenRouter API."""
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/models"
    headers = {
        "User-Agent": "Multimodel-Checker/1.0",
        "Accept": "application/json",
    }
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY not in ("your_openrouter_key_here", "mock"):
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
    except urllib.error.URLError as e:
        print(f"[Warning] Could not fetch live models from OpenRouter: {e}")
        return []
    except Exception as e:
        print(f"[Warning] Unexpected error fetching models: {e}")
        return []


def check_models(target_model_ids: List[str]) -> None:
    """Check target model IDs against live OpenRouter models list."""
    print("\nFetching live model directory from OpenRouter...")
    live_models = fetch_openrouter_models()
    
    # Map model ID -> full model object
    live_dict: Dict[str, Dict[str, Any]] = {m["id"]: m for m in live_models if "id" in m}
    live_ids_lower = {m_id.lower(): m_id for m_id in live_dict.keys()}

    if HAS_RICH:
        console = Console(highlight=False)
        table = Table(title="OpenRouter Model Listing Status", show_header=True, header_style="bold magenta")
        table.add_column("Input Model ID", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Official Name", style="white")
        table.add_column("Context Length", justify="right")
        table.add_column("Pricing (Prompt / Comp)", justify="right")
        table.add_column("Close Matches / Alternatives", style="yellow")
    else:
        table = None

    results = []

    for model_id in target_model_ids:
        # Check exact or case-insensitive match
        matched_id = None
        if model_id in live_dict:
            matched_id = model_id
        elif model_id.lower() in live_ids_lower:
            matched_id = live_ids_lower[model_id.lower()]

        if matched_id:
            m_info = live_dict[matched_id]
            status = "LISTED"
            name = m_info.get("name", matched_id)
            context = str(m_info.get("context_length", "N/A"))
            
            pricing = m_info.get("pricing", {})
            prompt_price = pricing.get("prompt", "0")
            comp_price = pricing.get("completion", "0")
            if prompt_price == "0" and comp_price == "0":
                price_str = "Free"
            else:
                price_str = f"${float(prompt_price)*1e6:.2f}/M | ${float(comp_price)*1e6:.2f}/M"
            
            alt_str = "-"
            is_listed = True
        else:
            status = "NOT LISTED"
            name = "Unknown Model ID"
            context = "-"
            price_str = "-"
            is_listed = False

            # Find fuzzy / close matches
            parts = model_id.lower().replace(":", "/").split("/")
            keywords = [p for p in parts if p not in ("free", "it", "v1")]
            
            close_matches = []
            for live_id in live_dict.keys():
                live_id_lower = live_id.lower()
                if any(kw in live_id_lower for kw in keywords if len(kw) > 2):
                    close_matches.append(live_id)

            if close_matches:
                # Prefer free matches if looking for free model
                if "free" in model_id.lower():
                    free_matches = [m for m in close_matches if ":free" in m.lower() or "free" in m.lower()]
                    if free_matches:
                        close_matches = free_matches
                alt_str = ", ".join(close_matches[:3])
            else:
                alt_str = "No close matches found"

        results.append({
            "model_id": model_id,
            "is_listed": is_listed,
            "status": status,
            "name": name,
            "context": context,
            "pricing": price_str,
            "alt": alt_str
        })

        if HAS_RICH and table:
            status_cell = "[bold green][LISTED][/bold green]" if is_listed else "[bold red][NOT LISTED][/bold red]"
            table.add_row(model_id, status_cell, name, context, price_str, alt_str)

    if HAS_RICH and table:
        console.print(table)
    else:
        print(f"\n{'MODEL ID':<45} | {'STATUS':<12} | {'OFFICIAL NAME':<30} | {'PRICING':<15}")
        print("-" * 110)
        for r in results:
            print(f"{r['model_id']:<45} | {r['status']:<12} | {r['name']:<30} | {r['pricing']:<15}")
            if not r['is_listed']:
                print(f"   └─ Suggestion: {r['alt']}")

    listed_count = sum(1 for r in results if r["is_listed"])
    total_count = len(results)
    
    summary_msg = f"Summary: {listed_count}/{total_count} specified models are currently listed on OpenRouter."
    if HAS_RICH:
        style = "bold green" if listed_count == total_count else "bold yellow"
        console.print(Panel(summary_msg, style=style))
    else:
        print(f"\n{summary_msg}")


def main():
    if len(sys.argv) > 1:
        target_models = sys.argv[1:]
    else:
        # Default to configured models in src/config.py
        target_models = [cfg.api_key for cfg in MODELS.values()]
        # Also include fallback keys if any
        for cfg in MODELS.values():
            for fallback in cfg.fallback_api_keys:
                if fallback not in target_models:
                    target_models.append(fallback)
        
        print("No model IDs specified on command line. Using models from src/config.py:")

    check_models(target_models)


if __name__ == "__main__":
    main()
