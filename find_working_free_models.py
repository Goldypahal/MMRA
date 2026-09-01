"""
find_working_free_models.py — Query OpenRouter's LIVE model catalog and print
a ready-to-paste MODELS dict for src/config.py.

Why this exists: OpenRouter's free-tier (":free") model roster rotates
constantly (entire free tiers have been added/removed within the same
week). Hardcoding model IDs in a script or article goes stale fast, so
this queries the live catalog instead of guessing.

Usage:
    python find_working_free_models.py
    python find_working_free_models.py --n 4          # how many models to pick (default 4)
    python find_working_free_models.py --test          # also send a live 1-token test call to each candidate
"""

import argparse
import json
import os
import sys
import urllib.request
from dotenv import load_dotenv

load_dotenv()

MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_models() -> list[dict]:
    req = urllib.request.Request(MODELS_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data.get("data", [])


def is_free(model: dict) -> bool:
    mid = model.get("id", "")
    pricing = model.get("pricing", {})
    prompt_price = pricing.get("prompt", "1")
    completion_price = pricing.get("completion", "1")
    try:
        free_by_price = float(prompt_price) == 0.0 and float(completion_price) == 0.0
    except (TypeError, ValueError):
        free_by_price = False
    return mid.endswith(":free") or free_by_price


def test_model(model_id: str, api_key: str) -> tuple[bool, str]:
    """Send a 1-token live call to confirm the model actually responds (not just listed)."""
    import urllib.request as ur

    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 5,
    }).encode()

    req = ur.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with ur.urlopen(req, timeout=30) as resp:
            json.loads(resp.read().decode())
        return True, "OK"
    except Exception as e:
        return False, str(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="How many free models to select")
    ap.add_argument("--test", action="store_true",
                     help="Also send a live test call to each candidate (needs OPENROUTER_API_KEY)")
    ap.add_argument("--prefer", nargs="*", default=["deepseek", "qwen", "llama", "gemma", "nemotron", "glm"],
                     help="Provider name substrings to prefer, in priority order")
    args = ap.parse_args()

    print("Fetching live model catalog from OpenRouter...")
    try:
        all_models = fetch_models()
    except Exception as e:
        print(f"[ERROR] Could not reach OpenRouter API: {e}")
        sys.exit(1)

    free_models = [m for m in all_models if is_free(m)]
    print(f"Found {len(free_models)} free models out of {len(all_models)} total.\n")

    if not free_models:
        print("No free models currently listed. Check https://openrouter.ai/models?q=free manually.")
        sys.exit(1)

    # Sort candidates by preference order, then alphabetically
    def pref_rank(m):
        mid = m["id"].lower()
        for i, p in enumerate(args.prefer):
            if p in mid:
                return i
        return len(args.prefer)

    free_models.sort(key=lambda m: (pref_rank(m), m["id"]))

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    selected = []

    for m in free_models:
        if len(selected) >= args.n:
            break
        mid = m["id"]
        if args.test:
            if not api_key:
                print("[ERROR] --test requires OPENROUTER_API_KEY set in your environment.")
                sys.exit(1)
            ok, detail = test_model(mid, api_key)
            status = "OK" if ok else f"FAIL ({detail})"
            print(f"  testing {mid:<55} {status}")
            if not ok:
                continue
        selected.append(mid)

    if len(selected) < args.n:
        print(f"\n[!] Only found {len(selected)} working models out of {args.n} requested.")
        print("    Either lower --n, drop --prefer filters, or check availability at")
        print("    https://openrouter.ai/models?q=free")

    print("\n" + "=" * 70)
    print("Paste this into src/config.py, replacing the MODELS dict:")
    print("=" * 70 + "\n")

    labels = ["A", "B", "C", "D", "E", "F"]
    colors = ["bold blue", "bold green", "bold yellow", "bold red", "bold magenta", "bold cyan"]

    print("MODELS: Dict[str, ModelConfig] = {")
    for i, mid in enumerate(selected):
        label = labels[i] if i < len(labels) else f"M{i}"
        color = colors[i] if i < len(colors) else "white"
        short = mid.split("/")[-1].split(":")[0][:12]
        name = short.replace("-", " ").title().replace(" ", "-")
        print(f'    "{label}": ModelConfig("{label}", "{name}", "{short}", '
              f'"{mid}", "{color}"),')
    print("}")
    print(f'\nC1_C2_MODEL: str = "{labels[0] if selected else "A"}"')

    print("\n(Tip: re-run this script periodically — the free roster changes weekly.)")


if __name__ == "__main__":
    main()
