"""
run_extended_accuracy_check.py — Score any LLM against the 70-task
adversarial extended set (tasks_extended.py / tasks_extended.json).

Two modes:

1) INSIDE MMRA (recommended): reuses MMRA's own src.client / src.graders,
   so results are directly comparable with the original 140-task experiment
   and land in the same SQLite DB.

       python run_extended_accuracy_check.py --model A
       python run_extended_accuracy_check.py --model A --category math
       python run_extended_accuracy_check.py --all-models

2) STANDALONE: if src/ isn't importable (e.g. you just want to grade
   tasks_extended.json against another API), this script falls back to a
   minimal local grader (exact / numeric / contains logic).
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Configure Windows console encoding for UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Try to hook into MMRA framework; fall back to standalone mode.
# ─────────────────────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, str(HERE))
    from src.client import call_model, MODELS
    from src.graders import score_response
    from src.tasks_extended import EXTENDED_TASKS
    from src.database import save_result
    from src.conditions import TaskResult
    MMRA_MODE = True
except Exception as e:
    MMRA_MODE = False
    with open(HERE / "tasks_extended.json", "r", encoding="utf-8") as f:
        EXTENDED_TASKS = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone fallback grader (subset of src/graders.py logic)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.\+\-\/\*\^=]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _standalone_grade(task: dict, response: str) -> dict:
    method = task["grader_hint"]
    answer = task["answer"]

    if not response.strip():
        return {"score": 0.0, "method": method, "detail": "empty response"}

    if method == "exact":
        s = 1.0 if _normalize(response) == _normalize(answer) else 0.0
    elif method == "numeric":
        nums_r = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", response.replace(",", ""))
        nums_a = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", answer.replace(",", ""))
        if not nums_r or not nums_a:
            s = 0.0
        else:
            r_val, a_val = float(nums_r[0]), float(nums_a[0])
            s = 1.0 if (a_val == 0 and abs(r_val) < 0.01) or \
                       (a_val != 0 and abs(r_val - a_val) / abs(a_val) <= 0.01) else 0.0
    elif method == "contains":
        resp_norm = _normalize(response)
        tokens = _normalize(answer).split()
        s = (sum(1 for t in tokens if t in resp_norm) / len(tokens)) if tokens else 0.0
    else:
        s = None

    return {"score": s, "method": method, "detail": "standalone grader"}


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring loop
# ─────────────────────────────────────────────────────────────────────────────
async def run(model_id: str, category: str | None, difficulty: str | None):
    tasks = EXTENDED_TASKS
    if category:
        tasks = [t for t in tasks if (t.category if MMRA_MODE else t["category"]) == category]
    if difficulty:
        tasks = [t for t in tasks if (t.difficulty if MMRA_MODE else t["difficulty"]) == difficulty]

    print(f"\nRunning {len(tasks)} adversarial tasks against model '{model_id}' "
          f"({'MMRA pipeline' if MMRA_MODE else 'standalone mode'})\n")

    rows = []
    correct = 0

    for t in tasks:
        task_id = t.id if MMRA_MODE else t["id"]
        text = t.text if MMRA_MODE else t["text"]
        category_ = t.category if MMRA_MODE else t["category"]
        difficulty_ = t.difficulty if MMRA_MODE else t["difficulty"]

        if MMRA_MODE:
            resp = await call_model(model_id, text, temperature=0.0)
            response_text = resp.text if resp.success else ""
            grade = await score_response(t, response_text)
            score = grade["score"]
            tokens = resp.tokens_total
            latency = resp.latency_ms
            
            # Save to MMRA database
            tr = TaskResult(
                task_id=task_id,
                category=category_,
                difficulty=difficulty_,
                condition="C1_EXTENDED",
                final_answer=grade.get("extracted", response_text[:50]),
                score=score if score is not None else 0.0,
                grader_method=grade.get("method", "unknown"),
                grader_detail=grade.get("detail", ""),
                failure_mode=grade.get("failure_mode", "None"),
                tokens_total=tokens,
                latency_ms=latency,
            )
            save_result(tr)
        else:
            raise RuntimeError(
                "Standalone mode requires wiring response_text from your custom API client."
            )

        if score is not None and score >= 0.8:
            correct += 1

        rows.append({
            "id": task_id,
            "category": category_,
            "difficulty": difficulty_,
            "score": score,
            "detail": grade.get("detail", ""),
            "failure_mode": grade.get("failure_mode", ""),
        })
        status = "PASS" if (score or 0) >= 0.8 else "FAIL"
        print(f"  [{status}] #{task_id:<4} {category_:<10} {difficulty_:<7} score={score}")

    n = len(tasks)
    print(f"\nExtended Set Accuracy ({model_id}): {correct}/{n} = {correct/n*100:.1f}%\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Extended Adversarial Benchmark Runner")
    ap.add_argument("--model", default="A", help="Model ID from src.config.MODELS (A/B/C/D)")
    ap.add_argument("--all-models", action="store_true", help="Run against all 4 configured models")
    ap.add_argument("--category", default=None,
                     help="math | logic | coding | science | language | knowledge | openended")
    ap.add_argument("--difficulty", default=None, help="Medium | Hard")
    ap.add_argument("--export", default=None, help="CSV path to export results")
    args = ap.parse_args()

    if not MMRA_MODE:
        print("[!] src/ not found — running in standalone mode.\n")

    model_ids = list(MODELS.keys()) if (args.all_models and MMRA_MODE) else [args.model]

    all_rows = []
    for mid in model_ids:
        rows = asyncio.run(run(mid, args.category, args.difficulty))
        for r in rows:
            r["model"] = mid
        all_rows.extend(rows)

    if args.export:
        import csv
        with open(args.export, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Exported {len(all_rows)} rows to {args.export}")


if __name__ == "__main__":
    main()
