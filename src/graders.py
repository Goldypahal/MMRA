"""
graders.py — Automated scoring for task responses.
Supports: exact match, numeric match, contains-check, LLM-as-judge.
"""

import re
import asyncio
from typing import Optional
from src.tasks import Task
from src.client import call_model, APIResponse


# ─────────────────────────────────────────────────────────────────────────────
# Core grading functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip trailing punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.\+\-\/\*\^=]", "", text)
    text = text.strip().rstrip(".,;:")
    text = re.sub(r"\s+", " ", text)
    return text


def grade_exact(response: str, answer: str) -> float:
    """1.0 if exact match after normalization, else 0.0."""
    return 1.0 if normalize(response) == normalize(answer) else 0.0


def grade_numeric(response: str, answer: str, tolerance: float = 0.01) -> float:
    """
    Extract first number from response and answer. Score 1.0 if within tolerance.
    """
    def extract_number(text: str) -> Optional[float]:
        # Handle scientific notation like 6.022e23
        matches = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text.replace(",", ""))
        return float(matches[0]) if matches else None

    r_val = extract_number(response)
    a_val = extract_number(answer)

    if r_val is None or a_val is None:
        return 0.0
    if a_val == 0:
        return 1.0 if abs(r_val) < tolerance else 0.0
    return 1.0 if abs(r_val - a_val) / abs(a_val) <= tolerance else 0.0


def grade_contains(response: str, answer: str) -> float:
    """
    Score 1.0 if any synonym group or key tokens appear in response.
    Supports '|' separated alternative acceptable phrases to prevent false negatives.
    """
    resp_norm = normalize(response)
    if "|" in answer:
        alternatives = [normalize(a) for a in answer.split("|") if a.strip()]
        for alt in alternatives:
            if alt in resp_norm:
                return 1.0
        all_tokens = set()
        for alt in alternatives:
            all_tokens.update(alt.split())
        hits = sum(1 for t in all_tokens if t in resp_norm)
        return hits / len(all_tokens) if all_tokens else 0.0

    tokens = normalize(answer).split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in resp_norm)
    return hits / len(tokens)


JUDGE_PROMPT = """You are an impartial, blind rubric-based grader for a research experiment.

Task given to the AI agent:
{task}

The AI agent's response:
{response}

Ground-truth expected answer / key concepts:
{answer}

Evaluate against this 5-point analytic rubric:
1. Core Claim / Final Answer Accuracy (0-2 points)
2. Logical Soundness & Non-contradiction (0-1 point)
3. Epistemic Honesty / Hallucination Avoidance (0-1 point)
4. Key Mechanism / Concept Coverage (0-1 point)

Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <integer 0-5>, "reasoning": "<one sentence summary>"}}"""


async def grade_llm(
    task: Task,
    response: str,
    judge_model: str = "B",   # Gemma as default judge
) -> tuple[float, str]:
    """
    Use an LLM as judge with analytic rubric for open-ended tasks.
    Returns (score_0_to_1, reasoning_string).
    Runs 3 independent judge calls and averages for reliability.
    """
    import json

    prompt = JUDGE_PROMPT.format(
        task=task.text,
        response=response,
        answer=task.answer,
    )

    judge_tasks = [call_model(judge_model, prompt, temperature=0.0) for _ in range(3)]
    results: list[APIResponse] = await asyncio.gather(*judge_tasks)

    scores = []
    reasoning_samples = []

    for r in results:
        if not r.success:
            continue
        try:
            text = r.text.strip()
            if "```" in text:
                text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
            data = json.loads(text)
            raw_score = float(data.get("score", 0))
            # 5-point rubric -> normalize 0-1
            scores.append(min(5.0, max(0.0, raw_score)) / 5.0)
            if "reasoning" in data:
                reasoning_samples.append(data["reasoning"])
        except Exception:
            pass

    if not scores:
        return 0.0, "Judge failed to parse"

    avg = sum(scores) / len(scores)
    reasoning = reasoning_samples[0] if reasoning_samples else "No reasoning"
    return round(avg, 4), reasoning


def classify_failure_mode(response: str, expected_answer: str, category: str, score: float) -> str:
    """
    Classify failure mode according to Paper 1 Section IV-D taxonomy:
    - "None" (if correct, score >= 0.8)
    - "Incomplete" (empty/truncated output)
    - "Reasoning error" (logical, step-by-step, or algorithmic error)
    - "Factual error" (incorrect domain fact)
    - "Hallucination" (ungrounded assertion)
    """
    if score >= 0.8:
        return "None"
    if not response or len(response.strip()) < 10:
        return "Incomplete"
    
    cat = (category or "").lower()
    resp_lower = response.lower()
    
    if "hallucin" in resp_lower or "unknown entity" in resp_lower:
        return "Hallucination"
    elif cat in ["math", "logic", "coding", "planning"]:
        return "Reasoning error"
    elif cat in ["science", "knowledge"]:
        return "Factual error"
    else:
        return "Reasoning error"


async def score_response(task: Task, response: str) -> dict:
    """
    Dispatch to correct grader based on task.grader_hint.
    Returns: { "score": float 0-1, "method": str, "detail": str, "failure_mode": str }
    """
    if not response or not response.strip():
        return {
            "score": 0.0,
            "method": task.grader_hint,
            "detail": "empty response",
            "failure_mode": "Incomplete",
        }

    method = task.grader_hint

    if method == "exact":
        s = grade_exact(response, task.answer)
        detail = f"'{normalize(response[:60])}' vs '{normalize(task.answer)}'"

    elif method == "numeric":
        s = grade_numeric(response, task.answer)
        detail = f"extracted value vs {task.answer}"

    elif method == "contains":
        s = grade_contains(response, task.answer)
        detail = f"{s*100:.0f}% key tokens found"

    elif method == "llm":
        s, reasoning = await grade_llm(task, response)
        detail = reasoning

    else:
        s = grade_contains(response, task.answer)
        detail = "fallback grader"

    fm = classify_failure_mode(response, task.answer, task.category, s)
    return {
        "score": s,
        "method": method if method != "llm" else "llm_judge",
        "detail": detail,
        "failure_mode": fm,
    }


from collections import Counter


def majority_vote(answers: list[str]) -> str:
    """
    Return the most common answer string after normalization.
    Ties broken by occurrence order.
    """
    if not answers:
        return ""

    norm_map = {}
    norm_list = []
    for a in answers:
        n = normalize(a)
        if n not in norm_map:
            norm_map[n] = a
        norm_list.append(n)

    counts = Counter(norm_list)
    winner_norm = counts.most_common(1)[0][0]
    return norm_map[winner_norm]


def extract_final_answer(text: str) -> str:
    """
    Extract concise final answer from model response.
    Checks explicit markers in reverse order (to find the final answer block, not prompt headers),
    LaTeX boxed syntax, equation results, and falls back to last non-empty line.
    """
    if not text or not text.strip():
        return ""

    text_clean = text.strip()

    # Check LaTeX boxed format \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text_clean)
    if boxed:
        return boxed[-1].strip()

    # Explicit final answer markers in order of specificity
    markers = [
        "REVISED FINAL ANSWER:",
        "FINAL ANSWER:",
        "Final Answer:",
        "Final answer:",
        "ANSWER:",
        "Answer:",
        "THEREFORE:",
        "Therefore:",
        "CONCLUSION:",
        "Conclusion:",
    ]
    for marker in markers:
        idx = text_clean.lower().rfind(marker.lower())
        if idx != -1:
            after = text_clean[idx + len(marker):].strip()
            line = after.split("\n")[0].strip()
            cleaned = re.sub(r"^\*+|\*+$", "", line).strip()
            if cleaned:
                return cleaned

    # Regex patterns matching last occurrences
    patterns = [
        r"(?:final answer|revised final answer|answer)[:\s]+([^\n]+)",
        r"=\s*([^\n]+)$",
    ]
    for pat in patterns:
        matches = re.findall(pat, text_clean, re.IGNORECASE)
        if matches:
            last_match = matches[-1].strip()
            last_match = re.sub(r"^\*+|\*+$", "", last_match).strip()
            if last_match:
                return last_match

    # Fallback to last non-empty line
    lines = [l.strip() for l in text_clean.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r"^\*+|\*+$", "", last).strip()
        return last

    return text_clean
