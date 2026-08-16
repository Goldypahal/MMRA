"""
conditions.py — Implements all 4 experimental conditions.

C1: Single model, one call.
C2: Single model, 3 calls, majority vote (self-consistency).
C3: All 4 models in parallel, majority vote (no communication).
C4: All 4 models answer, then each revises after seeing others' answers (debate).
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import MODELS, CONDITIONS, TEMPERATURE, C2_SAMPLES
from src.client import call_model, call_all_models, APIResponse
from src.tasks import Task
from src.graders import score_response, majority_vote, extract_final_answer


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: int
    category: str
    difficulty: str
    condition: str
    final_answer: str
    score: float
    grader_method: str
    grader_detail: str
    tokens_total: int
    latency_ms: float
    failure_mode: str = "None"
    # Per-model details (for C3, C4)
    model_responses: dict = field(default_factory=dict)   # model_id -> text
    model_scores: dict = field(default_factory=dict)      # model_id -> score
    model_tokens: dict = field(default_factory=dict)      # model_id -> tokens
    # C4-specific
    round1_responses: dict = field(default_factory=dict)
    round2_responses: dict = field(default_factory=dict)
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise and concise AI assistant. "
    "Answer the question directly and clearly. "
    "State your final answer explicitly at the end."
)

DEBATE_REVISION_PROMPT = """\
Task: {task}

Other agents' initial answers:
{other_answers}

Your initial answer was:
{your_answer}

Having reviewed all responses, perform the following 3-part evaluation and state your REVISED final answer:
1. Identify any information or reasoning you missed in your initial response.
2. Identify specific errors or flaws in the other agents' reasoning.
3. Identify any errors in your own reasoning revealed by others.

End your response with: "Final answer: <your answer>"
"""


def build_task_prompt(task: Task) -> str:
    return f"Question: {task.text}\n\nProvide your answer clearly. End with 'Final answer: <answer>'."


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Single Model
# ─────────────────────────────────────────────────────────────────────────────

async def run_c1(task: Task, model_id: str = "A") -> TaskResult:
    """One model, one call at temperature=0.0."""
    t0 = time.perf_counter()
    resp = await call_model(model_id, build_task_prompt(task),
                            temperature=0.0, system_prompt=SYSTEM_PROMPT)
    elapsed = (time.perf_counter() - t0) * 1000

    if not resp.success:
        return TaskResult(
            task_id=task.id, category=task.category, difficulty=task.difficulty,
            condition="C1", final_answer="", score=0.0,
            grader_method="error", grader_detail=resp.error or "api_error",
            tokens_total=0, latency_ms=elapsed, error=resp.error,
        )

    final = extract_final_answer(resp.text)
    graded = await score_response(task, final)

    return TaskResult(
        task_id=task.id, category=task.category, difficulty=task.difficulty,
        condition="C1",
        final_answer=final,
        score=graded["score"],
        grader_method=graded["method"],
        grader_detail=graded["detail"],
        failure_mode=graded.get("failure_mode", "None"),
        tokens_total=resp.tokens_total,
        latency_ms=round(elapsed, 1),
        model_responses={model_id: resp.text},
        model_tokens={model_id: resp.tokens_total},
    )


# ─────────────────────────────────────────────────────────────────────────────
# C2 — Self-Consistency (same model, 3 calls, majority vote)
# ─────────────────────────────────────────────────────────────────────────────

async def run_c2(task: Task, model_id: str = "A") -> TaskResult:
    """Same model, C2_SAMPLES independent calls at temperature=0.7, majority vote."""
    t0 = time.perf_counter()
    calls = [
        call_model(model_id, build_task_prompt(task),
                   temperature=0.7, system_prompt=SYSTEM_PROMPT)
        for _ in range(C2_SAMPLES)
    ]
    responses: list[APIResponse] = await asyncio.gather(*calls)
    elapsed = (time.perf_counter() - t0) * 1000

    successes = [r for r in responses if r.success]
    if not successes:
        return TaskResult(
            task_id=task.id, category=task.category, difficulty=task.difficulty,
            condition="C2", final_answer="", score=0.0,
            grader_method="error", grader_detail="all calls failed",
            tokens_total=0, latency_ms=elapsed,
        )

    answers = [extract_final_answer(r.text) for r in successes]
    final = majority_vote(answers)
    graded = await score_response(task, final)
    tokens = sum(r.tokens_total for r in responses)

    return TaskResult(
        task_id=task.id, category=task.category, difficulty=task.difficulty,
        condition="C2",
        final_answer=final,
        score=graded["score"],
        grader_method=graded["method"],
        grader_detail=graded["detail"],
        failure_mode=graded.get("failure_mode", "None"),
        tokens_total=tokens,
        latency_ms=round(elapsed, 1),
        model_responses={f"{model_id}_call{i}": r.text for i, r in enumerate(successes)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# C3 — Parallel Vote (4 models, no communication)
# ─────────────────────────────────────────────────────────────────────────────

async def run_c3(task: Task) -> TaskResult:
    """All 4 models answer independently in parallel; majority vote."""
    t0 = time.perf_counter()
    all_resps = await call_all_models(
        build_task_prompt(task), temperature=0.0, system_prompt=SYSTEM_PROMPT
    )
    elapsed = (time.perf_counter() - t0) * 1000

    answers, model_responses, model_tokens = {}, {}, {}
    for mid, resp in all_resps.items():
        if resp.success:
            ans = extract_final_answer(resp.text)
            answers[mid] = ans
            model_responses[mid] = resp.text
            model_tokens[mid] = resp.tokens_total

    if not answers:
        return TaskResult(
            task_id=task.id, category=task.category, difficulty=task.difficulty,
            condition="C3", final_answer="", score=0.0,
            grader_method="error", grader_detail="all models failed",
            tokens_total=0, latency_ms=elapsed,
        )

    final = majority_vote(list(answers.values()))
    graded = await score_response(task, final)

    # Per-model scores
    model_scores = {}
    for mid, ans in answers.items():
        g = await score_response(task, ans)
        model_scores[mid] = g["score"]

    return TaskResult(
        task_id=task.id, category=task.category, difficulty=task.difficulty,
        condition="C3",
        final_answer=final,
        score=graded["score"],
        grader_method=graded["method"],
        grader_detail=graded["detail"],
        failure_mode=graded.get("failure_mode", "None"),
        tokens_total=sum(model_tokens.values()),
        latency_ms=round(elapsed, 1),
        model_responses=model_responses,
        model_scores=model_scores,
        model_tokens=model_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# C4 — Multi-Agent Debate
# ─────────────────────────────────────────────────────────────────────────────

async def run_c4(task: Task) -> TaskResult:
    """
    Round 1: All 4 models answer independently.
    Round 2: Each model sees all others' Round 1 answers and revises.
    Final answer: majority vote on Round 2 answers.
    """
    t0 = time.perf_counter()

    # ── Round 1 ─────────────────────────────────────────────────────────────
    round1_resps = await call_all_models(
        build_task_prompt(task), temperature=0.0, system_prompt=SYSTEM_PROMPT
    )
    round1_answers = {}
    round1_texts = {}
    for mid, resp in round1_resps.items():
        if resp.success:
            ans = extract_final_answer(resp.text)
            round1_answers[mid] = ans
            round1_texts[mid] = resp.text

    # ── Round 2: Each model revises having seen all others ───────────────────
    revision_calls = {}
    for mid in MODELS:
        if mid not in round1_answers:
            continue
        other_answers = "\n---\n".join(
            f"Agent {oid} ({MODELS[oid].short}): {ans}"
            for oid, ans in round1_answers.items()
            if oid != mid
        )
        debate_prompt = DEBATE_REVISION_PROMPT.format(
            task=task.text,
            other_answers=other_answers,
            your_answer=round1_answers[mid],
        )
        revision_calls[mid] = call_model(mid, debate_prompt,
                                         temperature=0.0, system_prompt=SYSTEM_PROMPT)

    round2_resps_list = await asyncio.gather(*revision_calls.values(), return_exceptions=True)
    round2_resps = dict(zip(revision_calls.keys(), round2_resps_list))

    round2_answers = {}
    round2_texts = {}
    model_tokens = {}

    for mid, resp in round2_resps.items():
        if isinstance(resp, APIResponse) and resp.success:
            ans = extract_final_answer(resp.text)
            round2_answers[mid] = ans
            round2_texts[mid] = resp.text

    # Tokens = Round1 + Round2
    for mid in MODELS:
        t1 = round1_resps.get(mid)
        t2 = round2_resps.get(mid)
        tok1 = t1.tokens_total if isinstance(t1, APIResponse) and t1.success else 0
        tok2 = t2.tokens_total if isinstance(t2, APIResponse) and t2.success else 0
        model_tokens[mid] = tok1 + tok2

    elapsed = (time.perf_counter() - t0) * 1000

    if not round2_answers:
        # Fall back to round 1 if round 2 completely failed
        round2_answers = round1_answers

    final = majority_vote(list(round2_answers.values()))
    graded = await score_response(task, final)

    # Per-model scores (on round 2 answers)
    model_scores = {}
    for mid, ans in round2_answers.items():
        g = await score_response(task, ans)
        model_scores[mid] = g["score"]

    return TaskResult(
        task_id=task.id, category=task.category, difficulty=task.difficulty,
        condition="C4",
        final_answer=final,
        score=graded["score"],
        grader_method=graded["method"],
        grader_detail=graded["detail"],
        failure_mode=graded.get("failure_mode", "None"),
        tokens_total=sum(model_tokens.values()),
        latency_ms=round(elapsed, 1),
        model_responses=round2_texts,
        model_scores=model_scores,
        model_tokens=model_tokens,
        round1_responses=round1_texts,
        round2_responses=round2_texts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

async def run_condition(
    task: Task,
    condition_id: str,
    model_id: str = "A",
) -> TaskResult:
    """Route to the correct condition runner."""
    if condition_id == "C1":
        return await run_c1(task, model_id)
    elif condition_id == "C2":
        return await run_c2(task, model_id)
    elif condition_id == "C3":
        return await run_c3(task)
    elif condition_id == "C4":
        return await run_c4(task)
    else:
        raise ValueError(f"Unknown condition: {condition_id}")
