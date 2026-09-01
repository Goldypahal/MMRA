"""
analysis.py — Statistical analysis engine.
Paired t-tests, Cohen's d, Bonferroni correction,
token efficiency, model contribution, failure mode tagging.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
from dataclasses import dataclass

from src.config import CATEGORIES, CONDITIONS, MODELS
from src.tasks import ALL_TASKS


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PairedTTest:
    category: str
    c1_mean: float
    c4_mean: float
    improvement: float
    t_stat: float
    p_value: float
    p_bonferroni: float
    cohens_d: float
    significant: bool
    n: int


@dataclass
class ConditionPairComparison:
    comparison: str
    cond_a: str
    cond_b: str
    description: str
    mean_a: float
    mean_b: float
    gain: float
    t_stat: float
    p_value: float
    p_bonferroni: float
    cohens_d: float
    significant: bool
    n: int


@dataclass
class TokenEfficiency:
    condition: str
    avg_accuracy: float
    avg_tokens: float
    accuracy_per_1k_tokens: float


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy table
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with rows=categories, cols=conditions, values=mean accuracy.
    """
    if df.empty:
        return pd.DataFrame()

    table = df.groupby(["category", "condition"])["score"].mean().unstack("condition")
    # Ensure all conditions are present
    for cond in CONDITIONS:
        if cond not in table.columns:
            table[cond] = np.nan
    table = table[list(CONDITIONS.keys())]

    # Add row for grand mean
    table.loc["AVERAGE"] = table.mean()
    return table.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Paired t-tests with Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttests(df: pd.DataFrame, cond_a: str = "C1", cond_b: str = "C4") -> list[PairedTTest]:
    """
    For each category: paired t-test comparing two conditions on same task_ids.
    Bonferroni correction for number of categories tested.
    """
    results = []
    n_tests = len(CATEGORIES)

    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        a = sub[sub["condition"] == cond_a].set_index("task_id")["score"]
        b = sub[sub["condition"] == cond_b].set_index("task_id")["score"]

        # Align on shared task IDs
        common = a.index.intersection(b.index)
        if len(common) < 3:
            continue

        a_vals = a.loc[common].values
        b_vals = b.loc[common].values
        diff = b_vals - a_vals

        t_stat, p_value = stats.ttest_rel(b_vals, a_vals)
        p_bonf = min(p_value * n_tests, 1.0)

        # Cohen's d for paired samples
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0

        results.append(PairedTTest(
            category=cat,
            c1_mean=float(np.mean(a_vals)),
            c4_mean=float(np.mean(b_vals)),
            improvement=float(np.mean(b_vals) - np.mean(a_vals)),
            t_stat=float(t_stat),
            p_value=float(p_value),
            p_bonferroni=float(p_bonf),
            cohens_d=float(cohens_d),
            significant=p_bonf < 0.05,
            n=len(common),
        ))

    return sorted(results, key=lambda x: x.improvement, reverse=True)


def all_condition_pair_tests(df: pd.DataFrame) -> list[ConditionPairComparison]:
    """
    Computes paired t-tests across all 5 key condition pair comparisons:
    1. C1 vs C2: Single model vs Self-consistency (does voting within 1 model help?)
    2. C1 vs C3: Single model vs Parallel vote (does cross-model diversity help more?)
    3. C1 vs C4: Single model vs Multi-agent debate (full debate gain)
    4. C2 vs C3: Self-consistency vs Parallel vote (same-model vs cross-model voting)
    5. C3 vs C4: Parallel vote vs Multi-agent debate (voting vs inter-agent revision)
    """
    pairs = [
        ("C1", "C2", "Single model vs Self-consistency (Same-model sampling)"),
        ("C1", "C3", "Single model vs Parallel vote (Cross-model diversity)"),
        ("C1", "C4", "Single model vs Multi-agent debate (Full debate gain)"),
        ("C2", "C3", "Self-consistency vs Parallel vote (Sampling vs Diversity)"),
        ("C3", "C4", "Parallel vote vs Multi-agent debate (Voting vs Debate)"),
    ]
    results = []
    n_tests = len(pairs)

    for cond_a, cond_b, desc in pairs:
        sub_a = df[df["condition"] == cond_a].set_index("task_id")["score"]
        sub_b = df[df["condition"] == cond_b].set_index("task_id")["score"]
        common = sub_a.index.intersection(sub_b.index)
        if len(common) < 3:
            continue

        a_vals = sub_a.loc[common].values
        b_vals = sub_b.loc[common].values
        diff = b_vals - a_vals

        t_stat, p_val = stats.ttest_rel(b_vals, a_vals)
        p_bonf = min(p_val * n_tests, 1.0)
        sd_diff = np.std(diff, ddof=1)
        cohens_d = np.mean(diff) / sd_diff if sd_diff > 0 else 0.0

        results.append(ConditionPairComparison(
            comparison=f"{cond_a} vs {cond_b}",
            cond_a=cond_a,
            cond_b=cond_b,
            description=desc,
            mean_a=float(np.mean(a_vals)),
            mean_b=float(np.mean(b_vals)),
            gain=float(np.mean(diff)),
            t_stat=float(t_stat),
            p_value=float(p_val),
            p_bonferroni=float(p_bonf),
            cohens_d=float(cohens_d),
            significant=p_bonf < 0.05,
            n=len(common),
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Effect size summary
# ─────────────────────────────────────────────────────────────────────────────

def effect_size_label(d: float) -> str:
    d = abs(d)
    if d >= 0.8: return "Large"
    if d >= 0.5: return "Medium"
    if d >= 0.2: return "Small"
    return "Negligible"


# ─────────────────────────────────────────────────────────────────────────────
# Complexity × Condition interaction
# ─────────────────────────────────────────────────────────────────────────────

def complexity_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show accuracy per (difficulty × condition).
    Returns DataFrame: rows=difficulty, cols=conditions.
    """
    if df.empty:
        return pd.DataFrame()
    table = df.groupby(["difficulty", "condition"])["score"].mean().unstack("condition")
    for cond in CONDITIONS:
        if cond not in table.columns:
            table[cond] = np.nan
    table = table[list(CONDITIONS.keys())]
    # Sort difficulty
    order = ["Easy", "Medium", "Hard"]
    table = table.reindex([d for d in order if d in table.index])
    # Add gap column
    if "C1" in table.columns and "C4" in table.columns:
        table["C4-C1 Gap"] = table["C4"] - table["C1"]
    return table.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Token efficiency
# ─────────────────────────────────────────────────────────────────────────────

def token_efficiency(df: pd.DataFrame) -> list[TokenEfficiency]:
    """
    Compute accuracy-per-1000-tokens for each condition.
    Used for the Pareto efficiency plot.
    """
    if df.empty:
        return []
    results = []
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        avg_acc = sub["score"].mean()
        avg_tok = sub["tokens_total"].mean()
        results.append(TokenEfficiency(
            condition=cond,
            avg_accuracy=round(float(avg_acc), 4),
            avg_tokens=round(float(avg_tok), 1),
            accuracy_per_1k_tokens=round(float(avg_acc / (avg_tok / 1000)) if avg_tok > 0 else 0, 4),
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model contribution in C4 (novel finding)
# ─────────────────────────────────────────────────────────────────────────────

def model_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    For C4: which model's individual score was highest most often?
    Returns DataFrame with model win counts.
    """
    c4 = df[df["condition"] == "C4"].copy()
    if c4.empty:
        return pd.DataFrame()

    win_counts = {mid: 0 for mid in MODELS}
    total = 0

    for _, row in c4.iterrows():
        scores = row.get("model_scores", {})
        if not scores or not isinstance(scores, dict):
            continue
        if not scores:
            continue
        best_model = max(scores, key=lambda k: scores[k])
        win_counts[best_model] = win_counts.get(best_model, 0) + 1
        total += 1

    if total == 0:
        return pd.DataFrame()

    rows = []
    for mid, count in win_counts.items():
        rows.append({
            "model_id": mid,
            "model_name": MODELS[mid].name,
            "wins": count,
            "win_rate": round(count / total, 4) if total > 0 else 0,
        })

    return pd.DataFrame(rows).sort_values("wins", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode tagging
# ─────────────────────────────────────────────────────────────────────────────

def tag_failure_mode(response: str, task_answer: str, category: str = "") -> str:
    """
    Classify failure mode matching Paper 1 Section IV-D taxonomy:
    - Reasoning error | Factual error | Hallucination | Incomplete
    """
    if not response or len(response.strip()) < 10:
        return "Incomplete"
    
    cat = (category or "").lower()
    resp_lower = response.lower()
    
    if any(m in resp_lower for m in ["hallucin", "unknown entity", "unregistered", "fabricated"]):
        return "Hallucination"
    elif cat in ["math", "logic", "coding", "planning"]:
        return "Reasoning error"
    elif cat in ["science", "knowledge"]:
        return "Factual error"
    else:
        return "Reasoning error"


def failure_mode_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return percentage distribution of failure modes per condition (Table II of Paper 1).
    Operates on incorrect responses (score < 0.8).
    """
    if df.empty:
        return pd.DataFrame()

    failures = df[df["score"] < 0.8].copy()
    if failures.empty:
        return pd.DataFrame()

    if "failure_mode" not in failures.columns or failures["failure_mode"].isnull().all():
        task_map = {t.id: (t.answer, t.category) for t in ALL_TASKS}
        failures["failure_mode"] = failures.apply(
            lambda r: tag_failure_mode(
                r.get("final_answer", ""),
                task_map.get(r.get("task_id"), ("", ""))[0],
                task_map.get(r.get("task_id"), ("", ""))[1]
            ),
            axis=1
        )

    # Filter out 'None' if any
    failures = failures[failures["failure_mode"] != "None"]
    if failures.empty:
        return pd.DataFrame()

    counts = failures.groupby(["condition", "failure_mode"]).size().unstack(fill_value=0)
    
    # Ensure standard order of columns
    expected_modes = ["Reasoning error", "Factual error", "Hallucination", "Incomplete"]
    for mode in expected_modes:
        if mode not in counts.columns:
            counts[mode] = 0
    counts = counts[expected_modes]

    # Calculate percentage per condition
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100
    return pcts.round(1)


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's Test for Paired Binary Outcomes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class McNemarResult:
    cond_a: str
    cond_b: str
    a_correct_b_wrong: int  # b
    a_wrong_b_correct: int  # c
    both_correct: int       # a
    both_wrong: int         # d
    chi2_stat: float
    p_value: float
    significant: bool


def mcnemar_test(df: pd.DataFrame, cond_a: str = "C1", cond_b: str = "C4") -> Optional[McNemarResult]:
    """
    McNemar's test for paired binary correctness between two conditions.
    Uses Yates' continuity correction.
    """
    if df.empty:
        return None

    sub_a = df[df["condition"] == cond_a].set_index("task_id")["score"]
    sub_b = df[df["condition"] == cond_b].set_index("task_id")["score"]
    common = sub_a.index.intersection(sub_b.index)

    if len(common) == 0:
        return None

    # Binary correctness (score >= 0.8)
    a_corr = (sub_a.loc[common] >= 0.8).values
    b_corr = (sub_b.loc[common] >= 0.8).values

    both_corr = int(np.sum(a_corr & b_corr))
    both_wrng = int(np.sum(~a_corr & ~b_corr))
    a_corr_b_wrng = int(np.sum(a_corr & ~b_corr))
    a_wrng_b_corr = int(np.sum(~a_corr & b_corr))

    b = a_corr_b_wrng
    c = a_wrng_b_corr

    if (b + c) == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        # Yates continuity correction
        chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = float(stats.chi2.sf(chi2, 1))

    return McNemarResult(
        cond_a=cond_a,
        cond_b=cond_b,
        a_correct_b_wrong=b,
        a_wrong_b_correct=c,
        both_correct=both_corr,
        both_wrong=both_wrng,
        chi2_stat=float(chi2),
        p_value=float(p_val),
        significant=p_val < 0.05,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap 95% Confidence Intervals for Accuracy Differences
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    df: pd.DataFrame,
    cond_a: str = "C1",
    cond_b: str = "C4",
    n_boot: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """
    Compute 95% Bootstrap Confidence Interval for accuracy difference (Cond_B - Cond_A).
    """
    if df.empty:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    sub_a = df[df["condition"] == cond_a].set_index("task_id")["score"]
    sub_b = df[df["condition"] == cond_b].set_index("task_id")["score"]
    common = sub_a.index.intersection(sub_b.index)

    if len(common) == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    diffs = sub_b.loc[common].values - sub_a.loc[common].values
    n = len(diffs)

    rng = np.random.default_rng(random_seed)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_means, (alpha / 2.0) * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "ci_level": ci_level,
        "n_samples": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Debate Transition Matrix (Consensus Collapse vs Self-Correction)
# ─────────────────────────────────────────────────────────────────────────────

def debate_transitions(df: pd.DataFrame) -> dict:
    """
    Analyze C4 debate dynamics between Round 1 initial answers and Round 2 revisions.
    Measures:
    - Wrong -> Correct (Self-Correction Rate / Enlightenment)
    - Correct -> Wrong (Consensus Collapse / Suppression Rate)
    - Correct -> Correct (Stability Rate)
    - Wrong -> Wrong (Persistent Error Rate)
    """
    c4 = df[df["condition"] == "C4"].copy()
    if c4.empty:
        return {}

    from src.graders import grade_exact, grade_numeric, grade_contains
    from src.tasks import ALL_TASKS

    task_map = {t.id: t for t in ALL_TASKS}
    transitions = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0,
        "wrong_to_wrong": 0,
        "total_agent_revisions": 0,
    }

    for _, row in c4.iterrows():
        t = task_map.get(row["task_id"])
        if not t:
            continue
        r1 = row.get("round1_responses", {})
        r2 = row.get("round2_responses", {})
        if not isinstance(r1, dict) or not isinstance(r2, dict):
            continue

        for mid in r1:
            if mid not in r2:
                continue
            from src.graders import extract_final_answer, grade_exact, grade_numeric
            ans1 = extract_final_answer(r1[mid])
            ans2 = extract_final_answer(r2[mid])

            if t.grader_hint == "numeric":
                corr1 = grade_numeric(ans1, t.answer) >= 0.8
                corr2 = grade_numeric(ans2, t.answer) >= 0.8
            else:
                corr1 = grade_exact(ans1, t.answer) >= 0.8
                corr2 = grade_exact(ans2, t.answer) >= 0.8

            transitions["total_agent_revisions"] += 1
            if not corr1 and corr2:
                transitions["wrong_to_correct"] += 1
            elif corr1 and not corr2:
                transitions["correct_to_wrong"] += 1
            elif corr1 and corr2:
                transitions["correct_to_correct"] += 1
            else:
                transitions["wrong_to_wrong"] += 1

    total = max(1, transitions["total_agent_revisions"])
    return {
        "total_revisions": transitions["total_agent_revisions"],
        "self_correction_rate": round(transitions["wrong_to_correct"] / total, 4),
        "consensus_collapse_rate": round(transitions["correct_to_wrong"] / total, 4),
        "stability_rate": round(transitions["correct_to_correct"] / total, 4),
        "persistent_error_rate": round(transitions["wrong_to_wrong"] / total, 4),
        "counts": transitions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Empirical Difficulty Classification
# ─────────────────────────────────────────────────────────────────────────────

def empirical_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes empirical difficulty per task based on mean accuracy across all evaluated conditions:
    - Easy:   Accuracy > 85%
    - Medium: 50% <= Accuracy <= 85%
    - Hard:   Accuracy < 50%
    """
    if df.empty:
        return pd.DataFrame()
    task_acc = df.groupby(["task_id", "category"])["score"].mean().reset_index()

    def label_emp_diff(acc: float) -> str:
        if acc > 0.85:
            return "Easy"
        elif acc >= 0.50:
            return "Medium"
        else:
            return "Hard"

    task_acc["empirical_difficulty"] = task_acc["score"].apply(label_emp_diff)
    return task_acc.sort_values("task_id").reset_index(drop=True)
