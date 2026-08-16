# Multi-Agent vs Single-Model Research Framework
# Research Paper 1 — Implementation

A full empirical study comparing **4 LLM reasoning conditions** across **7 task categories** and **140 tasks**.  
All experiments run in the **terminal** using free OpenRouter models.

---

## Project Structure

```
Multimodel/
├── src/
│   ├── config.py         # Models, conditions, categories
│   ├── client.py         # Async OpenRouter API client
│   ├── tasks.py          # 140-task dataset (7 categories × 20 tasks)
│   ├── graders.py        # Exact / numeric / contains / LLM-as-judge scoring
│   ├── conditions.py     # C1, C2, C3, C4 runners
│   ├── database.py       # SQLite persistence + pandas export
│   ├── analysis.py       # t-tests, Cohen's d, Bonferroni, token efficiency
│   └── display.py        # Rich terminal tables and progress bars
├── scripts/
│   ├── smoke_test.py     # Test API keys + model connectivity
│   └── demo_debate.py    # Watch a single C4 debate live
├── results/              # Auto-created (SQLite DB + JSON/CSV exports)
├── run_experiment.py     # Main experiment runner
├── analyze.py            # Full statistical analysis report
├── requirements.txt
└── .env.example
```

---

## Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (free at https://openrouter.ai)
copy .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=sk-or-...

# 3. Test connectivity
python scripts/smoke_test.py
```

---

## Running the Experiment

```bash
# Quick test — 10 tasks, all 4 conditions (~40 API calls, ~5 min)
python run_experiment.py --n 10

# Single category
python run_experiment.py --category math

# Only C1 (baseline) and C4 (debate)
python run_experiment.py --conditions C1 C4

# Full experiment — 140 tasks × 4 conditions (~560–1120 calls, ~2–4 hours)
python run_experiment.py

# Resume after interruption (default behavior)
python run_experiment.py --resume
```

---

## Viewing Results

```bash
# Full statistical analysis report
python analyze.py

# Brief (accuracy table + stats only)
python analyze.py --brief

# Single category
python analyze.py --category logic

# Export to JSON + CSV
python analyze.py --export
```

---

## Watching a Live Debate (C4)

```bash
# Random task
python scripts/demo_debate.py

# Specific task
python scripts/demo_debate.py --task_id 15

# Random hard math task
python scripts/demo_debate.py --category math
```

---

## Experimental Design

| Condition | Name               | Calls/Task | Description |
|-----------|-------------------|------------|-------------|
| C1        | Single Model       | 1          | One model, one call, temperature=0.0 |
| C2        | Self-Consistency   | 3          | Same model, 3 calls at temp=0.7, majority vote |
| C3        | Parallel Vote      | 4          | All 4 models independently, majority vote |
| C4        | Multi-Agent Debate | 8          | Round 1: all answer. Round 2: each revises after seeing others. Majority vote. |

### Models (all free via OpenRouter)
| ID | Name | API Model |
|----|------|-----------|
| A  | DeepSeek-R1    | `deepseek/deepseek-r1:free` |
| B  | Gemma-4-31B-IT | `google/gemma-3-27b-it:free` |
| C  | Qwen3-30B-A3B  | `qwen/qwen3-30b-a3b:free` |
| D  | Llama-4-Scout  | `meta-llama/llama-4-scout:free` |

### Task Categories (20 tasks each)
| ID         | Category               | Grader      | Source |
|------------|----------------------|-------------|--------|
| math       | Mathematical Reasoning | Numeric     | MATH + GSM8K |
| logic      | Logical Deduction      | Exact/LLM   | LogiQA + BBH |
| coding     | Coding & Algorithms    | Exact       | HumanEval + MBPP |
| science    | Scientific Knowledge   | Exact       | SciQ + ARC |
| language   | Language Understanding | Contains    | SuperGLUE |
| knowledge  | World Knowledge        | Contains    | TriviaQA |
| openended  | Open-Ended Reasoning   | LLM Judge   | Custom |

---

## Statistical Analysis (Week 6)

The `analyze.py` script produces:

1. **Table 1** — Accuracy by category × condition (heatmap-style terminal table)
2. **Table 2** — Paired t-tests (C1 vs C4) with Bonferroni correction, Cohen's d
3. **Figure 2** — Complexity × condition interaction (Easy / Medium / Hard)
4. **Figure 3** — Token efficiency Pareto (accuracy per 1,000 tokens)
5. **Novel Finding** — Which model in C4 debate contributes most correct answers
6. **Failure Modes** — Factual Error / Reasoning Error / Hallucination / Incomplete

---

## Expected Results (from Implementation Plan)

- **C4 > C3 > C2 > C1** on overall accuracy
- Largest gains on **Hard tasks** (hypothesis: collective error-correction scales with difficulty)
- **Math and Coding** show highest absolute improvement
- **C4 token cost** is ~8× C1 but accuracy-per-1K-tokens may favor C2 or C3

---

## API Cost Estimate (Free Tier)

| Run | Tasks | Conditions | Est. Calls | Est. Time |
|-----|-------|-----------|------------|-----------|
| Quick test | 10 | C1+C4 | ~90 | ~5 min |
| Category run | 20 | all 4 | ~300 | ~20 min |
| Full experiment | 140 | all 4 | ~1,680 | ~2-4 hours |

Free-tier OpenRouter models have rate limits. The runner uses a semaphore (`CONCURRENCY=3`) to stay within limits. All results are checkpointed to SQLite — safe to interrupt and resume.
