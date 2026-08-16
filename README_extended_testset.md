# MMRA Extended Adversarial Test Set

A companion 70-task benchmark for [Goldypahal/MMRA](https://github.com/Goldypahal/MMRA),
purpose-built to trip up LLMs. Where the original `src/tasks.py` (140 tasks)
is broad Easy/Medium/Hard coverage, this set is narrow and adversarial:
**every single question targets one specific, documented way LLMs fail**
(digit-comparison bias, letter counting, closures-over-loop-variables,
NULL-in-`NOT IN`, popular-but-false trivia, hallucination bait, etc.)

## Files

| File | Purpose |
|---|---|
| `tasks_extended.py` | Drop-in Python module — reuses MMRA's own `Task` dataclass. IDs 201–270, no collision with the original 1–140. |
| `tasks_extended.json` | Same 70 tasks as portable JSON, for use outside the Python framework (any language/pipeline). |
| `answer_key_extended.md` | Ground-truth answers + a one-line explanation of *why* each question is a known LLM failure mode. Needed for manually grading the `llm`-graded (open-ended) items. |
| `run_extended_accuracy_check.py` | Runner that scores a model against the set. Auto-detects whether it's sitting inside the MMRA repo (uses `src.client` + `src.graders`) or running standalone. |

## Composition

70 tasks, 10 per category, all Medium or Hard (no Easy — this set is meant
to find failures, not confirm competence):

| Category | Medium | Hard | Failure modes targeted |
|---|---|---|---|
| math | 6 | 4 | decimal comparison, compounding %, modular arithmetic, red-herring numbers, circular permutations, telescoping series |
| logic | 5 | 5 | letter counting, Monty Hall, liar/truth-teller nesting, prisoner's paradox, pigeonhole guarantee vs. probability, liar paradox |
| coding | 5 | 5 | mutable default args, float equality, aliasing, operator precedence, late-binding closures, `NOT IN` + NULL, UnboundLocalError |
| science | 5 | 5 | popular myths (10% brain, Great Wall from space, glass-is-liquid), microgravity misconception, entropy/2nd-law nuance |
| language | 5 | 5 | letter/word counting, Winograd schema, garden-path sentences, lexical ambiguity, prosodic focus |
| knowledge | 5 | 5 | negation traps ("NOT one of..."), confusable historical figures, oversimplified textbook answers |
| openended | 5 | 5 | hallucination bait (nonexistent company/book), Bayesian reasoning, Fermi estimation, fallacy identification |

## Quick start

**Option A — inside the MMRA repo (recommended, gives you the same
statistical pipeline as `analyze.py`):**

```bash
# from the MMRA repo root
cp tasks_extended.py .
cp run_extended_accuracy_check.py .

python run_extended_accuracy_check.py --model A
python run_extended_accuracy_check.py --all-models --export results_extended.csv
python run_extended_accuracy_check.py --category coding --model C
```

To run it through all 4 conditions (C1 single-model, C2 self-consistency,
C3 parallel vote, C4 debate) exactly like the original experiment, merge
it into the master list before calling `run_experiment.py`:

```python
# in src/tasks.py, after ALL_TASKS is defined:
from tasks_extended import EXTENDED_TASKS
ALL_TASKS = ALL_TASKS + EXTENDED_TASKS
```
(then update the `assert len(ALL_TASKS) == 140` line to `210`, or remove it)

**Option B — standalone, against any model:**

Use `tasks_extended.json` directly — loop over it, send `text` to your
model of choice, and grade the response using the rules in
`answer_key_extended.md` (or reimplement the exact/numeric/contains logic
from `_standalone_grade()` in the runner script).

## Reading results

- **exact / numeric / contains** items are auto-gradable — no ambiguity in
  what "correct" means.
- **llm**-graded items (mostly in Language and Open-Ended) need either
  MMRA's built-in LLM-judge (`src/graders.py::grade_llm`) or manual
  scoring against the "Focus" column in `answer_key_extended.md`.
- Pay special attention to **#263** and **#265** — these are pure
  hallucination probes (a nonexistent company, an unread book). Any
  fluent, confident, *fabricated* answer should be scored as a failure
  regardless of how plausible it sounds.
