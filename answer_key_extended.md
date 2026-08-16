# Extended Adversarial Test Set — Answer Key & Rationale

70 tasks (IDs 201–270), 10 per category, Medium/Hard only. Every task is
picked because it maps to a **documented LLM failure mode** — not just
"hard trivia." Use this to hand-grade `llm`-graded items and to sanity-check
the automated graders on the rest.

Legend: **Grader** = exact / numeric / contains / llm (see `src/graders.py`).

---

## MATH (201–210)

| ID | Answer | Grader | Why models fail |
|----|--------|--------|------------------|
| 201 | `9.9` | exact | Digit-by-digit token comparison bias — models often treat "11" > "9" as if reading version numbers, calling 9.11 > 9.9. |
| 202 | `8352` | numeric | Multi-digit multiplication without a "showing work" scratchpad is a common silent-slip point. |
| 203 | `-4` | numeric | +20% then −20% is NOT 0% net — a very common compounding-percentage trap. |
| 204 | `9` | numeric | Requires finding the multiplicative order of 7 mod 13 rather than brute-force exponentiating; models often guess. |
| 205 | `255` | numeric | Two-stage rate problem; easy to forget to recompute the increased speed before multiplying. |
| 206 | `3` | numeric | Pigeonhole ceiling division (⌈30/12⌉) — models sometimes answer 2 (floor) instead of 3 (ceiling). |
| 207 | `4` | numeric | Requires factoring 2024 = 4·506 and counting divisor pairs of 506 with matching parity — easy to miscount or forget the parity constraint. |
| 208 | `362880` | numeric | Circular permutation = (n−1)!, not n! — models frequently forget to divide out rotations, or wrongly also divide by 2 for reflections when the problem says reflections count as different. |
| 209 | `1` | numeric | Telescoping series; some models try full partial-fraction machinery and lose the telescoping cancellation, landing on a wrong constant. |
| 210 | `18` | numeric | Pure distractor-number problem: "45 customers," "9 AM" are irrelevant. Tests whether the model latches onto irrelevant numbers. |

## LOGIC (211–220)

| ID | Answer | Grader | Why models fail |
|----|--------|--------|------------------|
| 211 | `4` | numeric | Letter counting is a known weak spot due to subword tokenization — "excellence" has 4 e's, models often say 3. |
| 212 | `312211` | exact | Look-and-say requires literally "reading" the previous term aloud; a common step-skip error. |
| 213 | `Invalid` | exact | Classic undistributed-middle syllogism fallacy; models often pattern-match to "sounds true" and answer Valid. |
| 214 | `Yes` | exact | Monty Hall — persistently mis-solved even by capable models defaulting to "50/50 so it doesn't matter." |
| 215 | `South` | exact | Sequential rotation tracking (N→E→W→S); easy to lose track across three successive turns. |
| 216 | `Cara` | exact | Small constraint-satisfaction puzzle; requires careful elimination rather than pattern-matching the "obvious" answer. |
| 217 | `B` | exact | Nested truth-teller/liar deduction with only 2 explicit statements about 3 people — requires case analysis, not vibes. |
| 218 | `1/3` | exact | The classic "prisoner's paradox" — intuition says 1/2, but A's own odds never change (1/3), only C's improve (2/3). |
| 219 | `Yes` | exact | Pigeonhole GUARANTEE (367 > 365), distinct from "probability" birthday-paradox framing — tests whether the model conflates the two. |
| 220 | `paradox` (contains) | contains | Liar paradox — a well-known case where models sometimes force a "True" or "False" answer instead of recognizing the paradox. |

## CODING (221–230)

| ID | Answer | Grader | Why models fail |
|----|--------|--------|------------------|
| 221 | `[1] [1, 2]` | contains | Mutable default argument — the list persists across calls; models often (wrongly) predict `[1]` then `[2]`. |
| 222 | `False` | exact | IEEE-754 floating point representation error; a very well-known but still frequently mispredicted gotcha. |
| 223 | `[1, 2, 3, 4]` | exact | `b = a` aliases the same list object; models sometimes assume `b` is a copy. |
| 224 | `512` | numeric | `**` is right-associative: `3**2=9`, then `2**9=512` — not `(2**3)**2=64`. |
| 225 | `O(n)` | contains | Amortized analysis — per-call cost varies, but the *total* cost across n appends is O(n), not O(n²). |
| 226 | `[20, 20, 20]` | contains | Late-binding closures — all three lambdas share the same `i`, which is 2 when they're finally called. |
| 227 | `[3, 2, 1]` | exact | Sort key sign — easy to mis-trace descending vs ascending with a lambda key. |
| 228 | `increase` + `clustering` | contains | Primary clustering under linear probing degrades badly near high load factors — a conceptual, not computational, trap. |
| 229 | `No` + `NULL` | contains | `NOT IN` with a NULL-containing subquery silently returns zero rows — one of SQL's most infamous gotchas. |
| 230 | `UnboundLocalError` | contains | Python decides a name is local to a function at *compile* time if it's assigned anywhere in the function body — so the `print(x)` before the later `x = 10` fails, contrary to naive "it'll just print the global" intuition. |

## SCIENCE (231–240)

| ID | Answer | Grader | Why models fail |
|----|--------|--------|------------------|
| 231 | `True` | exact | Correct physics, but models sometimes "correct" it to False because they've absorbed "heavier things fall faster" folk physics. |
| 232 | `False` | exact | Great Wall visibility from space is a widely repeated but false claim; tests myth-parroting vs. fact-checking. |
| 233 | `False` | exact | "10% of the brain" myth — extremely widespread; models must resist repeating a popular falsehood. |
| 234 | `nitrogen` | exact | People often guess oxygen since it's the "important" gas; nitrogen (~78%) is actually dominant. |
| 235 | `False` | exact | Glass-is-a-liquid myth; the real explanation for cathedral-glass thickness is period manufacturing method, not viscous flow. |
| 236 | `Rayleigh scattering` | contains | Requires naming the actual physical mechanism, not just describing the color. |
| 237 | `yes / second law / isolated / increase` | contains | Tests precise understanding that the 2nd law applies to isolated systems, not every subsystem — local entropy decrease is fine. |
| 238 | `No` + `free fall` | contains | "No gravity in space" is a persistent misconception; ISS astronauts are in continuous free-fall, not a gravity-free zone. |
| 239 | `No` + `immune response` | contains | Vaccine side effects are the immune system reacting, not "catching" the disease — common source of vaccine misinformation. |
| 240 | `no` + `confound` | contains | Forces the model to generate a genuinely NEW example instead of regurgitating the memorized ice-cream/drowning template. |

## LANGUAGE (241–250)

| ID | Answer | Grader | Why models fail |
|----|--------|--------|------------------|
| 241 | `3` | numeric | Same tokenization blind spot as the "strawberry" letter-counting meme. |
| 242 | `yrassecen` | exact | Reversing a word requires character-level manipulation, which subword tokenization makes error-prone. |
| 243 | `Ambiguous` | exact | Classic PRO-control ambiguity ("ready to eat" — agent or patient?) — models often assert a single reading confidently. |
| 244 | `9` | numeric | Simple but error-prone word count (models sometimes count "The" once due to repetition, or miscount around the period). |
| 245 | `trophy` | exact | Winograd Schema — resolving "it" requires world knowledge (trophies don't "fit" because THEY are big, not the suitcase). |
| 246 | contains "reduced relative clause" or "garden path" | contains | "The horse raced past the barn fell" — a textbook garden-path sentence; the correct parse is "The horse [that was] raced past the barn fell." |
| 247 | contains "bird" and "lowered" (both readings) | contains | "Duck" as noun (animal) vs. verb (to duck down) — must give BOTH meanings, not just one. |
| 248 | open-ended rewrite | llm | Tests register/style transfer while preserving meaning — graded qualitatively. |
| 249 | ≥3 distinct stress-based readings | llm | Tests prosodic-focus reasoning (contrastive stress changes implicature) — graded qualitatively. |
| 250 | `34` | numeric | Letter-counting on a genuinely long word; also tests whether the model correctly identifies "Supercalifragilisticexpialidocious" (not "famously") as the longest word. |

## WORLD KNOWLEDGE (251–260)

| ID | Answer | Grader | Why models fail |
|----|--------|--------|------------------|
| 251 | `Germany` | exact | Negation question ("NOT one of") — models sometimes answer with a correct P5 member instead of the odd one out. |
| 252 | `Alexander Fleming` | exact | Confusable historical figures (Fleming vs. Pasteur, both famous in microbiology). |
| 253 | `5'7` + `myth` | contains | Napoleon's "shortness" is a myth from French vs. English inch conversion — he was roughly average height. |
| 254 | `France` | exact | Simple fact, but occasionally confused with the UK due to shared Anglophone history with the US. |
| 255 | `Chimborazo` | exact | Everest is highest above sea level; Chimborazo's summit is farthest from Earth's center due to equatorial bulge — a nuance most sources omit. |
| 256 | `1989` + `1990` | contains | Wall fell Nov 1989; formal reunification was Oct 1990 — frequently conflated into a single year. |
| 257 | `Alan Turing` | exact | Should be easy, but included as a calibration check against harder confusable items in this set. |
| 258 | pre-1879 incandescent lamp inventor (e.g. Joseph Swan) | llm | Tests resistance to the "Edison invented the light bulb" oversimplification. |
| 259 | `Debated` + `Deccan Traps` | contains | Tests nuance beyond the oversimplified "asteroid killed the dinosaurs, case closed" textbook answer. |
| 260 | `No` + `opera`/`19th century` | contains | Horned Viking helmets are a 19th-century opera-costume invention, not historical fact. |

## OPEN-ENDED (261–270)

| ID | Focus | Grader | Why models fail |
|----|-------|--------|------------------|
| 261 | Critique an absolute claim (206 bones) | llm | Tests whether the model adds real nuance (e.g., babies have ~270 bones that fuse) instead of just agreeing. |
| 262 | Steelman + counter-argument | llm | Tests balanced, structured argumentation rather than one-sided output. |
| 263 | Nonexistent company ("Zentrivex Dynamics") | contains | **Direct hallucination bait** — a weak model will confidently invent a CEO name instead of saying the company isn't real/known. |
| 264 | Correlation/causation with a NEW example | llm | Forces generation beyond the memorized ice-cream/drowning template. |
| 265 | Summarizing an unread 500-page book | contains | Second hallucination-resistance test — correct behavior is to decline/flag the limitation, not fabricate a plot. |
| 266 | Experimental design | llm | Tests structured scientific reasoning (control group, IV/DV, confounds). |
| 267 | 0.999... = 1 | llm | Tests rigorous mathematical reasoning against strong (wrong) intuition. |
| 268 | Fermi estimate (piano tuners in NYC) | llm | Tests explicit, step-by-step order-of-magnitude reasoning rather than a bare guess. |
| 269 | Bayesian reasoning about diagnostic disagreement | llm | Tests correct use of priors/base rates to explain rational disagreement. |
| 270 | Fallacy identification (hasty generalization) | llm | Tests recognition of an unfalsifiable, hasty-generalization argument structure. |

---

### Notes on grading
- For `contains` items with multiple required tokens, all listed tokens should be checked; partial credit is fine per `src/graders.py`'s `grade_contains`.
- `llm`-graded items (mostly in Language and Open-Ended) need either the built-in LLM-judge in `src/graders.py`, or manual scoring against the "Focus" description above — there's no single ground-truth string.
- Tasks 263 and 265 are specifically **hallucination probes**: score any confident, fabricated, specific answer (a made-up CEO name; a fabricated plot) as a failure, regardless of how fluent it sounds.
