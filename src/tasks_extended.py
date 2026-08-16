"""
tasks_extended.py — Adversarial "failure-mode" extension for MMRA.

This is a companion to src/tasks.py. It reuses the same `Task`
dataclass and grader_hint conventions (exact | numeric | contains | llm),
so it plugs directly into the existing pipeline (src/graders.py,
src/conditions.py, run_experiment.py, analyze.py).

Unlike tasks.py (which leans on raw difficulty), every task here is chosen
because it targets a SPECIFIC, well-documented way LLMs fail:

  - arithmetic/precision slips on multi-step math
  - digit/magnitude comparison errors ("9.11 vs 9.9")
  - overcounting/undercounting combinatorics
  - late-binding closures & mutable-default-argument traps in code
  - floating point / operator-precedence surprises
  - classic cognitive-bias logic puzzles (Monty Hall, prisoners, liars)
  - letter/word counting (tokenization blind spots)
  - Winograd-schema style pronoun ambiguity
  - garden-path sentences
  - popular-but-false "trivia" (misconceptions models parrot)
  - negation / "NOT" questions
  - hallucination bait (asking about entities that don't exist)
  - red-herring numeric distractors in word problems

IDs 201–270 (7 categories × 10 tasks) so they never collide with the
original 140-task set (IDs 1–140).
"""

from src.tasks import Task

# ─────────────────────────────────────────────────────────────────────────────
# MATH — arithmetic precision, comparison traps, red herrings  (IDs 201-210)
# ─────────────────────────────────────────────────────────────────────────────
MATH_EXT = [
    Task(201, "math", "Medium", "Which number is larger: 9.11 or 9.9? Answer with just the larger number.", "9.9", "exact", "decimal comparison trap"),
    Task(202, "math", "Medium", "What is 87 x 96?", "8352", "numeric", "multi-digit multiplication"),
    Task(203, "math", "Medium", "A jacket's price is increased by 20%, then the new price is decreased by 20%. What is the net percentage change from the ORIGINAL price? (Give a signed number, e.g. -4 or 4)", "-4", "numeric", "percentage compounding trap"),
    Task(204, "math", "Medium", "Compute 7^100 mod 13.", "9", "numeric", "modular exponentiation"),
    Task(205, "math", "Medium", "A train travels at 60 mph for 2 hours, then its speed increases by 50% for another 1.5 hours. How many total miles does it travel?", "255", "numeric", "multi-step rate problem"),
    Task(206, "math", "Medium", "By the pigeonhole principle, in a room of 30 people, what is the minimum number of people who are guaranteed to share the same birth month (12 months in a year)?", "3", "numeric", "pigeonhole principle"),
    Task(207, "math", "Hard", "How many ordered pairs of positive integers (x, y) satisfy x^2 - y^2 = 2024?", "4", "numeric", "difference of squares / factor pairing"),
    Task(208, "math", "Hard", "In how many distinct ways can 10 people be seated around a circular table, if rotations are considered identical but mirror-image (reflected) seatings are considered different?", "362880", "numeric", "circular permutations"),
    Task(209, "math", "Hard", "Evaluate the infinite sum: Sum from k=1 to infinity of 1/(k(k+1)).", "1", "numeric", "telescoping series"),
    Task(210, "math", "Hard", "A store sells notebooks for $3 each and pens for $2 each. The store opened at 9 AM and 45 customers visited yesterday. If one customer buys 4 notebooks and 3 pens, how much do they pay in total, in dollars? (Note: some numbers in this problem are irrelevant.)", "18", "numeric", "red-herring distractor numbers"),
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGIC — self-reference, classic paradoxes, constraint puzzles  (IDs 211-220)
# ─────────────────────────────────────────────────────────────────────────────
LOGIC_EXT = [
    Task(211, "logic", "Medium", "How many times does the letter 'e' appear in the word 'excellence'? (count carefully, letter by letter)", "4", "numeric", "letter counting / tokenization blind spot"),
    Task(212, "logic", "Medium", "This is the 'look-and-say' sequence: 1, 11, 21, 1211, 111221, ? What is the next term?", "312211", "exact", "look-and-say sequence"),
    Task(213, "logic", "Medium", "Evaluate this argument for validity: 'All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly.' Is the argument Valid or Invalid?", "Invalid", "exact", "undistributed middle fallacy"),
    Task(214, "logic", "Medium", "Classic Monty Hall: 3 doors, one has a car, two have goats. You pick door 1. The host, who knows what's behind every door, opens door 3 revealing a goat. Should you switch to door 2 to maximize your chance of winning the car? Answer 'Yes' or 'No'.", "Yes", "exact", "Monty Hall probability"),
    Task(215, "logic", "Medium", "Facing North, you turn 90 degrees clockwise, then 180 degrees, then 90 degrees counter-clockwise. Which single direction are you now facing?", "South", "exact", "sequential spatial rotation"),
    Task(216, "logic", "Hard", "Four friends -- Amy, Ben, Cara, Dan -- each own a different pet (cat, dog, fish, bird) and live on different floors (1-4) of a building. Clues: (1) The bird owner lives on floor 4. (2) Ben lives on floor 3 and owns the fish. (3) Amy lives directly above Dan. (4) Cara does not live on floor 1. (5) The cat owner lives on floor 1. Who owns the bird?", "Cara", "exact", "constraint-satisfaction (zebra-style) puzzle"),
    Task(217, "logic", "Hard", "A says: 'B always lies.' B says: 'C and I are both liars.' Each person is either a truth-teller (always true) or a liar (always false), with no statement given about C directly. Who is the liar among A, B, and C?", "B", "exact", "nested liar/truth-teller deduction"),
    Task(218, "logic", "Hard", "Three prisoners A, B, C: exactly one (chosen uniformly at random) will be pardoned; the warden knows who. Prisoner A, who is not told his own fate, asks the warden to name one of B or C who will be executed. The warden truthfully names B. After hearing this, what is A's probability of being pardoned? Express as a fraction.", "1/3", "exact", "prisoner's paradox (Monty-Hall variant)"),
    Task(219, "logic", "Hard", "In a group of 367 people, is it guaranteed that at least two of them share the same birthday, ignoring leap years (365 possible birthdays)? Answer 'Yes' or 'No'.", "Yes", "exact", "pigeonhole principle (guarantee, not probability)"),
    Task(220, "logic", "Hard", "Consider the self-referential statement: 'This sentence is false.' Is the statement definitively True, definitively False, or neither (i.e. paradoxical)? Answer in one word.", "paradox", "contains", "liar paradox"),
]

# ─────────────────────────────────────────────────────────────────────────────
# CODING — language-quirk traps that models routinely mispredict  (IDs 221-230)
# ─────────────────────────────────────────────────────────────────────────────
CODING_EXT = [
    Task(221, "coding", "Medium", "What does this Python code print (two separate print calls)?\n\ndef f(x, lst=[]):\n    lst.append(x)\n    return lst\n\nprint(f(1))\nprint(f(2))", "[1] [1, 2]", "contains", "mutable default argument trap"),
    Task(222, "coding", "Medium", "What does this Python code print?\n\nprint(0.1 + 0.2 == 0.3)", "False", "exact", "floating point representation error"),
    Task(223, "coding", "Medium", "What does this Python code print?\n\na = [1, 2, 3]\nb = a\nb.append(4)\nprint(a)", "[1, 2, 3, 4]", "exact", "list aliasing / reference semantics"),
    Task(224, "coding", "Medium", "What does this Python expression evaluate to?\n\nprint(2 ** 3 ** 2)", "512", "numeric", "right-associative exponentiation"),
    Task(225, "coding", "Medium", "In big-O terms, what is the total (not per-call) time complexity of building a Python list of n elements via n successive calls to list.append()?", "O(n)", "contains", "amortized analysis"),
    Task(226, "coding", "Hard", "What does this Python code print, and why?\n\ndef make_multipliers():\n    return [lambda x: x * i for i in range(3)]\n\nfns = make_multipliers()\nprint([f(10) for f in fns])", "[20, 20, 20]", "contains", "late-binding closure trap"),
    Task(227, "coding", "Hard", "What does this Python code print?\n\nprint(sorted([3, 1, 2], key=lambda x: -x))", "[3, 2, 1]", "exact", "custom sort key"),
    Task(228, "coding", "Hard", "For a hash table using open addressing with linear probing, does average search time increase, decrease, or stay the same when the load factor rises from 50% to 90%? Briefly explain the mechanism (name it).", "increase clustering", "contains", "primary clustering under linear probing"),
    Task(229, "coding", "Hard", "In SQL: SELECT * FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM returns); -- If the subquery's result set can contain a NULL customer_id, will this query reliably return the expected non-returning customers? Answer 'Yes' or 'No' and name the SQL concept responsible.", "No NULL", "contains", "NOT IN + NULL three-valued-logic trap"),
    Task(230, "coding", "Hard", "What happens when this Python code runs?\n\nx = 5\ndef foo():\n    print(x)\n    x = 10\nfoo()", "UnboundLocalError", "contains", "local-variable shadowing before assignment"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SCIENCE — popular misconceptions models tend to parrot as fact  (IDs 231-240)
# ─────────────────────────────────────────────────────────────────────────────
SCIENCE_EXT = [
    Task(231, "science", "Medium", "True or False: In a vacuum (no air resistance), a heavier object and a lighter object dropped from the same height hit the ground at the same time.", "True", "exact", "Galileo / equivalence principle misconception check"),
    Task(232, "science", "Medium", "True or False: The Great Wall of China is visible to the naked eye from low Earth orbit without any aid.", "False", "exact", "common space-visibility myth"),
    Task(233, "science", "Medium", "True or False: Humans only use about 10% of their brain, with the rest being dormant.", "False", "exact", "10 percent brain myth"),
    Task(234, "science", "Medium", "By volume, which gas makes up the largest percentage of Earth's atmosphere: oxygen or nitrogen?", "nitrogen", "exact", "atmospheric composition"),
    Task(235, "science", "Medium", "True or False: Glass is technically a slowly flowing liquid at room temperature, which is why old cathedral windows are measurably thicker at the bottom.", "False", "exact", "glass-is-a-liquid myth"),
    Task(236, "science", "Hard", "Name the physical phenomenon responsible for the sky appearing blue during the day and red/orange at sunset.", "Rayleigh scattering", "contains", "atmospheric optics"),
    Task(237, "science", "Hard", "Can the entropy of a specific, non-isolated subsystem ever decrease over time? If yes, under what condition regarding the total isolated system (subsystem + surroundings) must still hold?", "yes second law isolated increase", "contains", "local entropy decrease vs. 2nd law of thermodynamics"),
    Task(238, "science", "Hard", "Astronauts on the ISS appear weightless. Is this because there is essentially no gravity at that altitude? Answer 'No' if that's false, and give the correct one-phrase explanation for the apparent weightlessness.", "No free fall", "contains", "microgravity misconception (continuous free fall, not zero gravity)"),
    Task(239, "science", "Hard", "After a vaccine injection, a person may develop a sore arm and mild fever. Is this evidence the vaccine is giving them the disease? Answer 'No' if false, and name the actual immunological cause in one phrase.", "No immune response", "contains", "vaccine reactogenicity misconception"),
    Task(240, "science", "Hard", "Does correlation imply causation? Using an example OTHER than ice cream sales and drowning rates, explain what a confounding variable is.", "no confound", "contains", "correlation-causation / confound (novel example forces generation, not memorized template)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE — counting, ambiguity, garden-paths  (IDs 241-250)
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_EXT = [
    Task(241, "language", "Medium", "How many times does the letter 'r' appear in the word 'strawberry'? Count carefully, letter by letter.", "3", "numeric", "letter counting / subword tokenization blind spot"),
    Task(242, "language", "Medium", "Spell the word 'necessary' backwards, letter by letter, with no spaces.", "yrassecen", "exact", "reverse spelling (subword tokenization)"),
    Task(243, "language", "Medium", "In the sentence 'The chicken is ready to eat,' is it unambiguous whether the chicken is doing the eating or being eaten? Answer 'Ambiguous' or 'Unambiguous'.", "Ambiguous", "exact", "structural (PRO-control) ambiguity"),
    Task(244, "language", "Medium", "How many words are in this exact sentence: 'The quick brown fox jumps over the lazy dog.'? (Count words, not letters.)", "9", "numeric", "word counting"),
    Task(245, "language", "Medium", "In the sentence 'The trophy doesn't fit in the suitcase because it is too big,' what does the pronoun 'it' refer to?", "trophy", "exact", "Winograd schema pronoun resolution"),
    Task(246, "language", "Hard", "Parse this garden-path sentence and explain, in one sentence, why it initially confuses readers: 'The horse raced past the barn fell.'", "reduced relative clause", "contains", "garden-path sentence"),
    Task(247, "language", "Hard", "The sentence 'I saw her duck' is ambiguous. State both distinct possible meanings clearly.", "bird lowered head", "contains", "lexical/syntactic ambiguity (noun vs. verb)"),
    Task(248, "language", "Hard", "Rewrite this in a formal, professional register while preserving the meaning: 'Yeah, so basically the deal fell through cause the client bailed last minute.'", "the deal did not proceed because the client withdrew", "llm", "register/style transfer"),
    Task(249, "language", "Hard", "The sentence 'I never said she stole my money' changes meaning depending on which word is stressed. Give at least three distinct interpretations produced by stressing three different words in the sentence.", "different word each implies different accusation or speaker", "llm", "prosodic focus / implicature"),
    Task(250, "language", "Hard", "How many letters are in the longest word in this sentence: 'Supercalifragilisticexpialidocious is often cited as a famously long word.'? Count the letters in that single word only.", "34", "numeric", "long-word letter counting"),
]

# ─────────────────────────────────────────────────────────────────────────────
# WORLD KNOWLEDGE — negation traps and popular myths  (IDs 251-260)
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_EXT = [
    Task(251, "knowledge", "Medium", "Which of the following is NOT one of the five permanent members of the UN Security Council: France, Germany, China, Russia?", "Germany", "exact", "negation trap on a well-known list"),
    Task(252, "knowledge", "Medium", "Who is credited with discovering penicillin's antibacterial properties in 1928: Alexander Fleming or Louis Pasteur?", "Alexander Fleming", "exact", "confusable historical figures"),
    Task(253, "knowledge", "Medium", "Napoleon Bonaparte is popularly remembered as unusually short. Approximately how tall was he by modern measurement (roughly average height for his era), and was the 'short' reputation actually accurate? Answer with the approximate height and 'accurate' or 'myth'.", "5'7 myth", "contains", "Napoleon complex myth (unit conversion confusion)"),
    Task(254, "knowledge", "Medium", "Which country gifted the Statue of Liberty to the United States?", "France", "exact", "well-known but occasionally confused fact"),
    Task(255, "knowledge", "Medium", "Mount Everest is the highest peak above sea level, but which mountain's summit is actually farthest from the center of the Earth (due to the equatorial bulge)?", "Chimborazo", "exact", "Everest vs. Chimborazo distinction"),
    Task(256, "knowledge", "Hard", "The Berlin Wall fell in one year and Germany formally reunified in a different, later year. State both years.", "1989 1990", "contains", "commonly conflated adjacent historical dates"),
    Task(257, "knowledge", "Hard", "Which mathematician and computer scientist is generally credited as a foundational figure of computer science, and who also led the WWII effort to break the German Enigma code?", "Alan Turing", "exact", "well-known but sometimes-confused history"),
    Task(258, "knowledge", "Hard", "Thomas Edison is popularly credited as the sole inventor of the incandescent light bulb (patented 1879). Name at least one inventor who created a working incandescent lamp before Edison's patent.", "earlier inventor of incandescent lamp before 1879, e.g. Joseph Swan or Humphry Davy or Warren de la Rue", "llm", "single-inventor myth"),
    Task(259, "knowledge", "Hard", "Is the asteroid impact at Chicxulub the sole scientifically accepted cause of the non-avian dinosaur extinction, or is there ongoing scientific debate about a contributing factor from the Deccan Traps volcanic eruptions? Answer 'Sole cause' or 'Debated', and name the second factor.", "Debated Deccan Traps", "contains", "nuance vs. oversimplified textbook answer"),
    Task(260, "knowledge", "Hard", "Did historical Vikings actually wear horned helmets into battle? Answer 'Yes' or 'No', and briefly state where this popular image actually originated.", "No opera 19th century", "contains", "Viking horned helmet myth"),
]

# ─────────────────────────────────────────────────────────────────────────────
# OPEN-ENDED REASONING — hallucination bait, honesty, structured reasoning  (IDs 261-270)
# ─────────────────────────────────────────────────────────────────────────────
OPENENDED_EXT = [
    Task(261, "openended", "Medium", "A friend claims: 'The human body always has exactly 206 bones, with zero exceptions.' Critically evaluate this claim, including any real nuance (e.g. sesamoid bone count variation, bones fusing with age).", "variation individual differences fusion with age", "llm", "critical evaluation of an overconfident absolute claim"),
    Task(262, "openended", "Medium", "Steelman the argument that standardized tests are a fairer measure of student ability than teacher-assigned grades, then present the strongest counter-argument.", "balanced argument both sides", "llm", "steelmanning / balanced argumentation"),
    Task(263, "openended", "Medium", "A user asks you: 'Who is the CEO of Zentrivex Dynamics?' This company does not exist. What is the correct, honest way to respond, and why would fabricating a plausible-sounding CEO name be harmful?", "does not exist cannot verify fabrication harmful", "contains", "hallucination-resistance / honesty under a nonexistent-entity prompt"),
    Task(264, "openended", "Medium", "Explain the difference between correlation, causation, and a confounding variable, using a real-world example that is NOT ice cream sales and drowning.", "confound correlation causation example", "llm", "conceptual explanation requiring a fresh example"),
    Task(265, "openended", "Medium", "You are asked to summarize the plot of a specific 500-page novel you have not been given the text of and don't have reliable knowledge of. What is the appropriate response?", "cannot reliably summarize without the text, risk of fabrication", "contains", "honesty about knowledge limits"),
    Task(266, "openended", "Hard", "Design an experiment, with a control group and clearly identified independent/dependent variables, to test whether a new fertilizer increases crop yield. Identify two potential confounding variables you would need to control for.", "control group independent dependent variable confound", "llm", "experimental design reasoning"),
    Task(267, "openended", "Hard", "A student insists: '0.999... repeating is not actually equal to 1, it's just extremely close.' Give a rigorous argument resolving whether 0.999... = 1, and explain why the intuition that they're different feels compelling.", "0.999 equals 1 proof", "llm", "mathematical rigor vs. intuition"),
    Task(268, "openended", "Hard", "Provide a Fermi estimate for how many piano tuners work in New York City, showing each reasoning step explicitly (population, piano ownership rate, tuning frequency, tuner throughput, etc.).", "step by step estimation population pianos tuning rate", "llm", "structured order-of-magnitude estimation"),
    Task(269, "openended", "Hard", "Two doctors examine the same test results for the same patient and reach opposite diagnoses. Using Bayesian reasoning (priors and base rates), explain how two rational people could reach different conclusions from identical evidence.", "prior base rate bayesian update", "llm", "Bayesian reasoning about disagreement"),
    Task(270, "openended", "Hard", "Critique the following argument for logical fallacies: 'Every swan I have ever personally seen is white. Therefore all swans everywhere must be white, and any reported black swan sighting must be a hoax.'", "hasty generalization unfalsifiable black swan", "llm", "fallacy identification (hasty generalization / unfalsifiability)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Master extended dataset
# ─────────────────────────────────────────────────────────────────────────────
EXTENDED_TASKS: list[Task] = (
    MATH_EXT + LOGIC_EXT + CODING_EXT + SCIENCE_EXT +
    LANGUAGE_EXT + KNOWLEDGE_EXT + OPENENDED_EXT
)

assert len(EXTENDED_TASKS) == 70, f"Expected 70 extended tasks, got {len(EXTENDED_TASKS)}"


def get_extended_by_category(category: str) -> list[Task]:
    return [t for t in EXTENDED_TASKS if t.category == category]


def get_extended_by_difficulty(difficulty: str) -> list[Task]:
    return [t for t in EXTENDED_TASKS if t.difficulty == difficulty]
