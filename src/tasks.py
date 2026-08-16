"""
tasks.py — High-difficulty 140-task dataset spanning 7 categories × 3 difficulty levels.
Designed for stress-testing individual model reasoning vs. multi-agent/multi-model frameworks.
Each task has: text, answer, category, difficulty, grader_hint.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    id: int
    category: str
    difficulty: str      # Easy | Medium | Hard
    text: str
    answer: str          # ground truth (exact string or numeric)
    grader_hint: str     # "exact" | "numeric" | "contains" | "llm"
    source: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# MATHEMATICAL REASONING  (20 Hard/Extreme Tasks)
# ─────────────────────────────────────────────────────────────────────────────
MATH_TASKS = [
    # Level: Hard (Formerly Easy)
    Task(1,  "math", "Easy",   "A farmer sells apples. In the first hour, he sells 1/2 of his total apples plus 1/2 of an apple. In the second hour, he sells 1/2 of the remaining apples plus 1/2 of an apple. In the third hour, he sells 1/2 of the remaining apples plus 1/2 of an apple. He now has 0 apples left. If no apples were ever cut or split, how many did he start with?", "7", "exact", "logic arithmetic"),
    Task(2,  "math", "Easy",   "A circle is inscribed in an equilateral triangle. If the area of the circle is 12*pi, what is the perimeter of the triangle?", "36*sqrt(3)", "exact", "geometry"),
    Task(3,  "math", "Easy",   "Find the units digit of the massive exponential sum: (7^2025) + (8^2026) + (9^2027).", "5", "exact", "modular arithmetic"),
    Task(4,  "math", "Easy",   "What is the sum of all distinct real values of x that satisfy the equation: |x^2 - 5x + 5|^((x^2 - 11x + 30)) = 1?", "16", "numeric", "algebra"),
    Task(5,  "math", "Easy",   "An infinite geometric series has a first term a = 5 and converges to a sum S. If the sum of the squares of the terms of this series converges to 25, find the common ratio r.", "0", "numeric", "infinite series"),
    Task(6,  "math", "Easy",   "Solve for the unique positive integer x that satisfies: log_2(x) + log_4(x) + log_16(x) = 7.", "16", "numeric", "logarithms"),
    Task(7,  "math", "Easy",   "How many zeros are at the end of the number 100! (100 factorial)?", "24", "exact", "number theory"),
    
    # Level: Very Hard (Medium)
    Task(8,  "math", "Medium", "In a group of 100 people, 70 like tea, 80 like coffee, and 85 like juice. What is the absolute minimum number of people who must like all three beverages?", "35", "numeric", "set theory"),
    Task(9,  "math", "Medium", "Calculate the value of the limit: lim(x -> 0) [ (1 - cos(3x)) / (x * sin(2x)) ].", "2.25", "numeric", "calculus"),
    Task(10, "math", "Medium", "How many positive integer divisors of 3,600,000 are not perfect squares?", "132", "numeric", "combinatorics"),
    Task(11, "math", "Medium", "Matrix A is a 2x2 matrix with eigenvalues 3 and -1. If B = A^3 - 2A, find the determinant of B.", "-3", "numeric", "linear algebra"),
    Task(12, "math", "Medium", "Find the sum of all integer solutions to the inequality: x^4 - 13x^2 + 36 <= 0.", "0", "numeric", "algebra"),
    Task(13, "math", "Medium", "A bag contains 6 red, 4 blue, and 2 green balls. If three balls are drawn simultaneously at random, what is the probability that they are all distinct colors? Express as a fraction.", "12/55", "exact", "probability"),
    Task(14, "math", "Medium", "Find the shortest distance from the origin (0,0) to the line represented by: 3x - 4y = 25.", "5", "numeric", "coordinate geometry"),
    
    # Level: Extreme (Hard)
    Task(15, "math", "Hard",   "Evaluate the definite integral: Integral from 0 to pi of [ x * sin(x) / (1 + cos^2(x)) ] dx.", "pi^2 / 4", "contains", "calculus"),
    Task(16, "math", "Hard",   "A fair coin is flipped repeatedly until three consecutive heads (HHH) appear. What is the expected number of total flips?", "14", "numeric", "markov chain"),
    Task(17, "math", "Hard",   "What is the smallest positive integer n for which the term n! is divisible by 2026?", "1013", "exact", "number theory"),
    Task(18, "math", "Hard",   "Find the number of ordered triples of positive integers (x, y, z) that satisfy the Diophantine equation: 1/x + 1/y + 1/z = 5/6.", "39", "numeric", "number theory"),
    Task(19, "math", "Hard",   "Let a, b, c be the roots of x^3 - 3x^2 + 2x - 5 = 0. Find the value of a^3 + b^3 + c^3.", "24", "numeric", "symmetric polynomials"),
    Task(20, "math", "Hard",   "Evaluate the infinite sum: Sum from n=1 to infinity of [ n^2 / 3^n ].", "1.5", "numeric", "series"),
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGICAL DEDUCTION  (20 Hard/Extreme Tasks)
# ─────────────────────────────────────────────────────────────────────────────
LOGIC_TASKS = [
    # Level: Hard (Formerly Easy)
    Task(21, "logic", "Easy",   "If all widgets are gadgets, and some gadgets are gizmos, but no gizmos are widgets, is it possible for all widgets to be gizmos? Answer 'Yes' or 'No'.", "No", "exact", "syllogisms"),
    Task(22, "logic", "Easy",   "A clock is losing exactly 6 minutes every hour. It was set correctly at 9:00 AM. What time will the clock show when the actual time is 5:00 PM on the same day? (Format: H:MM)", "4:12", "exact", "time arithmetic"),
    Task(23, "logic", "Easy",   "Three people (Alice, Bob, Carol) stand in a line. Alice says: 'I am standing next to Bob.' Bob says: 'Alice is lying.' Carol says: 'I am standing next to Alice.' If only one of them is telling the truth, who is it?", "Bob", "exact", "liar puzzle"),
    Task(24, "logic", "Easy",   "A set of 5 consecutive integers sums to 150. What is the value of the largest integer in this set?", "32", "numeric", "algebra logic"),
    Task(25, "logic", "Easy",   "If A is true when B is false, and C is true when A is true. If C is false, what can you deduce about B? Answer 'True', 'False', or 'Indeterminate'.", "True", "exact", "boolean deduction"),
    Task(26, "logic", "Easy",   "A bag has 10 black balls and 10 white balls. What is the minimum number of balls you must draw without looking to guarantee getting at least 2 balls of the same color?", "3", "numeric", "pigeonhole principle"),
    Task(27, "logic", "Easy",   "In a certain code, 'APPLE' is written as 'ETTPD'. How would 'PEACH' be written in the same code?", "TIEGL", "exact", "cipher"),
    
    # Level: Very Hard (Medium)
    Task(28, "logic", "Medium", "Five students (A, B, C, D, E) sit in a row. A cannot sit at either end. B must sit exactly two seats to the left of D. C must sit at one of the ends. How many valid seat arrangements are possible?", "2", "numeric", "permutations"),
    Task(29, "logic", "Medium", "You have three boxes: one containing only gold coins, one containing only silver coins, and one containing a mix of both. All three boxes are mislabeled. You draw a single coin from the box labeled 'Mixed' and it is gold. Which box is actually the 'Gold' box? Answer 'Gold', 'Silver', or 'Mixed'.", "Silver", "exact", "deduction"),
    Task(30, "logic", "Medium", "Four cards are on a table: 'D', '3', 'K', '7'. A rule states: 'If a card has a vowel on one side, it must have an even number on the other.' What is the minimum number of cards you must turn over to prove the rule holds, and which cards are they?", "0", "contains", "wason selection"),
    Task(31, "logic", "Medium", "In a village of 100 married couples, every man knows immediately if any other man's wife is unfaithful, but never if his own wife is. A decree states that if a man discovers his wife is unfaithful, he must execute her at midnight. The priest announces: 'At least one wife is unfaithful.' On the 3rd night, executions occur. How many wives were unfaithful?", "3", "numeric", "epistemic game theory"),
    Task(32, "logic", "Medium", "A traveler comes to a fork in the road where one path leads to the City of Truth (where everyone always tells the truth) and the other to the City of Lies (where everyone always lies). A native stands there. What single question can the traveler ask to determine the path to the City of Truth?", "leads to your city", "contains", "logical paradox"),
    Task(33, "logic", "Medium", "There are 8 identical-looking coins, but 1 is counterfeit and weighs slightly less than the others. Using a balance scale, what is the minimum number of weighings needed to guarantee finding the fake coin?", "2", "numeric", "optimization"),
    Task(34, "logic", "Medium", "At 3:15, what is the exact angle between the hour and minute hands of a standard analog clock?", "7.5", "numeric", "geometry"),
    
    # Level: Extreme (Hard)
    Task(35, "logic", "Hard",   "A logic gate network has 4 inputs (A, B, C, D) and output Y. Y is true if and only if a majority of the inputs are true. Write the minimal sum-of-products boolean expression for Y.", "ABC + ABD + ACD + BCD", "contains", "boolean optimization"),
    Task(36, "logic", "Hard",   "Three gods A, B, and C are called, in some order, Truth, False, and Random. Truth always speaks truly, False always speaks falsely, but whether Random speaks truly or falsely is a random decision. You must determine their identities by asking three yes/no questions; each question must be put to exactly one god. The gods understand English, but will answer in their own language, using words 'da' or 'ja' for 'yes' or 'no', and you do not know which means which. What should your first question be to find a god who is NOT Random?", "not random", "contains", "hardest logic puzzle"),
    Task(37, "logic", "Hard",   "You have a 3-gallon jug, a 5-gallon jug, and an unlimited supply of water. The jugs are unmarked. What is the minimum number of pouring steps (fill, empty, or pour) to measure exactly 4 gallons of water into the 5-gallon jug?", "6", "numeric", "state space"),
    Task(38, "logic", "Hard",   "Three logs of length 3m, 4m, and 5m are used to construct a triangular frame. A heavy weight is hung from the top vertex. Which log experiences the greatest compression stress? Answer '3m', '4m', '5m', or 'None'.", "3m", "exact", "structural statics"),
    Task(39, "logic", "Hard",   "If A says B is lying, B says C is lying, and C says both A and B are lying. If everyone is either a liar or a truth-teller, who is telling the truth?", "B", "exact", "truth logic"),
    Task(40, "logic", "Hard",   "A box contains 10 black hats and 10 white hats. Three logicians stand in a queue, each facing forward (so 3rd sees 2nd and 1st, 2nd sees 1st, 1st sees none). A hat is placed on each. 3rd logician says: 'I do not know my hat color.' 2nd says: 'I do not know my hat color.' Under what circumstances can the 1st logician deduce their hat color, and what is it?", "white", "contains", "common knowledge"),
]

# ─────────────────────────────────────────────────────────────────────────────
# CODING & ALGORITHMS  (20 Hard/Extreme Tasks)
# ─────────────────────────────────────────────────────────────────────────────
CODING_TASKS = [
    # Level: Hard (Formerly Easy)
    Task(41, "coding", "Easy",   "What is the average-case time complexity of quickselect when finding the k-th smallest element in an unsorted array of size n?", "O(n)", "exact", "complexity"),
    Task(42, "coding", "Easy",   "Which binary tree traversal method (Pre-order, In-order, Post-order, or Level-order) will print the nodes of a Binary Search Tree (BST) in ascending order?", "In-order", "contains", "traversals"),
    Task(43, "coding", "Easy",   "In Python, what is the output of the following list comprehension: [x for x in range(5) if x % 2 == 0 or x == 3]?", "[0, 2, 3, 4]", "exact", "python list"),
    Task(44, "coding", "Easy",   "What is the space complexity of a depth-first search (DFS) traversal on a graph with V vertices and E edges using an explicit recursion stack in the worst case?", "O(V)", "exact", "graph complexity"),
    Task(45, "coding", "Easy",   "In PostgreSQL, which isolation level completely prevents dirty reads, non-repeatable reads, and phantom reads?", "Serializable", "exact", "database"),
    Task(46, "coding", "Easy",   "What does the term 'GIL' stand for in CPython, and what is its main purpose? (Answer must mention 'Global Interpreter Lock' and thread execution)", "Global Interpreter Lock", "contains", "python internals"),
    Task(47, "coding", "Easy",   "Which HTTP method is designed to be idempotent and is used to update an existing resource or create it if it does not exist?", "PUT", "exact", "http"),
    
    # Level: Very Hard (Medium)
    Task(48, "coding", "Medium", "Write the recurrence relation representing the worst-case time complexity of Quicksort when using a naive partitioning scheme (e.g., picking the first element as pivot).", "T(n) = T(n-1) + O(n)", "contains", "algorithms"),
    Task(49, "coding", "Medium", "What is the maximum number of edges in a directed, strongly connected graph with 8 vertices that contains no self-loops?", "56", "numeric", "graph theory"),
    Task(50, "coding", "Medium", "Explain what a Bloom filter is and identify its two main characteristics regarding false positives and false negatives.", "no false negatives", "contains", "probabilistic data structures"),
    Task(51, "coding", "Medium", "In Rust, what does the borrow checker prevent at compile-time to guarantee memory safety without a garbage collector? (Mention 'data races' or 'dangling pointers')", "data races", "contains", "rust"),
    Task(52, "coding", "Medium", "What design pattern decouples an abstraction from its implementation so that the two can vary independently?", "Bridge", "contains", "gof patterns"),
    Task(53, "coding", "Medium", "What is the output of this Python statement: print(round(2.5) == round(3.5))?", "True", "exact", "floating point quirks"),
    Task(54, "coding", "Medium", "Explain how a B-Tree differs from a binary search tree in terms of node capacity and depth, and why this makes it highly suited for database indexing.", "multiple children", "contains", "data structures"),
    
    # Level: Extreme (Hard)
    Task(55, "coding", "Hard",   "Under Amdahl's Law, if 90% of a program can be parallelized, what is the theoretical maximum speedup factor when running on an infinite number of processors?", "10", "numeric", "concurrency"),
    Task(56, "coding", "Hard",   "Describe the exact time complexity of the Knuth-Morris-Pratt (KMP) string matching algorithm for a text of length N and a pattern of length M.", "O(N + M)", "contains", "string matching"),
    Task(57, "coding", "Hard",   "In distributed systems, what does the PACELC theorem state as an extension of the CAP theorem?", "partition latency consistency", "contains", "distributed database theory"),
    Task(58, "coding", "Hard",   "Explain the core difference between optimistic concurrency control (OCC) and pessimistic concurrency control (PCC) in transaction processing.", "locking vs validation", "contains", "concurrency control"),
    Task(59, "coding", "Hard",   "Identify all four necessary conditions (known as Coffman conditions) that must hold simultaneously for a system deadlock to occur.", "mutual exclusion hold and wait no preemption circular wait", "contains", "operating systems"),
    Task(60, "coding", "Hard",   "In Python, what does the 'yield from' expression do in generators, and how does it facilitate bidirectional communication in coroutines?", "delegates to subgenerator", "contains", "python generators"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SCIENTIFIC KNOWLEDGE  (20 Hard/Extreme Tasks)
# ─────────────────────────────────────────────────────────────────────────────
SCIENCE_TASKS = [
    # Level: Hard (Formerly Easy)
    Task(61, "science", "Easy",   "What is the dominant chemical compound that gives Venus its high reflectivity and thick clouds?", "sulfuric acid", "contains", "astrochemistry"),
    Task(62, "science", "Easy",   "In molecular biology, what specific enzyme catalyzes the synthesis of a complementary RNA molecule from a DNA template?", "RNA polymerase", "contains", "molecular biology"),
    Task(63, "science", "Easy",   "According to the Standard Model of particle physics, which gauge boson is responsible for mediating the strong force?", "gluon", "contains", "particle physics"),
    Task(64, "science", "Easy",   "What type of chemical reaction absorbs thermal energy from its surroundings, resulting in a positive change in enthalpy (+dH)?", "endothermic", "exact", "thermodynamics"),
    Task(65, "science", "Easy",   "What is the primary function of the mitochondria's inner membrane folds (cristae)? (Must mention 'surface area' or 'ATP synthesis')", "ATP synthesis", "contains", "cell biology"),
    Task(66, "science", "Easy",   "What is the IUPAC name for the simplest ketone?", "acetone or propan-2-one", "contains", "organic chemistry"),
    Task(67, "science", "Easy",   "What thermodynamic state function represents the total heat content of a system at constant pressure?", "enthalpy", "exact", "thermodynamics"),
    
    # Level: Very Hard (Medium)
    Task(68, "science", "Medium", "What is the pH of a 1.0 x 10^-8 M solution of hydrochloric acid (HCl) in water at 25°C? (Hint: Consider autoionization of water)", "6.98", "numeric", "analytical chemistry"),
    Task(69, "science", "Medium", "Explain the concept of quantum tunneling and identify one practical electronic component that relies on it.", "tunnel diode", "contains", "quantum electronics"),
    Task(70, "science", "Medium", "Under Einstein's Special Relativity, if a spaceship travels at 0.8c relative to Earth, by what factor (Lorentz factor gamma) are times dilated for its crew?", "1.67", "numeric", "relativity"),
    Task(71, "science", "Medium", "What biological cycle converts atmospheric nitrogen (N2) into ammonia (NH3), and what genus of bacteria is famously responsible for this in root nodules?", "Rhizobium", "contains", "biochemistry"),
    Task(72, "science", "Medium", "What is the primary difference between a transition state and an intermediate in a multi-step chemical reaction mechanism?", "transition state is energy maximum", "contains", "reaction kinetics"),
    Task(73, "science", "Medium", "What are the three components of a single nucleotide subunit in DNA?", "sugar phosphate nitrogenous base", "contains", "biochemistry"),
    Task(74, "science", "Medium", "State the second law of thermodynamics in terms of the entropy of an isolated system.", "entropy increases", "contains", "thermodynamics"),
    
    # Level: Extreme (Hard)
    Task(75, "science", "Hard",   "Describe the chemical mechanism of CRISPR-Cas9 genome editing. What is the role of the guide RNA (gRNA) vs the Cas9 protein?", "gRNA directs Cas9 cleaves", "contains", "molecular genetics"),
    Task(76, "science", "Hard",   "What does the Heisenberg Uncertainty Principle state mathematically for position (x) and momentum (p)?", "h/4pi", "contains", "quantum mechanics"),
    Task(77, "science", "Hard",   "Explain the Bohr effect in biochemistry and how it influences oxygen binding to hemoglobin in active tissues.", "high CO2 lowers pH releases oxygen", "contains", "physiology"),
    Task(78, "science", "Hard",   "What is the exact physical meaning of the 'triple point' of a pure substance on a phase diagram?", "three phases coexist", "contains", "physical chemistry"),
    Task(79, "science", "Hard",   "In astrophysics, what is the Schwarzschild radius formula for a non-rotating black hole of mass M?", "2GM/c^2", "contains", "general relativity"),
    Task(80, "science", "Hard",   "What biochemical pathway represents the primary source of cellular ATP generation under anaerobic conditions in human muscle cells, and what is its final byproduct?", "lactic acid", "contains", "cellular respiration"),
]

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE UNDERSTANDING  (20 Hard/Extreme Tasks)
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_TASKS = [
    # Level: Hard (Formerly Easy)
    Task(81, "language", "Easy",   "What grammatical category does the word 'whom' belong to in modern English, and when is it syntactically preferred over 'who'?", "objective case pronoun", "contains", "syntax"),
    Task(82, "language", "Easy",   "Identify the main rhetorical device used in this sentence: 'Youth is wasted on the young.'", "paradox", "contains", "rhetoric"),
    Task(83, "language", "Easy",   "What is the antonym of the word 'laconic'?", "verbose or loquacious", "contains", "vocabulary"),
    Task(84, "language", "Easy",   "Correct the dangling modifier in this sentence: 'Walking down the street, the trees were beautiful.' (Answer must rewrite to clarify who is walking)", "while I was walking", "contains", "grammar"),
    Task(85, "language", "Easy",   "What is the difference between denotation and connotation in semantics?", "literal vs implied", "contains", "semantics"),
    Task(86, "language", "Easy",   "What linguistic term describes words that are spelled the same but have different meanings and pronunciations (e.g., 'tear' as in crying vs. 'tear' as in ripping)?", "heteronyms", "exact", "morphology"),
    Task(87, "language", "Easy",   "Identify the subject complement in: 'The soup smelled delicious.'", "delicious", "exact", "syntax"),
    
    # Level: Very Hard (Medium)
    Task(88, "language", "Medium", "What syntactic phenomenon is demonstrated by the sentence: 'The old man the boat.'? Explain why it causes temporary parsing failure.", "garden-path sentence", "contains", "psycholinguistics"),
    Task(89, "language", "Medium", "In morphology, what is the difference between inflectional morphemes and derivational morphemes? (Answer must mention 'word class' or 'part of speech')", "derivational changes class", "contains", "morphology"),
    Task(90, "language", "Medium", "Identify the rhetorical and structural device: 'Fair is foul, and foul is fair.'", "chiasmus", "contains", "rhetoric"),
    Task(91, "language", "Medium", "What is anaphoric reference? Give a short example.", "pronoun refers back", "contains", "discourse analysis"),
    Task(92, "language", "Medium", "Explain the difference between a synthetic language (like Latin or Russian) and an analytic language (like English) in terms of syntax and morphology.", "inflections vs word order", "contains", "typology"),
    Task(93, "language", "Medium", "Translate this active sentence into passive voice while preserving semantic nuances: 'Many critics highly praised the novelist's final book.'", "was highly praised by many critics", "contains", "grammar"),
    Task(94, "language", "Medium", "What is the subjunctive mood in English grammar? Give an example of its use.", "if I were", "contains", "grammar"),
    
    # Level: Extreme (Hard)
    Task(95, "language", "Hard",   "What is the difference between Noam Chomsky's concepts of 'competence' and 'performance' in generative grammar?", "underlying knowledge vs actual use", "contains", "linguistics"),
    Task(96, "language", "Hard",   "Explain the Sapir-Whorf hypothesis (linguistic relativity). What is the difference between its strong form and weak form?", "language determines thought vs influences", "contains", "sociolinguistics"),
    Task(97, "language", "Hard",   "Deconstruct the syntactic ambiguity in the sentence: 'I saw the man with the telescope.' Describe both possible meanings.", "telescope to look vs man had it", "contains", "semantics"),
    Task(98, "language", "Hard",   "What is a semantic role? Define the roles of 'Agent' and 'Patient' and identify them in: 'The dog bit the cat.'", "dog is agent cat is patient", "contains", "semantics"),
    Task(99, "language", "Hard",   "Explain the cooperative principle in pragmatics. List Grice's four conversational maxims.", "quality quantity relation manner", "contains", "pragmatics"),
    Task(100,"language", "Hard",   "What is the difference between phonemes and allophones in phonology?", "meaning-distinguishing vs phonetic variants", "contains", "phonology"),
]

# ─────────────────────────────────────────────────────────────────────────────
# WORLD KNOWLEDGE  (20 Hard/Extreme Tasks)
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_TASKS = [
    # Level: Hard (Formerly Easy)
    Task(101,"knowledge","Easy",   "Which 1648 peace treaties ended the Thirty Years' War in the Holy Roman Empire?", "Peace of Westphalia", "contains", "history"),
    Task(102,"knowledge","Easy",   "Which country is completely surrounded by South Africa?", "Lesotho", "exact", "geography"),
    Task(103,"knowledge","Easy",   "Who wrote the seminal economic treatise 'The Wealth of Nations' in 1776?", "Adam Smith", "contains", "economics"),
    Task(104,"knowledge","Easy",   "In what year did the Eastern Roman (Byzantine) Empire fall with the capture of Constantinople?", "1453", "exact", "history"),
    Task(105,"knowledge","Easy",   "What is the deepest known point in the Earth's oceans?", "Mariana Trench or Challenger Deep", "contains", "geography"),
    Task(106,"knowledge","Easy",   "Which artistic movement is characterized by a reaction against Renaissance symmetry, led by Caravaggio and Bernini?", "Baroque", "exact", "art history"),
    Task(107,"knowledge","Easy",   "What is the capital of Canada?", "Ottawa", "exact", "geography"),
    
    # Level: Very Hard (Medium)
    Task(108,"knowledge","Medium", "What was the main purpose of the Bretton Woods Agreement of 1944, and what international monetary standard did it establish?", "pegged to USD and gold", "contains", "economic history"),
    Task(109,"knowledge","Medium", "Describe the significance of the 1215 Magna Carta regarding sovereign power and the rule of law.", "limited king's power", "contains", "constitutional history"),
    Task(110,"knowledge","Medium", "Identify the five permanent members of the United Nations Security Council (UNSC).", "US UK China France Russia", "contains", "international relations"),
    Task(111,"knowledge","Medium", "What economic term describes a situation where inflation is high, economic growth slows down, and unemployment remains steadily high?", "stagflation", "exact", "macroeconomics"),
    Task(112,"knowledge","Medium", "Which German philosopher formulated the concept of the 'Categorical Imperative' as a supreme ethical principle?", "Immanuel Kant", "contains", "philosophy"),
    Task(113,"knowledge","Medium", "What is the Principal-Agent problem in economics and corporate governance?", "misalignment of incentives", "contains", "economics"),
    Task(114,"knowledge","Medium", "What major philosophical position asserts that all knowledge is derived from sensory experience, championed by Locke, Berkeley, and Hume?", "empiricism", "exact", "epistemology"),
    
    # Level: Extreme (Hard)
    Task(115,"knowledge","Hard",   "Explain what Kurt Gödel's First Incompleteness Theorem states regarding formal mathematical systems.", "undecidable statements exist", "contains", "mathematics history"),
    Task(116,"knowledge","Hard",   "What were the core structural causes of the 2007-2008 global financial crisis? (Must mention 'subprime mortgages' or 'securitization')", "subprime mortgage", "contains", "economic history"),
    Task(117,"knowledge","Hard",   "In game theory, what is a Nash Equilibrium, and how does it differ from a dominant strategy?", "no incentive to deviate", "contains", "economics"),
    Task(118,"knowledge","Hard",   "Explain the difference between inductive and deductive reasoning. Which one is primarily used to formulate scientific hypotheses?", "inductive for formulating", "contains", "philosophy of science"),
    Task(119,"knowledge","Hard",   "What was the main trigger and outcome of the Peloponnesian War (431–404 BC)?", "Athens vs Sparta Sparta won", "contains", "ancient history"),
    Task(120,"knowledge","Hard",   "Who designed the architecture of the standard modern computer, describing a system where CPU, memory, and I/O devices share a bus?", "John von Neumann", "contains", "computer history"),
]

# ─────────────────────────────────────────────────────────────────────────────
# OPEN-ENDED REASONING  (20 Custom-Authored High-Reasoning Tasks)
# ─────────────────────────────────────────────────────────────────────────────
OPENENDED_TASKS = [
    # System Design & Architecture (5)
    Task(121,"openended","Hard",  "Design a URL shortening service (like bit.ly) that can handle 100 million write operations and 1 billion read operations per day. Describe your database choices, hashing algorithms, and caching layer design.", "hashing caching sharding high availability", "llm", "system design"),
    Task(122,"openended","Hard",  "Design a highly scalable, distributed rate limiter for a public-facing API. Explain the pros and cons of Token Bucket, Leaking Bucket, and Sliding Window Log algorithms, and how you would maintain state across global servers.", "rate limiting token bucket sliding window synchronization", "llm", "system design"),
    Task(123,"openended","Medium","Design a modern real-time notification service for a massive social network. What technologies would you use for persistent connections (WebSockets vs Server-Sent Events), message queuing, and push delivery?", "websockets sse queuing kafka redis", "llm", "system design"),
    Task(124,"openended","Hard",  "How would you scale an existing single-node relational SQL database to handle 20x its current transactional and query load? Propose a structured, step-by-step roadmap from optimization to distributed partitioning.", "read replicas sharding caching connection pooling indexing", "llm", "system design"),
    Task(125,"openended","Medium","Analyze the architecture trade-offs between a monolithic system and a microservices architecture for a rapidly growing tech startup with 50 developers. Under what conditions is each preferred?", "operational complexity velocity decoupling bounds", "llm", "system design"),
    
    # Argumentative & Geopolitical (5)
    Task(126,"openended","Medium","Critically evaluate the argument: 'Artificial intelligence will net-create more jobs than it destroys over the next 30 years.' Provide structured, evidence-based arguments for both the automation and augmentation hypotheses.", "creative destruction structural unemployment lump of labor", "llm", "argumentation"),
    Task(127,"openended","Medium","Is a Universal Basic Income (UBI) economically viable and socially desirable at a national scale in a highly developed economy? Synthesize the strongest arguments from both fiscal and behavioral perspectives.", "disincentive labor inflation taxation safety net", "llm", "argumentation"),
    Task(128,"openended","Hard",  "Should social media platforms be legally treated as common carriers or publishers regarding user-generated content? Evaluate the legal, ethical, and societal implications under US Section 230 and global standards.", "section 230 editorial control censorship free speech", "llm", "argumentation"),
    Task(129,"openended","Medium","Analyze the risks and benefits of open-sourcing foundational AI models. Formulate a balanced policy recommendation regarding commercial access, model safety, and democratization.", "democratization misuse safety guardrails proliferation", "llm", "argumentation"),
    Task(130,"openended","Hard",  "Debate the philosophical conflict in AI alignment between utilitarian ethics (maximizing positive outcomes) and deontological ethics (adhering to inviolable rules). How can an alignment engineer balance these frameworks?", "utilitarian deontology rules utility alignment safety", "llm", "argumentation"),
    
    # Creative Problem-Solving & Business Strategy (5)
    Task(131,"openended","Medium","You are the CEO of a hyper-growth food delivery startup that is currently burning cash. Propose three highly structured, non-obvious operational interventions to achieve profitability within 12 months without dropping driver wages.", "density batching dynamic pricing margin unit economics", "llm", "problem solving"),
    Task(132,"openended","Hard",  "A major metropolitan city is facing severe, chronic traffic congestion. Propose a multi-disciplinary, tech-enabled solution that goes beyond 'building more public transit' or 'adding road lanes'.", "congestion pricing micromobility active transit routing", "llm", "problem solving"),
    Task(133,"openended","Easy",  "Explain the core mechanical concept of Deep Learning and Neural Networks to a 10-year-old child using a highly intuitive, physical-world analogy.", "analogy training layers weights patterns feedback", "llm", "explanation"),
    Task(134,"openended","Medium","A high-performing software engineering team experiences a 40% drop in velocity and morale after transitioning permanently to remote work. Diagnosed key issues: communication silos, lack of informal knowledge sharing, and meeting fatigue. Propose an actionable framework to reverse this.", "documentation sync async mentorship culture socialization", "llm", "problem solving"),
    Task(135,"openended","Hard",  "Design a comprehensive, semester-long curriculum for teaching 'AI Alignment & Safety' to advanced high school students. What 5 core topics must it cover, and what is a sample project for the final week?", "alignment orthogonality power seeking instrumental convergence governance", "llm", "education"),
    
    # Ethical Dilemmas (5)
    Task(136,"openended","Medium","A self-driving vehicle is in an unavoidable crash scenario: it must either swerve and hit a group of three elderly pedestrians, or continue straight and crash into a concrete barrier, killing its single passenger. Analyze this using three distinct ethical frameworks.", "trolley problem utilitarian deontology virtue ethics", "llm", "ethics"),
    Task(137,"openended","Hard",  "Is it ethical for developers to train massive commercial AI models on copyrighted text and artwork under the doctrine of 'Fair Use' without explicit consent or compensation? Argue from the perspectives of copyright law, utility, and labor rights.", "fair use transformative labor rights derivative work licensing", "llm", "ethics"),
    Task(138,"openended","Hard",  "A pharmaceutical corporation discovers a life-saving drug but sets the price so high that low-income patients cannot afford it, justifying this by the need to fund high-risk future drug research. Evaluate the morality of this pricing structure and propose a fair policy compromise.", "patent pricing access innovation incentive public health", "llm", "ethics"),
    Task(139,"openended","Medium","Should national governments mandate invisible cryptographic watermarking for all AI-generated media to curb deepfakes and misinformation? Evaluate the technical feasibility, censorship concerns, and impact on privacy.", "deepfakes tracking provenance privacy circumvention", "llm", "ethics"),
    Task(140,"openended","Hard",  "Is it ethical for employers to use AI-driven personality and behavioral assessment tools during the hiring process? Analyze concerns around systematic bias, algorithmic transparency, and the reduction of human potential to data points.", "proxy variables bias transparency disparate impact hiring", "llm", "ethics"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Master dataset
# ─────────────────────────────────────────────────────────────────────────────
ALL_TASKS: list[Task] = (
    MATH_TASKS + LOGIC_TASKS + CODING_TASKS + SCIENCE_TASKS +
    LANGUAGE_TASKS + KNOWLEDGE_TASKS + OPENENDED_TASKS
)

assert len(ALL_TASKS) == 140, f"Expected 140 tasks, got {len(ALL_TASKS)}"


def get_tasks_by_category(category: str) -> list[Task]:
    return [t for t in ALL_TASKS if t.category == category]


def get_tasks_by_difficulty(difficulty: str) -> list[Task]:
    return [t for t in ALL_TASKS if t.difficulty == difficulty]


def get_task_subset(n: int = 140) -> list[Task]:
    """
    Return n tasks sampled evenly across all 7 categories.
    If n >= 140, returns ALL_TASKS.
    """
    if n >= 140:
        return ALL_TASKS

    cats = list(dict.fromkeys(t.category for t in ALL_TASKS))
    per_cat = max(1, n // len(cats))
    selected = []
    for cat_id in cats:
        cat_tasks = get_tasks_by_category(cat_id)
        selected.extend(cat_tasks[:per_cat])

    if len(selected) < n:
        remaining = [t for t in ALL_TASKS if t not in selected]
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]
