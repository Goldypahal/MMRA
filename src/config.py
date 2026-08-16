"""
config.py — Central configuration for the research experiment.
Models, conditions, categories all defined here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ── Model Registry ────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    id: str
    name: str
    short: str
    api_key: str          # primary model identifier on OpenRouter
    color: str            # rich markup color
    fallback_api_keys: list[str] = field(default_factory=list)


MODELS: Dict[str, ModelConfig] = {
    "A": ModelConfig("A", "DeepSeek-R1",  "DeepSeek", "deepseek/deepseek-r1:free",  "bold blue",   fallback_api_keys=["nvidia/nemotron-3-super-120b-a12b:free"]),
    "B": ModelConfig("B", "Gemma-4-31B",  "Gemma",    "google/gemma-4-31b-it:free", "bold green",  fallback_api_keys=["google/gemma-4-26b-a4b-it:free"]),
    "C": ModelConfig("C", "Qwen-3-30B",   "Qwen",     "qwen/qwen3-30b-a3b:free",    "bold yellow", fallback_api_keys=["cohere/north-mini-code:free"]),
    "D": ModelConfig("D", "Llama-4-Scout", "Llama",    "meta-llama/llama-4-scout:free","bold red",    fallback_api_keys=["nvidia/nemotron-3.5-lightning:free"]),
}

C1_C2_MODEL: str = "A"

# ── Conditions ────────────────────────────────────────────────────────────────
@dataclass
class Condition:
    id: str
    name: str
    description: str
    api_calls_per_task: int   # approximate


CONDITIONS: Dict[str, Condition] = {
    "C1": Condition("C1", "Single Model",
                    "One model, one call per task. Temperature=0.0.",
                    api_calls_per_task=1),
    "C2": Condition("C2", "Self-Consistency",
                    "Same model, 3 independent calls. Majority vote on answers.",
                    api_calls_per_task=3),
    "C3": Condition("C3", "Parallel Vote",
                    "All 4 models answer independently. Majority vote. No communication.",
                    api_calls_per_task=4),
    "C4": Condition("C4", "Multi-Agent Debate",
                    "All 4 models answer (Round 1), then each sees others' answers and revises (Round 2). Majority vote.",
                    api_calls_per_task=8),
}

# ── Task Categories ───────────────────────────────────────────────────────────
@dataclass
class Category:
    id: str
    name: str
    icon: str
    grader: str           # "exact" | "numeric" | "llm"
    benchmark_source: str


CATEGORIES: Dict[str, Category] = {
    "math":      Category("math",      "Mathematical Reasoning", "∑",   "numeric", "MATH Dataset (Hendrycks 2021) + GSM8K"),
    "logic":     Category("logic",     "Logical Deduction",      "⊕",   "exact",   "LogiQA + BIG-Bench Hard"),
    "coding":    Category("coding",    "Coding & Algorithms",    "</>", "exact",   "HumanEval + MBPP"),
    "science":   Category("science",   "Scientific Knowledge",   "⚗",   "exact",   "SciQ + ARC Challenge"),
    "language":  Category("language",  "Language Understanding", "📝",  "exact",   "SuperGLUE + COPA"),
    "knowledge": Category("knowledge", "World Knowledge",        "🌐",  "exact",   "TriviaQA + Natural Questions"),
    "openended": Category("openended", "Open-Ended Reasoning",   "💡",  "llm",     "Custom-authored (20 tasks)"),
}

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

# ── Experiment Parameters ─────────────────────────────────────────────────────
TASKS_PER_CATEGORY: int = 20          # 20 × 7 = 140 total
TEMPERATURE: float = 0.0              # deterministic; set to 0.7 for C2 diversity
C2_SAMPLES: int = 3                   # self-consistency sample count
MAX_TOKENS: int = 1024
REQUEST_TIMEOUT: int = 60             # seconds
MAX_RETRIES: int = 3
RETRY_DELAY: float = 2.0              # seconds between retries

# Results storage
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DB: str = str(BASE_DIR / "results" / "results.db")
RESULTS_JSON: str = str(BASE_DIR / "results" / "results.json")
