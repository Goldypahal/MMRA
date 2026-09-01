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
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GLM_API_KEY: str = os.getenv("GLM_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "glm": "https://open.bigmodel.cn/api/paas/v4/",
    "cohere": "https://api.cohere.com/v2",
    "openrouter": OPENROUTER_BASE_URL,
}

# ── Model Registry ────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    id: str
    name: str
    short: str
    api_key: str          # model name/identifier for completion request payload
    color: str            # rich markup color
    fallback_api_keys: list[str] = field(default_factory=list)
    provider_api_key: str = ""  # dedicated provider API key
    provider: str = "openrouter" # "openai", "gemini", "glm", "cohere", "openrouter"
    base_url: str = ""     # optional custom base URL


MODELS: Dict[str, ModelConfig] = {
    "A": ModelConfig(
        id="A",
        name="DeepSeek-R1",
        short="DeepSeek",
        api_key="deepseek/deepseek-r1:free",
        color="bold cyan",
        fallback_api_keys=["deepseek/deepseek-r1-distill-qwen-32b:free", "deepseek/deepseek-chat:free"],
        provider="openrouter",
        provider_api_key=OPENROUTER_API_KEY,
        base_url=PROVIDER_BASE_URLS["openrouter"],
    ),
    "B": ModelConfig(
        id="B",
        name="Gemma-4-31B",
        short="Gemma",
        api_key="google/gemma-4-31b-it:free",
        color="bold green",
        fallback_api_keys=["google/gemma-3-27b-it:free", "google/gemma-2-9b-it:free"],
        provider="openrouter",
        provider_api_key=OPENROUTER_API_KEY,
        base_url=PROVIDER_BASE_URLS["openrouter"],
    ),
    "C": ModelConfig(
        id="C",
        name="Qwen3-30B",
        short="Qwen",
        api_key="qwen/qwen3-30b-a3b:free",
        color="bold yellow",
        fallback_api_keys=["qwen/qwen-2.5-72b-instruct:free", "qwen/qwen-2.5-coder-32b-instruct:free"],
        provider="openrouter",
        provider_api_key=OPENROUTER_API_KEY,
        base_url=PROVIDER_BASE_URLS["openrouter"],
    ),
    "D": ModelConfig(
        id="D",
        name="Llama-4-Scout",
        short="Llama",
        api_key="meta-llama/llama-4-scout:free",
        color="bold magenta",
        fallback_api_keys=["meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.1-8b-instruct:free"],
        provider="openrouter",
        provider_api_key=OPENROUTER_API_KEY,
        base_url=PROVIDER_BASE_URLS["openrouter"],
    ),
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

# ── Experiment & Concurrency Parameters ───────────────────────────────────────
TASKS_PER_CATEGORY: int = 20          # 20 × 7 = 140 total
TEMPERATURE: float = 0.0              # deterministic; set to 0.7 for C2 diversity
C2_SAMPLES: int = 3                   # self-consistency sample count
MAX_TOKENS: int = 1024
REQUEST_TIMEOUT: int = 45             # seconds
MAX_RETRIES: int = 2
RETRY_DELAYS: list[float] = [1.0, 3.0] # seconds between retries

MAX_CONCURRENT_REQUESTS: int = 15     # global request semaphore
MAX_CONCURRENT_TASKS: int = 10        # concurrent task workers
CACHE_ENABLED: bool = True

CATEGORY_MAX_TOKENS = {
    "math": 250,
    "logic": 250,
    "coding": 500,
    "science": 200,
    "language": 150,
    "knowledge": 150,
    "openended": 600,
}

# Results storage
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DB: str = str(BASE_DIR / "results" / "results.db")
RESULTS_JSON: str = str(BASE_DIR / "results" / "results.json")
CACHE_FILE: str = str(BASE_DIR / "results" / ".api_cache.json")
