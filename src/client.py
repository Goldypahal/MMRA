"""
client.py — Async API client for all 4 models via OpenRouter (with Offline Mock Mode support).
Handles retries, rate-limiting (429), response parsing, and 100% offline execution without API keys.
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError

import hashlib
import json

from src.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENAI_API_KEY, GLM_API_KEY, GEMINI_API_KEY, COHERE_API_KEY,
    MODELS, ModelConfig, TEMPERATURE, MAX_TOKENS, REQUEST_TIMEOUT,
    MAX_CONCURRENT_REQUESTS, CACHE_ENABLED, CACHE_FILE, RETRY_DELAYS
)

# Max retry attempts for 429 rate-limit errors
MAX_RETRIES = len(RETRY_DELAYS)

_GLOBAL_SEMAPHORE: Optional[asyncio.Semaphore] = None

def get_semaphore() -> asyncio.Semaphore:
    global _GLOBAL_SEMAPHORE
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _GLOBAL_SEMAPHORE is None or getattr(_GLOBAL_SEMAPHORE, "_loop", None) != current_loop:
        _GLOBAL_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _GLOBAL_SEMAPHORE

_CACHE_DATA: Optional[dict] = None

def _load_cache() -> dict:
    global _CACHE_DATA
    if _CACHE_DATA is not None:
        return _CACHE_DATA
    if CACHE_ENABLED and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _CACHE_DATA = json.load(f)
                return _CACHE_DATA
        except Exception:
            pass
    _CACHE_DATA = {}
    return _CACHE_DATA

def _save_cache() -> None:
    if not CACHE_ENABLED or _CACHE_DATA is None:
        return
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_CACHE_DATA, f, indent=2)
    except Exception:
        pass

def make_cache_key(model_name: str, prompt: str, temperature: float, max_tokens: int, system_prompt: Optional[str] = None) -> str:
    raw = f"{model_name}:{prompt}:{system_prompt or ''}:{temperature}:{max_tokens}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Telemetry Tracking ────────────────────────────────────────────────────────
TELEMETRY = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "retries": 0,
    "failures": 0,
    "latencies": [],
    "tokens_saved": 0,
}


def get_telemetry_summary() -> dict:
    """Return latency P50/P95, cache hit rate, and execution telemetry summary."""
    import numpy as np
    total = max(1, TELEMETRY["total_requests"])
    hits = TELEMETRY["cache_hits"]
    lats = sorted(TELEMETRY["latencies"]) if TELEMETRY["latencies"] else [0.0]
    p50 = float(np.percentile(lats, 50)) if lats else 0.0
    p95 = float(np.percentile(lats, 95)) if lats else 0.0
    avg_lat = float(np.mean(lats)) if lats else 0.0

    return {
        "total_requests": TELEMETRY["total_requests"],
        "cache_hits": hits,
        "cache_hit_rate": round(hits / total, 4),
        "retries": TELEMETRY["retries"],
        "failures": TELEMETRY["failures"],
        "avg_latency_ms": round(avg_lat, 1),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "tokens_saved": TELEMETRY["tokens_saved"],
    }


@dataclass
class APIResponse:
    model_id: str
    model_name: str
    text: str
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    latency_ms: float
    success: bool
    requested_model: str = ""
    actual_model: str = ""
    fallback_used: bool = False
    fallback_level: int = 0
    is_mock: bool = False
    error: Optional[str] = None


def is_mock_mode() -> bool:
    """Return True ONLY if MMRA_MOCK_MODE environment variable is explicitly set."""
    env_mock = os.getenv("MMRA_MOCK_MODE", "0").lower()
    return env_mock in ("1", "true", "yes")


def _make_client_for_model(model_cfg: ModelConfig, override_model_name: Optional[str] = None) -> tuple[AsyncOpenAI, str]:
    """
    Creates an AsyncOpenAI client targeting the model's direct provider base_url and API key.
    If USE_OMNIROUTE is enabled, routes request to local OmniRoute proxy (http://localhost:20128/v1).
    Returns (AsyncOpenAI_client, model_name_string).
    """
    from src.config import USE_OMNIROUTE, OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY
    model_name = override_model_name or model_cfg.api_key

    if USE_OMNIROUTE:
        base_url = OMNIROUTE_BASE_URL
        api_key = OMNIROUTE_API_KEY
    else:
        base_url = model_cfg.base_url or OPENROUTER_BASE_URL
        api_key = model_cfg.provider_api_key or OPENROUTER_API_KEY or "mock_key"

        if "/" in model_name and not model_cfg.base_url:
            base_url = OPENROUTER_BASE_URL
            api_key = OPENROUTER_API_KEY or api_key

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=REQUEST_TIMEOUT,
    )
    return client, model_name


async def _mock_call_model(
    model_id: str,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> APIResponse:
    """Generate realistic offline mock responses only when MMRA_MOCK_MODE=1 is set."""
    await asyncio.sleep(0.05)  # Simulate network micro-latency
    model_cfg = MODELS[model_id]

    ans = "42"
    if "9.11" in prompt or "9.9" in prompt:
        ans = "9.9"
    elif "2025" in prompt or "units digit" in prompt.lower():
        ans = "0"
    elif "apples" in prompt.lower() or "farmer" in prompt.lower():
        ans = "10"
    elif "geometric series" in prompt.lower() or "sum s" in prompt.lower():
        ans = "15"
    elif "binary tree" in prompt.lower() or "traversal" in prompt.lower():
        ans = "In-order"
    elif "cellulose" in prompt.lower():
        ans = "lacks cellulase enzyme"
    elif "validity" in prompt.lower() or "all roses" in prompt.lower():
        ans = "Invalid"
    elif "whom" in prompt.lower() or "grammatical" in prompt.lower():
        ans = "pronoun"
    elif "quickselect" in prompt.lower() or "time complexity" in prompt.lower():
        ans = "O(n)"
    elif "circle" in prompt.lower() and "inscribed" in prompt.lower():
        ans = "12"
    else:
        m = re.search(r'Answer:\s*([^\n\.]+)', prompt)
        if m:
            ans = m.group(1).strip()

    reasoning = (
        f"[{model_cfg.name} Mock Reasoning]: Evaluating problem constraints step-by-step.\n"
        f"Step 1: Parse input conditions.\n"
        f"Step 2: Apply mathematical/logical deduction.\n"
        f"Step 3: Deduce exact target response '{ans}'.\n\n"
        f"Final answer: {ans}"
    )

    return APIResponse(
        model_id=model_id,
        model_name=model_cfg.name,
        text=reasoning,
        tokens_prompt=120,
        tokens_completion=180,
        tokens_total=300,
        latency_ms=85.0,
        success=True,
        requested_model=model_cfg.api_key,
        actual_model=f"mock-{model_cfg.api_key}",
        fallback_used=False,
        fallback_level=0,
        is_mock=True,
    )


async def call_model(
    model_id: str,
    prompt: str,
    temperature: float = TEMPERATURE,
    system_prompt: Optional[str] = None,
    _retry_count: int = 0,
    override_api_key: Optional[str] = None,
    fallback_level: int = 0,
    max_tokens: int = MAX_TOKENS,
) -> APIResponse:
    """
    Single async call to one model with global rate-limiting and response caching.
    """
    model_cfg = MODELS[model_id]
    if is_mock_mode():
        return await _mock_call_model(model_id, prompt, system_prompt)

    client, model_name = _make_client_for_model(model_cfg, override_model_name=override_api_key)

    # Check Cache
    cache_key = make_cache_key(model_name, prompt, temperature, max_tokens, system_prompt)
    if CACHE_ENABLED:
        cache = _load_cache()
        if cache_key in cache:
            c = cache[cache_key]
            tok = c.get("tokens_total", 200)
            TELEMETRY["total_requests"] += 1
            TELEMETRY["cache_hits"] += 1
            TELEMETRY["tokens_saved"] += tok
            return APIResponse(
                model_id=model_id,
                model_name=model_cfg.name,
                text=c["text"],
                tokens_prompt=c.get("tokens_prompt", 100),
                tokens_completion=c.get("tokens_completion", 100),
                tokens_total=tok,
                latency_ms=c.get("latency_ms", 5.0),
                success=True,
                requested_model=model_cfg.api_key,
                actual_model=model_name,
                fallback_used=(fallback_level > 0),
                fallback_level=fallback_level,
                is_mock=False,
            )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    sem = get_semaphore()
    async with sem:
        try:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency = (time.perf_counter() - t0) * 1000
            text = resp.choices[0].message.content or ""
            usage = resp.usage

            result = APIResponse(
                model_id=model_id,
                model_name=model_cfg.name,
                text=text,
                tokens_prompt=usage.prompt_tokens if usage else 0,
                tokens_completion=usage.completion_tokens if usage else 0,
                tokens_total=usage.total_tokens if usage else 0,
                latency_ms=round(latency, 1),
                success=True,
                requested_model=model_cfg.api_key,
                actual_model=model_name,
                fallback_used=(fallback_level > 0),
                fallback_level=fallback_level,
                is_mock=False,
            )

            TELEMETRY["total_requests"] += 1
            TELEMETRY["cache_misses"] += 1
            TELEMETRY["latencies"].append(result.latency_ms)

            # Save to Cache
            if CACHE_ENABLED:
                cache = _load_cache()
                cache[cache_key] = {
                    "text": text,
                    "tokens_prompt": result.tokens_prompt,
                    "tokens_completion": result.tokens_completion,
                    "tokens_total": result.tokens_total,
                    "latency_ms": result.latency_ms,
                }
                _save_cache()

            return result

        except Exception as e:
            err_str = str(e)

            # Check for 404 / 400 / credit limit / model unavailable — try fallback models if available
            is_unavailable = (
                "404" in err_str
                or "400" in err_str
                or "402" in err_str
                or "unavailable" in err_str.lower()
                or "not found" in err_str.lower()
                or "no endpoint" in err_str.lower()
                or "credit" in err_str.lower()
                or "401" in err_str
            )
            if is_unavailable and getattr(model_cfg, "fallback_api_keys", None):
                fallbacks = model_cfg.fallback_api_keys
                curr_idx = fallbacks.index(override_api_key) if override_api_key in fallbacks else -1
                next_idx = curr_idx + 1
                if next_idx < len(fallbacks):
                    next_fallback = fallbacks[next_idx]
                    print(f"  [Fallback] {model_cfg.name} model {model_name} unavailable. Trying fallback level {next_idx+1}: {next_fallback}...")
                    fallback_resp = await call_model(
                        model_id, prompt, temperature, system_prompt,
                        _retry_count=_retry_count, override_api_key=next_fallback,
                        fallback_level=next_idx + 1, max_tokens=max_tokens
                    )
                    if fallback_resp.success:
                        return fallback_resp

            # Check for 429 rate limit — retry with exponential backoff
            is_rate_limit = (
                "429" in err_str
                or isinstance(e, RateLimitError)
                or (isinstance(e, APIStatusError) and e.status_code == 429)
            )

            if is_rate_limit and _retry_count < MAX_RETRIES:
                wait_sec = RETRY_DELAYS[min(_retry_count, len(RETRY_DELAYS)-1)]
                print(f"  [429] {model_cfg.name} rate-limited. Waiting {wait_sec}s (attempt {_retry_count+1}/{MAX_RETRIES})...")
                await asyncio.sleep(wait_sec)
                return await call_model(
                    model_id, prompt, temperature, system_prompt,
                    _retry_count=_retry_count + 1, override_api_key=override_api_key,
                    fallback_level=fallback_level, max_tokens=max_tokens
                )

            # On real API error when retries/fallbacks exhaust, return explicit failure APIResponse
            return APIResponse(
                model_id=model_id,
                model_name=model_cfg.name,
                text="",
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                latency_ms=0.0,
                success=False,
                requested_model=model_cfg.api_key,
                actual_model=model_name,
                fallback_used=(fallback_level > 0),
                fallback_level=fallback_level,
                is_mock=False,
                error=err_str,
            )


async def call_all_models(
    prompt: str,
    temperature: float = TEMPERATURE,
    system_prompt: Optional[str] = None,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, APIResponse]:
    """Call all 4 models concurrently. Returns dict keyed by model_id."""
    tasks = {
        mid: call_model(mid, prompt, temperature, system_prompt, max_tokens=max_tokens)
        for mid in MODELS
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    output = {}
    for mid, res in zip(tasks.keys(), results):
        if isinstance(res, Exception):
            cfg = MODELS[mid]
            output[mid] = APIResponse(
                model_id=mid, model_name=cfg.name, text="",
                tokens_prompt=0, tokens_completion=0, tokens_total=0,
                latency_ms=0, success=False, error=str(res),
            )
        else:
            output[mid] = res
    return output


async def smoke_test() -> bool:
    """Quick test: ping all 4 models with a trivial prompt."""
    prompt = "Reply with exactly: OK"
    responses = await call_all_models(prompt)
    all_ok = True
    for mid, resp in responses.items():
        cfg = MODELS[mid]
        if resp.success:
            print(f"  OK {cfg.name} — {resp.latency_ms:.0f}ms — {resp.tokens_total} tokens")
        else:
            print(f"  FAIL {cfg.name} — ERROR: {resp.error}")
            all_ok = False
    return all_ok
