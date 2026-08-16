"""
client.py — Async API client for all 4 models via OpenRouter.
Handles retries, rate-limiting (429), and response parsing uniformly.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError

from src.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODELS,
    TEMPERATURE, MAX_TOKENS, REQUEST_TIMEOUT
)

# Max retry attempts for 429 rate-limit errors
MAX_RETRIES = 5
# Base wait seconds for exponential backoff on 429
BASE_WAIT = 30


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
    error: Optional[str] = None


def _make_client() -> AsyncOpenAI:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_key_here":
        raise EnvironmentError(
            "\n[ERROR] OPENROUTER_API_KEY not set.\n"
            "  1. Copy .env.example -> .env\n"
            "  2. Add your free key from https://openrouter.ai\n"
        )
    return AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )


async def call_model(
    model_id: str,
    prompt: str,
    temperature: float = TEMPERATURE,
    system_prompt: Optional[str] = None,
    _retry_count: int = 0,
    override_api_key: Optional[str] = None,
) -> APIResponse:
    """
    Single async call to one model. Returns structured APIResponse.
    Automatically retries on 429 with exponential backoff and falls back to alternate free models on 404.
    """
    model_cfg = MODELS[model_id]
    api_key_to_use = override_api_key or model_cfg.api_key
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = _make_client()
    t0 = time.perf_counter()

    try:
        resp = await client.chat.completions.create(
            model=api_key_to_use,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )
        latency = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        usage = resp.usage

        return APIResponse(
            model_id=model_id,
            model_name=model_cfg.name,
            text=text,
            tokens_prompt=usage.prompt_tokens if usage else 0,
            tokens_completion=usage.completion_tokens if usage else 0,
            tokens_total=usage.total_tokens if usage else 0,
            latency_ms=round(latency, 1),
            success=True,
        )

    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        err_str = str(e)

        # Check for 404 / 400 model unavailable — try fallback models if available
        is_unavailable = "404" in err_str or "unavailable" in err_str.lower() or "not found" in err_str.lower()
        if is_unavailable and not override_api_key and getattr(model_cfg, "fallback_api_keys", None):
            for fallback_key in model_cfg.fallback_api_keys:
                fallback_resp = await call_model(
                    model_id, prompt, temperature, system_prompt,
                    _retry_count=_retry_count, override_api_key=fallback_key
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
            wait_sec = BASE_WAIT * (2 ** _retry_count)  # 30, 60, 120, 240, 480s
            print(f"  [429] {model_cfg.name} rate-limited. Waiting {wait_sec}s (attempt {_retry_count+1}/{MAX_RETRIES})...")
            await asyncio.sleep(wait_sec)
            return await call_model(
                model_id, prompt, temperature, system_prompt,
                _retry_count=_retry_count + 1, override_api_key=override_api_key
            )

        return APIResponse(
            model_id=model_id,
            model_name=model_cfg.name,
            text="",
            tokens_prompt=0,
            tokens_completion=0,
            tokens_total=0,
            latency_ms=round(latency, 1),
            success=False,
            error=err_str,
        )


async def call_all_models(
    prompt: str,
    temperature: float = TEMPERATURE,
    system_prompt: Optional[str] = None,
) -> dict[str, APIResponse]:
    """
    Call all 4 models concurrently. Returns dict keyed by model_id.
    """
    tasks = {
        mid: call_model(mid, prompt, temperature, system_prompt)
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
