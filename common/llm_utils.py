# common/llm_utils.py
# Same retry discipline as every earlier repo: a 429 is a structured
# error the SDK already understands, a dropped connection is a raw
# transport failure that surfaces as an httpx/httpcore exception and
# isn't always absorbed by the SDK's own retry logic.
#
# New in this repo: generate_with_backoff takes an optional temperature.
# Every prior repo in this series left temperature at the SDK default
# because nothing needed to control it. This post's whole point requires
# controlling it: zero-shot, few-shot, and chain-of-thought want a low,
# close-to-deterministic temperature so a single call is a fair,
# repeatable comparison point, while self-consistency specifically needs
# real sampling diversity across its N calls, that's the entire
# mechanism a majority vote depends on. Passing temperature=None (the
# default) leaves the SDK's own default behavior unchanged, so this is
# additive, not a behavior change for callers that don't need it.

import time
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from common.rate_limiter import throttle

MAX_RETRIES = 4


def generate_with_backoff(client, model: str, contents: str, temperature: float = None):
    """Drop-in replacement for client.models.generate_content(...).
    Retries on rate limits (429) and on transport-level connection
    failures. Anything else, a real API error unrelated to connectivity
    or rate limiting, is not retried and raises immediately."""
    config = genai_types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            throttle()
            if config is not None:
                return client.models.generate_content(model=model, contents=contents, config=config)
            return client.models.generate_content(model=model, contents=contents)
        except genai_errors.ClientError as e:
            last_error = e
            if e.code == 429:
                wait_seconds = 2 ** attempt * 5
                print(f"  (rate limited, waiting {wait_seconds}s, retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_seconds)
            else:
                raise
        except Exception as e:
            last_error = e
            wait_seconds = 3 * (attempt + 1)
            print(f"  (connection issue: {type(e).__name__}, waiting {wait_seconds}s, retry {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_seconds)
    raise last_error
