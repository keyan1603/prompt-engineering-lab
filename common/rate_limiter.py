# common/rate_limiter.py
# Same pattern as every earlier repo in this series: one shared throttle
# for every Gemini call, configurable per model's free-tier RPM via .env.
# This repo makes more calls per eval run than most earlier ones,
# self-consistency alone fires SELF_CONSISTENCY_SAMPLES calls for a
# single ticket, across 4 techniques and ~12 tickets that adds up fast.

import os
import time
from dotenv import load_dotenv

load_dotenv()

REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "15"))
_min_interval_seconds = 60.0 / REQUESTS_PER_MINUTE if REQUESTS_PER_MINUTE > 0 else 0.0
_last_call_time = 0.0


def throttle():
    global _last_call_time
    if _min_interval_seconds <= 0:
        return
    elapsed = time.time() - _last_call_time
    remaining = _min_interval_seconds - elapsed
    if remaining > 0:
        time.sleep(remaining)
    _last_call_time = time.time()
