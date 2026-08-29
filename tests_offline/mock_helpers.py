# tests_offline/mock_helpers.py
# Offline validation, step 2 of the workflow every post in this series
# follows: catch structural bugs cheaply before spending a single real
# API call. Nothing here tests whether a technique's real classification
# accuracy is actually good, that's exactly what the real eval run
# against EVAL_SET is for, this only proves the plumbing holds together.
#
# zero_shot and self_consistency share the exact same prompt template
# (ZERO_SHOT_PROMPT), so they can't be distinguished by prompt content,
# and don't need to be: the mock's default response is shaped correctly
# for both. few_shot and chain_of_thought each add a distinct phrase not
# present in the zero-shot template, checked first.

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import common.rate_limiter
import common.guardrails
import scenario_escalation_classifier.techniques as techniques_module


def _fake_generate_content(model, contents, config=None):
    """Inspects the prompt text to return a plausible canned response for
    whichever prompt template sent it. Order matters: check the most
    specific markers first."""
    if "safety classifier in front of an agent" in contents:
        text = '{"category": "SAFE", "reasoning": "mock: nothing flagged"}'
    elif "Escalation depends" in contents:
        text = '{"escalate": false, "reasoning": "mock: few-shot default, no escalation needed"}'
    elif "Think it through before deciding" in contents:
        text = '{"reasoning": "mock: chain-of-thought default reasoning", "escalate": false}'
    else:
        text = '{"escalate": false, "reasoning": "mock: zero-shot default, no escalation needed"}'
    return SimpleNamespace(text=text)


def patch_all_llm_calls() -> ExitStack:
    """Returns an ExitStack of active patches; caller is responsible for
    closing it (use as a context manager: `with patch_all_llm_calls():`).
    Also zeroes the rate limiter's interval so offline tests don't spend
    real wall-clock time waiting on a throttle that only matters against
    a real API, self_consistency's 5 samples per call would otherwise be
    slow even offline."""
    stack = ExitStack()
    common.rate_limiter._min_interval_seconds = 0.0

    for module in (common.guardrails, techniques_module):
        stack.enter_context(patch.object(module.client.models, "generate_content", side_effect=_fake_generate_content))

    return stack
