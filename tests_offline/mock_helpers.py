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
# present in the zero-shot template, checked first. persona, self_critique,
# and prompt_ensemble's 3 phrasings all also parse the same
# {"escalate":..., "reasoning":...} shape as zero_shot (self_critique's
# parsing defaults a missing "revised" key to False), so they fall
# through to the same default response too, no dedicated markers needed.
# rubric_decomposition and tree_of_thoughts need their own dispatch
# entries: their parsing expects different keys entirely (5 boolean
# criteria for rubric, "tentative_escalate" instead of "escalate" for a
# tree_of_thoughts reasoning path), a response shaped for the other
# techniques would parse as a structural failure for these two, not
# exercising their real happy path.
#
# xml_structured, system_role, directional_stimulus, and the second
# (decide) call inside prompt_chaining all also parse the same
# {"escalate":..., "reasoning":...} shape, so they fall through to the
# default too. system_role's system_instruction text isn't inspected at
# all here, this mock only looks at `contents`, which is fine since its
# user-content prompt needs the same default shape regardless.
# prompt_chaining's first (extract) call, abstention_aware, and
# meta_prompting's one-time template-revision call each need their own
# dispatch entries for the same reason rubric/tree_of_thoughts do.

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
    elif "scoring a support ticket against a fixed" in contents:
        text = ('{"financial_harm": false, "data_or_security_exposure": false, "safety_risk": false, '
                '"hard_deadline_at_risk": false, "major_account_at_risk": false, "reasoning": "mock: no criteria apply"}')
    elif "synthesizing 3 independent reviewers" in contents:
        text = '{"escalate": false, "reasoning": "mock: synthesis default, no escalation needed"}'
    elif "one of several independent reviewers" in contents:
        text = '{"reasoning": "mock: reasoning path default", "tentative_escalate": false}'
    elif "extract two things" in contents:
        text = '{"summary": "mock: neutral summary of the ticket", "impacts": []}'
    elif "does not give you enough information to decide either way" in contents:
        text = '{"escalate": false, "reasoning": "mock: abstention-aware default, sufficient information present", "insufficient_information": false}'
    elif "You are improving a prompt template" in contents:
        text = '{"revised_template": "Mock revised template. Ticket: {ticket} Respond with only a JSON object, no markdown fences, in this exact shape: {\\"escalate\\": <true|false>, \\"reasoning\\": \\"<one sentence>\\"}"}'
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
