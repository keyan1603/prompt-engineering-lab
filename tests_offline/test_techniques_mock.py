# tests_offline/test_techniques_mock.py
# Smoke test: all 15 techniques run end to end through classifier.py's
# guardrail-and-tracer wrapper for one ticket. The mock always returns
# escalate=false (or all-false criteria/paths for rubric_decomposition
# and tree_of_thoughts), so this only checks the plumbing (a valid
# boolean comes back, guardrails don't block, a trace_id is produced,
# and the right technique-specific extra metadata is present on the
# right technique and absent everywhere else), not real classification
# accuracy.

from tests_offline.mock_helpers import patch_all_llm_calls
from scenario_escalation_classifier.classifier import classify
from scenario_escalation_classifier.techniques import (
    zero_shot, few_shot, chain_of_thought, self_consistency,
    persona, self_critique, rubric_decomposition, prompt_ensemble, tree_of_thoughts,
    xml_structured, system_role, prompt_chaining, abstention_aware, directional_stimulus, meta_prompting
)

TICKET = "How do I change my billing email address?"

SIMPLE_TECHNIQUES = (
    ("zero_shot", zero_shot),
    ("few_shot", few_shot),
    ("chain_of_thought", chain_of_thought),
    ("persona", persona),
    ("xml_structured", xml_structured),
    ("system_role", system_role),
    ("prompt_chaining", prompt_chaining),
    ("directional_stimulus", directional_stimulus),
    ("meta_prompting", meta_prompting),
)


def _assert_contract(name, result):
    assert result["blocked_at"] is None, f"{name}: unexpectedly blocked at {result['blocked_at']}"
    assert result["predicted_escalate"] is False, f"{name}: expected mocked False, got {result['predicted_escalate']}"
    assert result["trace_id"], f"{name}: expected a non-empty trace_id"
    print(f"  [{name}] OK: predicted_escalate={result['predicted_escalate']}, trace_id={result['trace_id']}")


def main():
    with patch_all_llm_calls():
        for name, fn in SIMPLE_TECHNIQUES:
            result = classify(name, fn, TICKET)
            _assert_contract(name, result)
            for key in ("votes", "flags", "revised", "paths", "abstained"):
                assert result[key] is None, f"{name}: expected no {key}, got {result[key]}"

        abstain_result = classify("abstention_aware", abstention_aware, TICKET)
        _assert_contract("abstention_aware", abstain_result)
        assert abstain_result["abstained"] is False, f"abstention_aware: expected abstained=False from the mock (sufficient information), got {abstain_result['abstained']}"

        sc_result = classify("self_consistency", self_consistency, TICKET)
        _assert_contract("self_consistency", sc_result)
        assert sc_result["votes"] == [False] * 5, f"self_consistency: expected 5 unanimous False votes from the mock, got {sc_result['votes']}"

        critique_result = classify("self_critique", self_critique, TICKET)
        _assert_contract("self_critique", critique_result)
        assert critique_result["revised"] is False, f"self_critique: expected revised=False from the mock, got {critique_result['revised']}"

        rubric_result = classify("rubric_decomposition", rubric_decomposition, TICKET)
        _assert_contract("rubric_decomposition", rubric_result)
        assert rubric_result["flags"] == {
            "financial_harm": False, "data_or_security_exposure": False, "safety_risk": False,
            "hard_deadline_at_risk": False, "major_account_at_risk": False
        }, f"rubric_decomposition: expected all-false flags from the mock, got {rubric_result['flags']}"

        ensemble_result = classify("prompt_ensemble", prompt_ensemble, TICKET)
        _assert_contract("prompt_ensemble", ensemble_result)
        assert ensemble_result["votes"] == [False, False, False], f"prompt_ensemble: expected 3 unanimous False votes from the mock, got {ensemble_result['votes']}"

        tot_result = classify("tree_of_thoughts", tree_of_thoughts, TICKET)
        _assert_contract("tree_of_thoughts", tot_result)
        assert tot_result["paths"] is not None and len(tot_result["paths"]) == 3, f"tree_of_thoughts: expected 3 reasoning paths, got {tot_result['paths']}"

    print("techniques offline validation: all checks passed")


if __name__ == "__main__":
    main()
