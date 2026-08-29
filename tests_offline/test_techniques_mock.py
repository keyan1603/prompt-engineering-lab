# tests_offline/test_techniques_mock.py
# Smoke test: all 4 techniques run end to end through classifier.py's
# guardrail-and-tracer wrapper for one ticket. The mock always returns
# escalate=false, so this only checks the plumbing (a valid boolean
# comes back, guardrails don't block, a trace_id is produced, votes are
# present only for self_consistency), not real classification accuracy.

from tests_offline.mock_helpers import patch_all_llm_calls
from scenario_escalation_classifier.classifier import classify
from scenario_escalation_classifier.techniques import zero_shot, few_shot, chain_of_thought, self_consistency

TICKET = "How do I change my billing email address?"


def _assert_contract(name, result):
    assert result["blocked_at"] is None, f"{name}: unexpectedly blocked at {result['blocked_at']}"
    assert result["predicted_escalate"] is False, f"{name}: expected mocked False, got {result['predicted_escalate']}"
    assert result["trace_id"], f"{name}: expected a non-empty trace_id"
    print(f"  [{name}] OK: predicted_escalate={result['predicted_escalate']}, trace_id={result['trace_id']}")


def main():
    with patch_all_llm_calls():
        for name, fn in (("zero_shot", zero_shot), ("few_shot", few_shot), ("chain_of_thought", chain_of_thought)):
            result = classify(name, fn, TICKET)
            _assert_contract(name, result)
            assert result["votes"] is None, f"{name}: only self_consistency should report votes, got {result['votes']}"

        sc_result = classify("self_consistency", self_consistency, TICKET)
        _assert_contract("self_consistency", sc_result)
        assert sc_result["votes"] == [False] * 5, f"self_consistency: expected 5 unanimous False votes from the mock, got {sc_result['votes']}"

    print("techniques offline validation: all checks passed")


if __name__ == "__main__":
    main()
