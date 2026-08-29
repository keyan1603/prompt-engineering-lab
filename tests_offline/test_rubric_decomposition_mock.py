# tests_offline/test_rubric_decomposition_mock.py
# The standard mock returns all 5 rubric criteria as false, which never
# exercises the OR-aggregation logic (escalate=True if ANY criterion is
# true). This scripts a response with exactly one criterion true
# (financial_harm) and checks that predicted_escalate comes back True
# even though 4 of 5 criteria are false, and that flags reports exactly
# which criterion triggered it.

from unittest.mock import patch
from types import SimpleNamespace

from tests_offline.mock_helpers import patch_all_llm_calls
import scenario_escalation_classifier.techniques as techniques_module

TICKET = "Some ticket describing active overcharging."


def _scripted_response(model, contents, config=None):
    return SimpleNamespace(text=(
        '{"financial_harm": true, "data_or_security_exposure": false, "safety_risk": false, '
        '"hard_deadline_at_risk": false, "major_account_at_risk": false, '
        '"reasoning": "mock: active overcharging is ongoing financial harm"}'
    ))


def main():
    with patch_all_llm_calls():
        with patch.object(techniques_module.client.models, "generate_content", side_effect=_scripted_response):
            result = techniques_module.rubric_decomposition(TICKET)

    assert result["predicted_escalate"] is True, f"expected True from a single true criterion (OR aggregation), got {result['predicted_escalate']}"
    assert result["flags"]["financial_harm"] is True, f"expected financial_harm flagged true, got {result['flags']}"
    assert result["flags"]["safety_risk"] is False, f"expected safety_risk flagged false, got {result['flags']}"
    print(f"  OK: OR-aggregation correctly escalated on 1 of 5 criteria, flags={result['flags']}")
    print("rubric_decomposition offline validation: all checks passed")


if __name__ == "__main__":
    main()
