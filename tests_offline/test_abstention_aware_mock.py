# tests_offline/test_abstention_aware_mock.py
# The standard mock always reports sufficient information, which never
# exercises the actual abstention path. This scripts a response where
# the model explicitly declines to guess, and checks that comes back as
# predicted_escalate=None with abstained=True, not the same
# predicted_escalate=None a parse failure would produce (abstained=None
# in that case). Scoring treats both as incorrect against a ground-truth
# label, but operationally they mean different things, one is a bug,
# the other is the model doing exactly what it was told.

from unittest.mock import patch
from types import SimpleNamespace

from tests_offline.mock_helpers import patch_all_llm_calls
import scenario_escalation_classifier.techniques as techniques_module

TICKET = "Something might be wrong, not sure, hard to tell from here."


def _scripted_response(model, contents, config=None):
    return SimpleNamespace(text=(
        '{"escalate": null, "reasoning": "mock: not enough detail in the ticket to judge real impact", '
        '"insufficient_information": true}'
    ))


def main():
    with patch_all_llm_calls():
        with patch.object(techniques_module.client.models, "generate_content", side_effect=_scripted_response):
            result = techniques_module.abstention_aware(TICKET)

    assert result["predicted_escalate"] is None, f"expected None on an explicit abstention, got {result['predicted_escalate']}"
    assert result["abstained"] is True, f"expected abstained=True for an explicit abstention, got {result['abstained']}"
    print(f"  OK: explicit abstention correctly distinguished from a parse failure, abstained={result['abstained']}")
    print("abstention_aware offline validation: all checks passed")


if __name__ == "__main__":
    main()
