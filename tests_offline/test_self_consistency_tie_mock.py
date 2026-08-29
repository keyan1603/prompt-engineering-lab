# tests_offline/test_self_consistency_tie_mock.py
# The standard mock always votes unanimously, so it never exercises the
# tie-breaking path. This scripts 5 samples as true, false, true, false,
# garbage (unparseable), which produces a genuine 2-2 tie among the 4
# parseable votes. Checks that a tie resolves to predicted_escalate=None
# rather than arbitrarily picking a side.

from unittest.mock import patch
from types import SimpleNamespace

from tests_offline.mock_helpers import patch_all_llm_calls
import scenario_escalation_classifier.techniques as techniques_module

TICKET = "Some ambiguous ticket text."


def _scripted_responses():
    responses = iter([
        SimpleNamespace(text='{"escalate": true, "reasoning": "mock vote 1"}'),
        SimpleNamespace(text='{"escalate": false, "reasoning": "mock vote 2"}'),
        SimpleNamespace(text='{"escalate": true, "reasoning": "mock vote 3"}'),
        SimpleNamespace(text='{"escalate": false, "reasoning": "mock vote 4"}'),
        SimpleNamespace(text='not valid json at all'),
    ])

    def side_effect(model, contents, config=None):
        return next(responses)
    return side_effect


def main():
    with patch_all_llm_calls():
        with patch.object(techniques_module.client.models, "generate_content", side_effect=_scripted_responses()):
            result = techniques_module.self_consistency(TICKET)

    assert result["votes"] == [True, False, True, False], f"expected 4 parseable votes (garbage excluded), got {result['votes']}"
    assert result["predicted_escalate"] is None, f"expected a 2-2 tie to resolve to None, got {result['predicted_escalate']}"
    print(f"  OK: tie correctly resolved to predicted_escalate=None, votes={result['votes']}")
    print("self_consistency tie offline validation: all checks passed")


if __name__ == "__main__":
    main()
