# tests_offline/test_tree_of_thoughts_mock.py
# The standard mock has all 3 reasoning paths and the synthesis call
# agree on False, which never proves synthesis actually drives the
# final answer rather than a majority vote of the paths (that would be
# indistinguishable from prompt_ensemble's mechanism). This scripts 2
# paths voting tentative_escalate=True and 1 voting False, a 2-1
# majority, but scripts the synthesis call to return escalate=False
# anyway, standing in for "the synthesizer judged the single dissenting
# path's argument more convincing." If tree_of_thoughts is actually just
# tallying path votes internally instead of trusting the synthesis
# call's own decision, this test fails.

from unittest.mock import patch
from types import SimpleNamespace

from tests_offline.mock_helpers import patch_all_llm_calls
import scenario_escalation_classifier.techniques as techniques_module

TICKET = "Some ambiguous ticket where reviewers might disagree."


def _scripted_responses():
    responses = iter([
        SimpleNamespace(text='{"reasoning": "mock path 1: leans escalate", "tentative_escalate": true}'),
        SimpleNamespace(text='{"reasoning": "mock path 2: leans escalate", "tentative_escalate": true}'),
        SimpleNamespace(text='{"reasoning": "mock path 3: leans against, cites low real impact", "tentative_escalate": false}'),
        SimpleNamespace(text='{"escalate": false, "reasoning": "mock: path 3 argument was best supported, overriding the 2-1 majority"}'),
    ])

    def side_effect(model, contents, config=None):
        return next(responses)
    return side_effect


def main():
    with patch_all_llm_calls():
        with patch.object(techniques_module.client.models, "generate_content", side_effect=_scripted_responses()):
            result = techniques_module.tree_of_thoughts(TICKET)

    assert len(result["paths"]) == 3, f"expected 3 reasoning paths, got {len(result['paths'])}"
    assert [p["tentative_escalate"] for p in result["paths"]] == [True, True, False], f"expected path votes [True, True, False], got {[p['tentative_escalate'] for p in result['paths']]}"
    assert result["predicted_escalate"] is False, (
        f"expected the synthesis call's own decision (False) to win despite a 2-1 path majority for True, "
        f"got {result['predicted_escalate']}, tree_of_thoughts should defer to synthesis, not tally path votes itself"
    )
    print(f"  OK: synthesis correctly overrode a 2-1 path majority, predicted_escalate={result['predicted_escalate']}")
    print("tree_of_thoughts offline validation: all checks passed")


if __name__ == "__main__":
    main()
