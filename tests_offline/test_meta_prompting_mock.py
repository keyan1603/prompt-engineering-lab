# tests_offline/test_meta_prompting_mock.py
# meta_prompting's whole design point is deriving ONE improved template
# through a bounded search against DEV_SET, then applying it
# consistently, not re-deriving it per ticket. This counts how many
# times the module-level client actually gets called while classifying
# 3 different tickets in a row: it should be exactly 4 (1 template-
# revision call, then 1 classification call per ticket), not 6 (which
# would mean the revision call fired every time instead of being
# cached). Also resets the module-level cache first, since it's a
# process-lifetime cache and an earlier test in the same run could have
# already populated it.

from unittest.mock import patch
from types import SimpleNamespace

from tests_offline.mock_helpers import patch_all_llm_calls, _fake_generate_content
import scenario_escalation_classifier.techniques as techniques_module

TICKETS = [
    "How do I change my billing email address?",
    "Feature request: dark mode please.",
    "The app crashed once yesterday, hasn't happened again.",
]


def main():
    techniques_module._meta_prompt_cache["value"] = None
    call_count = {"n": 0}

    def counting_side_effect(model, contents, config=None):
        call_count["n"] += 1
        return _fake_generate_content(model, contents, config)

    with patch_all_llm_calls():
        with patch.object(techniques_module.client.models, "generate_content", side_effect=counting_side_effect):
            for ticket in TICKETS:
                result = techniques_module.meta_prompting(ticket)
                assert result["predicted_escalate"] is False, f"expected mocked False, got {result['predicted_escalate']}"

    assert call_count["n"] == 4, f"expected exactly 4 real calls (1 template revision + 3 classifications) across 3 tickets, got {call_count['n']}, the revision call should be cached, not repeated per ticket"
    print(f"  OK: template revision cached correctly, {call_count['n']} total calls across 3 tickets (1 revision + 3 classify)")
    print("meta_prompting offline validation: all checks passed")


if __name__ == "__main__":
    main()
