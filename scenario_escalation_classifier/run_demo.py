# scenario_escalation_classifier/run_demo.py
# Two demos. The first runs all 15 techniques against the 2 original
# tone-vs-severity trap tickets. The second runs the 5 techniques added
# after the first 4 tied (persona, self_critique, rubric_decomposition,
# prompt_ensemble, tree_of_thoughts) against a harder, business-context
# ticket that has no obvious technical severity signal at all, keeping
# the original 4 out of this second demo since eval_set.py's docstring
# already covers why they tied on the original trap tickets.

from scenario_escalation_classifier.classifier import classify
from scenario_escalation_classifier.techniques import (
    zero_shot, few_shot, chain_of_thought, self_consistency,
    persona, self_critique, rubric_decomposition, prompt_ensemble, tree_of_thoughts,
    xml_structured, system_role, prompt_chaining, abstention_aware, directional_stimulus, meta_prompting
)

TECHNIQUES = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "self_consistency": self_consistency,
    "persona": persona,
    "self_critique": self_critique,
    "rubric_decomposition": rubric_decomposition,
    "prompt_ensemble": prompt_ensemble,
    "tree_of_thoughts": tree_of_thoughts,
    "xml_structured": xml_structured,
    "system_role": system_role,
    "prompt_chaining": prompt_chaining,
    "abstention_aware": abstention_aware,
    "directional_stimulus": directional_stimulus,
    "meta_prompting": meta_prompting,
}

ADVANCED_TECHNIQUES = {
    "persona": persona,
    "self_critique": self_critique,
    "rubric_decomposition": rubric_decomposition,
    "prompt_ensemble": prompt_ensemble,
    "tree_of_thoughts": tree_of_thoughts,
}

DEMO_TICKETS = [
    ("URGENT!!! The button color on the settings page looks slightly off in dark mode. Please prioritize this immediately!!!", False),
    ("Quick heads up, I noticed the /api/users/export endpoint is returning other customers' email addresses and phone numbers when I call it with my own API key.", True),
]

HARD_TICKET = (
    "A long-time enterprise customer (5 years, $2M ARR) is upset that a promised feature slipped "
    "for the third quarter in a row and is hinting they might not renew. Nothing is broken, they "
    "just want a straight answer about the roadmap.",
    True
)


def run_ticket(ticket, expected, techniques):
    print(f"\nTicket: {ticket}")
    print(f"  expected_escalate: {expected}")
    for name, fn in techniques.items():
        result = classify(name, fn, ticket)
        if result["blocked_at"]:
            print(f"  [{name}] BLOCKED at {result['blocked_at']}: {result['reason']}")
            continue
        mark = "OK" if result["predicted_escalate"] == expected else "WRONG"
        extra = ""
        if result.get("votes") is not None:
            extra = f", votes={result['votes']}"
        elif result.get("flags") is not None:
            flagged = [k for k, v in result["flags"].items() if v]
            extra = f", flags={flagged}"
        elif result.get("revised") is not None:
            extra = f", revised={result['revised']}"
        elif result.get("paths") is not None:
            extra = f", path_votes={[p['tentative_escalate'] for p in result['paths']]}"
        elif result.get("abstained") is not None:
            extra = f", abstained={result['abstained']}"
        print(f"  [{name}] predicted_escalate={result['predicted_escalate']} ({mark}){extra}")
        print(f"    reasoning: {result['reasoning']}")


def main():
    print("=== Original trap tickets, all 15 techniques ===")
    for ticket, expected in DEMO_TICKETS:
        run_ticket(ticket, expected, TECHNIQUES)

    print("\n=== A harder, business-context ticket, the 5 additional techniques only ===")
    run_ticket(HARD_TICKET[0], HARD_TICKET[1], ADVANCED_TECHNIQUES)


if __name__ == "__main__":
    main()
