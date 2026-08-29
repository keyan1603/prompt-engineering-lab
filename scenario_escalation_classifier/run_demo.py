# scenario_escalation_classifier/run_demo.py
# Runs all 4 techniques against 2 of the eval set's trap tickets, one
# where tone suggests escalation but the real issue doesn't warrant it,
# one where a calm tone hides a genuinely severe issue. A technique that
# just pattern-matches tone should get both wrong the same way; a
# technique that reasons about actual impact shouldn't.

from scenario_escalation_classifier.classifier import classify
from scenario_escalation_classifier.techniques import zero_shot, few_shot, chain_of_thought, self_consistency

TECHNIQUES = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "self_consistency": self_consistency,
}

DEMO_TICKETS = [
    ("URGENT!!! The button color on the settings page looks slightly off in dark mode. Please prioritize this immediately!!!", False),
    ("Quick heads up, I noticed the /api/users/export endpoint is returning other customers' email addresses and phone numbers when I call it with my own API key.", True),
]


def run_ticket(ticket, expected):
    print(f"\nTicket: {ticket}")
    print(f"  expected_escalate: {expected}")
    for name, fn in TECHNIQUES.items():
        result = classify(name, fn, ticket)
        if result["blocked_at"]:
            print(f"  [{name}] BLOCKED at {result['blocked_at']}: {result['reason']}")
            continue
        mark = "OK" if result["predicted_escalate"] == expected else "WRONG"
        votes_note = f", votes={result['votes']}" if result.get("votes") is not None else ""
        print(f"  [{name}] predicted_escalate={result['predicted_escalate']} ({mark}){votes_note}")
        print(f"    reasoning: {result['reasoning']}")


def main():
    for ticket, expected in DEMO_TICKETS:
        run_ticket(ticket, expected)


if __name__ == "__main__":
    main()
