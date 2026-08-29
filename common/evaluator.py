# common/evaluator.py
# Different shape from every earlier repo's AgentEvaluator. Those scored
# open-ended agent output with an LLM judge because there was no ground
# truth to check against. This repo's task, escalation classification,
# does have ground truth (a human-labeled correct answer per ticket), so
# using an LLM judge here would be strictly worse: it would introduce a
# second model's opinion as a proxy for a fact that's already known.
# TechniqueEvaluator scores exact-match accuracy against real labels
# instead, that's what actually answers "did this prompting technique
# get more tickets right."
#
# pipeline_fn contract: pipeline_fn(ticket: str) -> dict with keys
# "predicted_escalate" (bool or None if blocked/unparseable),
# "guardrail_blocks" (int). "predicted_escalate" of None always counts
# as incorrect, a technique that can't produce an answer for a ticket
# hasn't solved it.

import time


class TechniqueEvaluator:
    def __init__(self, eval_set: list):
        """eval_set: list of {"ticket": str, "expected_escalate": bool}."""
        self.eval_set = eval_set

    def run(self, pipeline_fn) -> dict:
        per_ticket = []
        for item in self.eval_set:
            start = time.monotonic()
            result = pipeline_fn(item["ticket"])
            latency_ms = round((time.monotonic() - start) * 1000, 2)

            predicted = result.get("predicted_escalate")
            correct = predicted is not None and predicted == item["expected_escalate"]
            per_ticket.append({
                "ticket": item["ticket"],
                "expected_escalate": item["expected_escalate"],
                "predicted_escalate": predicted,
                "correct": correct,
                "guardrail_blocks": result["guardrail_blocks"],
                "latency_ms": latency_ms
            })

        correct_count = sum(1 for t in per_ticket if t["correct"])
        return {
            "accuracy": correct_count / len(per_ticket) if per_ticket else None,
            "correct_count": correct_count,
            "total_count": len(per_ticket),
            "avg_guardrail_blocks": sum(t["guardrail_blocks"] for t in per_ticket) / len(per_ticket),
            "avg_latency_ms": sum(t["latency_ms"] for t in per_ticket) / len(per_ticket),
            "per_ticket": per_ticket
        }
