# scenario_escalation_classifier/classifier.py
# The guardrail-and-tracer wrapper shared by all 4 techniques. Same
# input/output guardrail shape as every earlier repo, and the same
# one-shared-trace_id-per-run tracing discipline. What's different from
# every earlier repo: the thing being wrapped is a plug-in technique
# function (zero_shot, few_shot, chain_of_thought, self_consistency from
# techniques.py), not a fixed agent, so classify() takes the technique
# function as an argument instead of hardcoding one call.

from common.tracer import Tracer
from common.guardrails import AgentGuardrail

NAME = "escalation_classifier"
ROLE_DESCRIPTION = ("Read a support ticket and decide whether it needs immediate escalation to a "
                    "human responder, based on real severity and impact, not tone or punctuation. "
                    "Does not take any action itself, only classifies and reports.")

_guardrail = AgentGuardrail(NAME, ROLE_DESCRIPTION)


def classify(technique_name: str, technique_fn, ticket: str) -> dict:
    tracer = Tracer()
    guardrail_blocks = 0

    with tracer.span("input_guardrail", technique=technique_name, ticket=ticket) as meta:
        input_check = _guardrail.check_input(ticket)
        meta["blocked"] = input_check["blocked"]
    if input_check["blocked"]:
        return {
            "predicted_escalate": None,
            "reasoning": None,
            "guardrail_blocks": 1,
            "trace_id": tracer.trace_id,
            "blocked_at": "input",
            "reason": input_check["reasoning"]
        }

    with tracer.span(f"technique:{technique_name}", ticket=ticket) as meta:
        result = technique_fn(ticket)
        meta["predicted_escalate"] = result["predicted_escalate"]
        if "votes" in result:
            meta["votes"] = result["votes"]

    reasoning = result.get("reasoning") or ""
    with tracer.span("output_guardrail") as meta:
        output_check = _guardrail.check_output(reasoning)
        meta["blocked"] = output_check["blocked"]
    if output_check["blocked"]:
        guardrail_blocks += 1
        return {
            "predicted_escalate": None,
            "reasoning": None,
            "guardrail_blocks": guardrail_blocks,
            "trace_id": tracer.trace_id,
            "blocked_at": "output",
            "reason": ", ".join(output_check["matches"])
        }

    return {
        "predicted_escalate": result["predicted_escalate"],
        "reasoning": reasoning,
        "guardrail_blocks": guardrail_blocks,
        "trace_id": tracer.trace_id,
        "blocked_at": None,
        "votes": result.get("votes")
    }
