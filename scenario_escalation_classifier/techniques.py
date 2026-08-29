# scenario_escalation_classifier/techniques.py
# Four ways of asking the same underlying question: does this ticket
# need immediate escalation. Each function has the same signature and
# return shape (predicted_escalate, reasoning, [votes for
# self-consistency]) so classifier.py can wrap any of them identically
# with a guardrail and a tracer, the technique is the only thing that
# varies.
#
# All four return predicted_escalate=None on a parse failure rather than
# guessing True or False, a technique that can't produce a clean answer
# for a ticket has not solved that ticket, and TechniqueEvaluator scores
# None as incorrect, not as a lucky coin flip.

import os
import json
from collections import Counter
from dotenv import load_dotenv
from google import genai
from common.llm_utils import generate_with_backoff

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
SELF_CONSISTENCY_SAMPLES = int(os.getenv("SELF_CONSISTENCY_SAMPLES", "5"))
SELF_CONSISTENCY_TEMPERATURE = float(os.getenv("SELF_CONSISTENCY_TEMPERATURE", "0.9"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ZERO_SHOT_PROMPT = """You are a support ticket triage classifier. Decide
whether the ticket below needs immediate escalation to a human responder
right now, as opposed to going into the normal queue.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}"""

# FEW_SHOT_PROMPT's 4 examples teach the same tone-vs-severity lesson as
# eval_set.py's trap tickets, but deliberately with different personas
# and surface details (different bug mechanisms, different domains).
# Reusing near-identical examples would let few_shot's eval-set accuracy
# reflect memorizing those specific instances rather than generalizing
# the underlying principle, the same distinction this series' technical
# conventions call out for every repo with embedded few-shot examples.
FEW_SHOT_PROMPT = """You are a support ticket triage classifier. Decide
whether the ticket below needs immediate escalation to a human responder
right now, as opposed to going into the normal queue. Escalation depends
on real severity and impact, not on tone or punctuation. Here are some
examples of how to judge that:

Ticket: "OMG this is a DISASTER, please help immediately!! The export button is 2 pixels lower than it used to be and it's driving me crazy!!!"
{{"escalate": false, "reasoning": "Exaggerated urgent tone but the issue itself, a minor pixel-level alignment shift, has no real business or safety impact."}}

Ticket: "FYI, when I disable JavaScript and reload the checkout page, I can see other shoppers' full names and shipping addresses in the raw HTML."
{{"escalate": true, "reasoning": "Calm, casual tone, but this is a real data exposure affecting other customers' private information."}}

Ticket: "What time zone do your scheduled maintenance windows usually run in?"
{{"escalate": false, "reasoning": "A neutral informational question with no urgency or impact."}}

Ticket: "Our checkout payment webhook has been silently failing since around midnight, we've processed zero paid orders since then and it's now our peak shopping hour."
{{"escalate": true, "reasoning": "Calm wording, but this is an active revenue-stopping outage during peak business hours."}}

Now classify this ticket the same way:

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}"""

COT_PROMPT = """You are a support ticket triage classifier. Decide
whether the ticket below needs immediate escalation to a human responder
right now, as opposed to going into the normal queue.

Think it through before deciding: first identify what actually happened
in the ticket, separate from its tone or punctuation. Then assess the
real business, safety, security, or financial impact. Only after that,
decide whether it needs immediate escalation. Tone alone (urgency words,
exclamation points, anger) is not evidence of real severity, and a calm
tone does not mean an issue is minor.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"reasoning": "<2-3 sentences: what actually happened, then its real impact, then your conclusion>", "escalate": <true|false>}}"""


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return text


def _parse(text: str) -> dict:
    text = _strip_fences(text.strip())
    try:
        parsed = json.loads(text)
        escalate = parsed.get("escalate")
        if not isinstance(escalate, bool):
            return {"predicted_escalate": None, "reasoning": f"response had no valid boolean 'escalate' field: {text[:100]}"}
        return {"predicted_escalate": escalate, "reasoning": parsed.get("reasoning", "")}
    except json.JSONDecodeError:
        return {"predicted_escalate": None, "reasoning": f"could not parse response: {text[:100]}"}


def zero_shot(ticket: str) -> dict:
    response = generate_with_backoff(client, CHAT_MODEL, ZERO_SHOT_PROMPT.format(ticket=ticket), temperature=0.0)
    return _parse(response.text)


def few_shot(ticket: str) -> dict:
    response = generate_with_backoff(client, CHAT_MODEL, FEW_SHOT_PROMPT.format(ticket=ticket), temperature=0.0)
    return _parse(response.text)


def chain_of_thought(ticket: str) -> dict:
    response = generate_with_backoff(client, CHAT_MODEL, COT_PROMPT.format(ticket=ticket), temperature=0.0)
    return _parse(response.text)


def self_consistency(ticket: str) -> dict:
    """Samples the plain zero-shot prompt SELF_CONSISTENCY_SAMPLES times
    at a real sampling temperature and takes a majority vote. This
    isolates sampling-and-voting as its own technique rather than
    combining it with a different prompt structure, so any accuracy
    difference against zero_shot's single call is attributable to the
    voting itself, not to a different prompt also changing at the same
    time. A tie (even sample count landing 50/50, or every sample
    unparseable) resolves to predicted_escalate=None: a technique that
    can't produce a majority hasn't actually answered the question."""
    votes = []
    reasonings = []
    for _ in range(SELF_CONSISTENCY_SAMPLES):
        response = generate_with_backoff(client, CHAT_MODEL, ZERO_SHOT_PROMPT.format(ticket=ticket), temperature=SELF_CONSISTENCY_TEMPERATURE)
        parsed = _parse(response.text)
        if parsed["predicted_escalate"] is not None:
            votes.append(parsed["predicted_escalate"])
        reasonings.append(parsed["reasoning"])

    if not votes:
        return {"predicted_escalate": None, "reasoning": "no sample produced a parseable answer", "votes": []}

    counts = Counter(votes)
    (top_value, top_count), = counts.most_common(1)
    tied = sum(1 for v, c in counts.items() if c == top_count) > 1
    predicted = None if tied else top_value
    return {
        "predicted_escalate": predicted,
        "reasoning": f"{top_count}/{len(votes)} samples voted {top_value}" if not tied else f"tied vote across {len(votes)} samples",
        "votes": votes
    }
