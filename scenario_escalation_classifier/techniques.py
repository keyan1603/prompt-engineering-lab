# scenario_escalation_classifier/techniques.py
# Fifteen ways of asking the same underlying question: does this ticket
# need immediate escalation. Each function has the same signature and
# return shape (predicted_escalate, reasoning, plus optional extra
# metadata like votes/flags/paths/abstained depending on the technique)
# so classifier.py can wrap any of them identically with a guardrail and
# a tracer, the technique is the only thing that varies.
#
# All fifteen return predicted_escalate=None on a parse failure rather
# than guessing True or False, a technique that can't produce a clean
# answer for a ticket has not solved that ticket, and TechniqueEvaluator
# scores None as incorrect, not as a lucky coin flip.
#
# The first 4 (zero_shot, few_shot, chain_of_thought, self_consistency)
# are the original comparison. The next 5 (persona, self_critique,
# rubric_decomposition, prompt_ensemble, tree_of_thoughts) are named
# techniques from https://www.promptingguide.ai/techniques, adapted to
# this task: self_critique is that page's Reflexion, rubric_decomposition
# is Generate Knowledge Prompting. The final 6 (xml_structured,
# system_role, prompt_chaining, abstention_aware, directional_stimulus,
# meta_prompting) round out to 15, drawing on Anthropic's own prompting
# guidance (https://docs.claude.com/en/docs/build-with-claude/prompt-engineering)
# and the rest of promptingguide.ai's list.
#
# Several named techniques genuinely do not apply to this task and are
# not implemented here: ReAct and Automatic Reasoning and Tool-use need
# real tools, this classifier has none. Retrieval Augmented Generation
# needs an external knowledge base, there is nothing to retrieve for a
# single ticket's classification. Program-Aided Language Models needs
# numeric computation, this is a boolean decision. Multimodal CoT needs
# images, tickets here are plain text. Meta Prompting / Automatic Prompt
# Engineer (searching for a better prompt with an LLM) IS implemented,
# as meta_prompting below, but only in a bounded form: it searches
# against a separate DEV_SET and is scored only on EVAL_SET, never on
# the tickets it was tuned against, a full unbounded search loop is
# still out of scope here.

import os
import json
from collections import Counter
from dotenv import load_dotenv
from google import genai
from common.llm_utils import generate_with_backoff
from scenario_escalation_classifier.eval_set import DEV_SET

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


PERSONA_PROMPT = """You are a veteran incident commander with 15 years
of experience triaging support tickets at high-growth SaaS companies.
You have seen thousands of tickets and know that tone and punctuation
are unreliable signals, real severity comes from business, safety,
security, and financial impact. Decide whether the ticket below needs
immediate escalation to a human responder right now, as opposed to
going into the normal queue.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}"""


def persona(ticket: str) -> dict:
    """Same underlying question as zero_shot, framed through an expert
    persona instead of a bare instruction. One call, temperature 0, for
    a fair single-shot comparison against the other single-call
    techniques."""
    response = generate_with_backoff(client, CHAT_MODEL, PERSONA_PROMPT.format(ticket=ticket), temperature=0.0)
    return _parse(response.text)


CRITIQUE_PROMPT = """You are reviewing a colleague's escalation decision
on a support ticket. Reconsider it critically: could the initial
decision be over-influenced by tone or surface language rather than
real severity? Is there hidden business, safety, security, or financial
impact the initial pass might have missed, or reasons the initial pass
over-reacted to alarming-sounding but low-impact language? Revise the
decision only if you find a real reason to, otherwise confirm it.

Ticket: {ticket}

Initial decision: escalate={initial_escalate}
Initial reasoning: {initial_reasoning}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>", "revised": <true|false>}}"""


def self_critique(ticket: str) -> dict:
    """Reflexion-style: an initial zero_shot pass, then a second call
    that critiques that pass and either confirms or revises it. 2 calls
    per ticket. If the initial pass itself can't be parsed, there is
    nothing for the critique to react to, so this returns None rather
    than critiquing a made-up starting point."""
    initial = zero_shot(ticket)
    if initial["predicted_escalate"] is None:
        return {"predicted_escalate": None, "reasoning": "initial zero-shot pass produced no parseable answer", "revised": None}

    response = generate_with_backoff(
        client, CHAT_MODEL,
        CRITIQUE_PROMPT.format(ticket=ticket, initial_escalate=initial["predicted_escalate"], initial_reasoning=initial["reasoning"]),
        temperature=0.0
    )
    text = _strip_fences(response.text.strip())
    try:
        parsed = json.loads(text)
        escalate = parsed.get("escalate")
        if not isinstance(escalate, bool):
            return {"predicted_escalate": None, "reasoning": f"critique response had no valid boolean 'escalate' field: {text[:100]}", "revised": None}
        return {"predicted_escalate": escalate, "reasoning": parsed.get("reasoning", ""), "revised": bool(parsed.get("revised", False))}
    except json.JSONDecodeError:
        return {"predicted_escalate": None, "reasoning": f"could not parse critique response: {text[:100]}", "revised": None}


RUBRIC_CRITERIA = ["financial_harm", "data_or_security_exposure", "safety_risk", "hard_deadline_at_risk", "major_account_at_risk"]

RUBRIC_PROMPT = """You are scoring a support ticket against a fixed
escalation rubric. For each criterion below, answer true or false based
only on what the ticket actually states, not on tone or punctuation.

- financial_harm: does the ticket describe real, ongoing financial harm to the customer or company (e.g. active overcharging, a revenue-stopping outage)?
- data_or_security_exposure: does the ticket describe an actual security vulnerability or exposure of private data, not just a general bug?
- safety_risk: does the ticket describe a real risk to a person's physical safety or wellbeing?
- hard_deadline_at_risk: does the ticket describe a specific, imminent deadline or commitment that will be missed without immediate action?
- major_account_at_risk: does the ticket describe a large or strategically important customer relationship at real risk (e.g. an explicit churn signal, a compliance deadline for a major account)?

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"financial_harm": <true|false>, "data_or_security_exposure": <true|false>, "safety_risk": <true|false>, "hard_deadline_at_risk": <true|false>, "major_account_at_risk": <true|false>, "reasoning": "<one sentence per criterion you marked true, or 'no criteria apply' if all false>"}}"""


def rubric_decomposition(ticket: str) -> dict:
    """Generate Knowledge Prompting, adapted: surface the specific
    criteria that would justify escalation before deciding, instead of
    asking for one holistic judgment. Escalates if the model marks ANY
    criterion true, and that aggregation is a plain OR computed in code
    here, not another LLM judgment call, so the final decision is fully
    traceable to which specific criteria the model actually flagged."""
    response = generate_with_backoff(client, CHAT_MODEL, RUBRIC_PROMPT.format(ticket=ticket), temperature=0.0)
    text = _strip_fences(response.text.strip())
    try:
        parsed = json.loads(text)
        flags = {c: parsed.get(c) for c in RUBRIC_CRITERIA}
        if not all(isinstance(v, bool) for v in flags.values()):
            return {"predicted_escalate": None, "reasoning": f"rubric response missing a valid boolean for one or more criteria: {text[:100]}", "flags": flags}
        escalate = any(flags.values())
        flagged = [c for c, v in flags.items() if v]
        reasoning = parsed.get("reasoning", "") or (f"flagged: {', '.join(flagged)}" if flagged else "no criteria flagged")
        return {"predicted_escalate": escalate, "reasoning": reasoning, "flags": flags}
    except json.JSONDecodeError:
        return {"predicted_escalate": None, "reasoning": f"could not parse rubric response: {text[:100]}", "flags": None}


ENSEMBLE_PROMPTS = [
    """Read the support ticket below and decide: does this need to be
handled by a human right now, or can it wait in the normal queue?

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}""",

    """You triage incoming support tickets. Some require an immediate
human response, most don't. Which is this one?

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}""",

    """Given the ticket text below, would a reasonable support lead pull
a human in immediately, or let it sit in the standard queue?

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}"""
]


def prompt_ensemble(ticket: str) -> dict:
    """3 distinctly-phrased prompts asking the same underlying question,
    one call each at temperature 0, majority vote. Diversity here comes
    from phrasing, not sampling temperature, isolating whether varied
    wording surfaces a different signal than self_consistency's varied
    sampling of the exact same prompt does."""
    votes = []
    for prompt_template in ENSEMBLE_PROMPTS:
        response = generate_with_backoff(client, CHAT_MODEL, prompt_template.format(ticket=ticket), temperature=0.0)
        parsed = _parse(response.text)
        if parsed["predicted_escalate"] is not None:
            votes.append(parsed["predicted_escalate"])

    if not votes:
        return {"predicted_escalate": None, "reasoning": "no ensemble prompt produced a parseable answer", "votes": []}

    counts = Counter(votes)
    (top_value, top_count), = counts.most_common(1)
    tied = sum(1 for v, c in counts.items() if c == top_count) > 1
    predicted = None if tied else top_value
    return {
        "predicted_escalate": predicted,
        "reasoning": f"{top_count}/{len(votes)} prompt phrasings voted {top_value}" if not tied else f"tied vote across {len(votes)} phrasings",
        "votes": votes
    }


TOT_PATH_PROMPT = """You are one of several independent reviewers
reasoning through the same support ticket. Think through what actually
happened, and its real business, safety, security, or financial impact,
separate from tone, and reach a tentative conclusion.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"reasoning": "<2-3 sentences>", "tentative_escalate": <true|false>}}"""

TOT_SYNTHESIS_PROMPT = """You are synthesizing 3 independent reviewers'
reasoning about the same support ticket into one final decision. Do not
simply count votes, read each reviewer's actual reasoning and judge
which argument is best supported by what the ticket actually says.
Reviewers can be wrong, a single well-argued reviewer can outweigh two
weaker ones.

Ticket: {ticket}

Reviewer 1 reasoning: {path1_reasoning}
Reviewer 1 tentative answer: {path1_escalate}

Reviewer 2 reasoning: {path2_reasoning}
Reviewer 2 tentative answer: {path2_escalate}

Reviewer 3 reasoning: {path3_reasoning}
Reviewer 3 tentative answer: {path3_escalate}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence explaining which reviewer's argument you found most convincing and why>"}}"""


def tree_of_thoughts(ticket: str) -> dict:
    """Generates 3 independent reasoning paths at a real sampling
    temperature, exploring different lines of reasoning the way
    self_consistency explores different samples of the same prompt, then
    a separate synthesis call reads all 3 paths' actual reasoning and
    picks or synthesizes the best-supported conclusion. This is the
    mechanism that distinguishes it from prompt_ensemble, which only
    votes on final answers and never looks at the reasoning behind them,
    a single well-argued path can outweigh two weaker ones here instead
    of automatically losing a vote 2 to 1."""
    paths = []
    for _ in range(3):
        response = generate_with_backoff(client, CHAT_MODEL, TOT_PATH_PROMPT.format(ticket=ticket), temperature=0.7)
        text = _strip_fences(response.text.strip())
        try:
            parsed = json.loads(text)
            paths.append({"reasoning": parsed.get("reasoning", ""), "tentative_escalate": parsed.get("tentative_escalate")})
        except json.JSONDecodeError:
            paths.append({"reasoning": f"could not parse path response: {text[:100]}", "tentative_escalate": None})

    if all(p["tentative_escalate"] is None for p in paths):
        return {"predicted_escalate": None, "reasoning": "no reasoning path produced a parseable answer", "paths": paths}

    response = generate_with_backoff(
        client, CHAT_MODEL,
        TOT_SYNTHESIS_PROMPT.format(
            ticket=ticket,
            path1_reasoning=paths[0]["reasoning"], path1_escalate=paths[0]["tentative_escalate"],
            path2_reasoning=paths[1]["reasoning"], path2_escalate=paths[1]["tentative_escalate"],
            path3_reasoning=paths[2]["reasoning"], path3_escalate=paths[2]["tentative_escalate"]
        ),
        temperature=0.0
    )
    parsed = _parse(response.text)
    return {"predicted_escalate": parsed["predicted_escalate"], "reasoning": parsed["reasoning"], "paths": paths}


XML_PROMPT = """<role>You are a support ticket triage classifier.</role>
<task>Decide whether the ticket below needs immediate escalation to a human responder right now, as opposed to going into the normal queue.</task>
<ticket>{ticket}</ticket>
<output_format>Respond with only a JSON object, no markdown fences, in this exact shape: {{"escalate": <true|false>, "reasoning": "<one sentence>"}}</output_format>"""


def xml_structured(ticket: str) -> dict:
    """Anthropic's own prompting guidance recommends XML tags to mark
    the boundaries of a prompt's parts (role, task, input, output
    format) instead of the free-form paragraph structure every other
    prompt in this file uses. Same underlying question as zero_shot,
    delivered with explicit structural markup instead of prose."""
    response = generate_with_backoff(client, CHAT_MODEL, XML_PROMPT.format(ticket=ticket), temperature=0.0)
    return _parse(response.text)


SYSTEM_ROLE_INSTRUCTION = ("You are a veteran incident commander with 15 years of experience triaging support "
                            "tickets at high-growth SaaS companies. Tone and punctuation are unreliable signals, "
                            "real severity comes from business, safety, security, and financial impact.")

SYSTEM_ROLE_PROMPT = """Decide whether the ticket below needs immediate
escalation to a human responder right now, as opposed to going into the
normal queue.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}"""


def system_role(ticket: str) -> dict:
    """The exact same guidance persona embeds directly in the user
    content, delivered instead through the API's own dedicated system-
    instruction channel. A genuinely different mechanism worth comparing
    directly against persona, not just a restyled version of it: does
    where the guidance lives, not just what it says, change behavior."""
    response = generate_with_backoff(
        client, CHAT_MODEL, SYSTEM_ROLE_PROMPT.format(ticket=ticket),
        temperature=0.0, system_instruction=SYSTEM_ROLE_INSTRUCTION
    )
    return _parse(response.text)


EXTRACT_PROMPT = """Read the support ticket below and extract two things:
(1) a neutral, one-sentence factual summary of what actually happened,
with tone and punctuation removed, (2) a list of any concrete business,
safety, security, or financial impacts explicitly stated or clearly
implied.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"summary": "<one neutral sentence>", "impacts": ["<impact 1>", "..."]}}"""

CHAIN_DECIDE_PROMPT = """Given this neutral summary and list of impacts
extracted from a support ticket, decide whether it needs immediate
escalation to a human responder right now, as opposed to going into the
normal queue.

Summary: {summary}
Impacts: {impacts}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}"""


def prompt_chaining(ticket: str) -> dict:
    """2 calls, each with a narrow job. The first extracts a neutral
    summary and an explicit impacts list from the raw ticket, stripping
    tone out at the extraction step rather than asking a single call to
    both read past the tone and decide. The second call never sees the
    original ticket text at all, only the first call's structured
    output, decoupling "understand what happened" from "decide what to
    do about it" into two prompts instead of one prompt doing both
    jobs."""
    extract_response = generate_with_backoff(client, CHAT_MODEL, EXTRACT_PROMPT.format(ticket=ticket), temperature=0.0)
    text = _strip_fences(extract_response.text.strip())
    try:
        extracted = json.loads(text)
        summary = extracted.get("summary", "")
        impacts = extracted.get("impacts", [])
    except json.JSONDecodeError:
        return {"predicted_escalate": None, "reasoning": f"could not parse extraction step: {text[:100]}"}

    decide_response = generate_with_backoff(
        client, CHAT_MODEL,
        CHAIN_DECIDE_PROMPT.format(summary=summary, impacts=", ".join(impacts) if impacts else "(none identified)"),
        temperature=0.0
    )
    result = _parse(decide_response.text)
    result["reasoning"] = f"[summary: {summary}] {result['reasoning']}"
    return result


ABSTAIN_PROMPT = """You are a support ticket triage classifier. Decide
whether the ticket below needs immediate escalation to a human responder
right now, as opposed to going into the normal queue. If the ticket
genuinely does not give you enough information to decide either way,
say so explicitly instead of guessing.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false|null>, "reasoning": "<one sentence>", "insufficient_information": <true|false>}}"""


def abstention_aware(ticket: str) -> dict:
    """Anthropic's prompting guidance explicitly recommends giving the
    model permission to say "I don't know" rather than forcing a guess.
    Distinct from every other technique's predicted_escalate=None: those
    all mean a parse failure, the technique tried and produced something
    unusable. Here, "abstained": True means the model successfully
    followed instructions and explicitly declined to guess, a different
    outcome that happens to look the same in the accuracy table (both
    score as incorrect against a ground-truth label) but means something
    different operationally, one is a bug, the other is the model doing
    exactly what it was told."""
    response = generate_with_backoff(client, CHAT_MODEL, ABSTAIN_PROMPT.format(ticket=ticket), temperature=0.0)
    text = _strip_fences(response.text.strip())
    try:
        parsed = json.loads(text)
        insufficient = bool(parsed.get("insufficient_information", False))
        escalate = parsed.get("escalate")
        if insufficient or not isinstance(escalate, bool):
            return {"predicted_escalate": None, "reasoning": parsed.get("reasoning", ""), "abstained": True}
        return {"predicted_escalate": escalate, "reasoning": parsed.get("reasoning", ""), "abstained": False}
    except json.JSONDecodeError:
        return {"predicted_escalate": None, "reasoning": f"could not parse response: {text[:100]}", "abstained": None}


DSP_PROMPT = """You are a support ticket triage classifier. Decide
whether the ticket below needs immediate escalation to a human responder
right now, as opposed to going into the normal queue.

Hint: look specifically for concrete signals of financial loss, security
or data exposure, safety risk, a hard deadline, or a major account
relationship at risk. The presence or absence of exclamation points and
urgent-sounding words is not one of these signals.

Ticket: {ticket}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"escalate": <true|false>, "reasoning": "<one sentence>"}}"""


def directional_stimulus(ticket: str) -> dict:
    """A short steering hint naming the specific signals to look for,
    without worked examples (few_shot) or a fully decomposed, code-
    aggregated rubric (rubric_decomposition). The lightest-touch cousin
    of both: one call, a holistic judgment, guided by a concise pointer
    toward what actually matters instead of either teaching the
    distinction through examples or forcing it into separately scored
    criteria."""
    response = generate_with_backoff(client, CHAT_MODEL, DSP_PROMPT.format(ticket=ticket), temperature=0.0)
    return _parse(response.text)


META_CRITIQUE_PROMPT = """You are improving a prompt template for a
support ticket escalation classifier. Below is the current prompt
template and a few example tickets with their correct answers. Propose
a revised version of the template that would help a model reach the
correct answer on cases shaped like these, by improving the template's
general instructions, not by embedding these specific examples into the
template itself. The revised template must keep a literal {{ticket}}
placeholder and the same JSON output format.

Current template:
{current_prompt}

Example tickets and correct answers, for improving general instructions only:
{dev_examples}

Respond with only a JSON object, no markdown fences, in this exact shape:
{{"revised_template": "<the full revised prompt template text, containing a literal {{ticket}} placeholder>"}}"""

_meta_prompt_cache = {"value": None}


def _get_meta_optimized_prompt() -> str:
    """Computed once, cached, not once per ticket, the whole point of
    meta-prompting is deriving ONE improved template through a bounded
    search, then applying it consistently, not re-deriving it per call.
    Searches only against DEV_SET, 4 tickets never scored by any
    technique, specifically so the eval numbers this template is later
    judged by can't have leaked into the process that produced it. Falls
    back to a plain-zero-shot-equivalent template, unmodified, if the
    model's proposed revision is missing the required {ticket}
    placeholder or fails to parse, a broken template is worse than no
    improvement.

    The returned template is meant for plain substring replacement
    (`.replace("{ticket}", ticket)`), not `str.format()`. A model-
    proposed template is free text the model wrote to look like JSON,
    "{"escalate": ...}" with single braces, not Python format syntax
    with the double braces this file's own hand-written prompts use to
    escape a literal brace. Calling `.format(ticket=ticket)` on a
    model-provided template raises KeyError the moment it hits the
    JSON example's own single braces, a real bug caught by an offline
    test scripting exactly this case, not a hypothetical."""
    if _meta_prompt_cache["value"] is not None:
        return _meta_prompt_cache["value"]

    dev_examples_text = "\n".join(f"- \"{d['ticket']}\" -> escalate={d['expected_escalate']}" for d in DEV_SET)
    response = generate_with_backoff(
        client, CHAT_MODEL,
        META_CRITIQUE_PROMPT.format(current_prompt=ZERO_SHOT_PROMPT, dev_examples=dev_examples_text),
        temperature=0.0
    )
    text = _strip_fences(response.text.strip())
    # Single-brace fallback: ZERO_SHOT_PROMPT is a str.format() template
    # (double braces escape the literal JSON example), converted here to
    # the same single-brace, replace()-compatible shape a model-provided
    # template would naturally use, so both paths through this function
    # return something meta_prompting can substitute the same way.
    revised = ZERO_SHOT_PROMPT.replace("{{", "{").replace("}}", "}")
    try:
        parsed = json.loads(text)
        candidate = parsed.get("revised_template", "")
        if "{ticket}" in candidate:
            revised = candidate
    except json.JSONDecodeError:
        pass

    _meta_prompt_cache["value"] = revised
    return revised


def meta_prompting(ticket: str) -> dict:
    """Meta Prompting / a bounded Automatic Prompt Engineer: uses an LLM
    to propose a better prompt template, then uses that template like
    any single-call technique. The bound that keeps this honest is
    _get_meta_optimized_prompt's DEV_SET/EVAL_SET separation, see that
    function and DEV_SET's own comment in eval_set.py. Uses a plain
    substring replace, not str.format(), see
    _get_meta_optimized_prompt's docstring for why that distinction is
    a real bug fix, not a style choice."""
    template = _get_meta_optimized_prompt()
    response = generate_with_backoff(client, CHAT_MODEL, template.replace("{ticket}", ticket), temperature=0.0)
    return _parse(response.text)
