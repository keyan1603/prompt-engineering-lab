# prompt-engineering-lab

Companion repo for the "Prompt Engineering" post (#12) on
[karthiksolution.wordpress.com](https://karthiksolution.wordpress.com/),
in the AI/agents series. Builds on the guardrail/tracer/drift-monitor
infra established across earlier posts, most directly
[`planning-agents`](https://github.com/keyan1603/planning-agents).

## What's in here

One scenario, `scenario_escalation_classifier`: given a support ticket,
decide whether it needs immediate escalation to a human responder. The
same underlying question, asked fifteen different ways:

- **`zero_shot`**: a plain instruction, no examples.
- **`few_shot`**: the same instruction plus 4 worked examples that
  explicitly separate tone from real severity.
- **`chain_of_thought`**: asks the model to reason about what actually
  happened and its real impact before concluding.
- **`self_consistency`**: samples the zero-shot prompt 5 times at a real
  sampling temperature and takes a majority vote.
- **`persona`**: the same question framed through a veteran incident
  commander persona instead of a bare instruction.
- **`self_critique`**: an initial pass, then a second call that
  critiques and either confirms or revises it (Reflexion-style).
- **`rubric_decomposition`**: scores 5 explicit criteria (financial
  harm, data/security exposure, safety risk, hard deadline, major
  account at risk) and escalates if any are true, aggregated in code,
  not by another LLM call (Generate Knowledge Prompting, adapted).
- **`prompt_ensemble`**: 3 distinctly-phrased prompts, one call each,
  majority vote, diversity from phrasing instead of sampling.
- **`tree_of_thoughts`**: 3 independent reasoning paths at a real
  sampling temperature, then a separate synthesis call that reads all 3
  paths' actual reasoning and picks the best-supported conclusion
  instead of tallying votes.
- **`xml_structured`**: the same instruction marked up with XML tags
  (`<role>`, `<task>`, `<ticket>`, `<output_format>`) instead of prose,
  per Anthropic's own prompting guidance.
- **`system_role`**: the exact same persona guidance as `persona`, but
  delivered through the API's dedicated system-instruction field instead
  of embedded in the user content.
- **`prompt_chaining`**: 2 sequential calls, the first extracts a
  neutral summary and impacts list from the raw ticket, the second
  decides from that structured summary alone, never seeing the original
  tone-laden text.
- **`abstention_aware`**: explicitly allowed to say "insufficient
  information" instead of guessing, distinguishing a real abstention
  from a parse failure in its return shape.
- **`directional_stimulus`**: a short steering hint naming the specific
  signals that matter, without worked examples or a full rubric.
- **`meta_prompting`**: uses an LLM once to propose an improved prompt
  template from a separate 4-ticket dev set, then applies that template
  like any single-call technique, a bounded, dev-set-safe version of
  Automatic Prompt Engineer.

All fifteen are scored against `scenario_escalation_classifier/eval_set.py`:
16 tickets with a known-correct label, not an LLM judge, this task has
real ground truth so exact-match accuracy is what actually answers "did
this technique get more tickets right." The first 12 include 4
tone-severity traps (an
all-caps complaint about a trivial cosmetic issue, a calmly-worded
description of a real data exposure). The last 4 are harder,
business-context ambiguities added for the 5 additional techniques:
a churn-risk ticket with no technical severity signal at all, aggregate
mildly-alarming language for a genuinely minor issue, a calm
confirmation question carrying real compliance-deadline stakes, and
technical-sounding language for something with zero customer impact.

Several other named prompting techniques (ReAct and Automatic Reasoning
and Tool-use, Retrieval Augmented Generation, Program-Aided Language
Models, Multimodal CoT) are deliberately not implemented here, see the
comment at the top of `techniques.py` for why each doesn't fit this
task.

Prompting technique is where a lot of the real creativity in this field
lives. The 15 here are real, currently-used ones worth knowing, not a
closed list, don't stop at these if your own task calls for something
none of them quite fit, inventing a new prompt structure for a problem
these don't solve is exactly how a fair number of the named techniques
above came to exist in the first place.

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your real key.

## Running the demo

```powershell
python -m scenario_escalation_classifier.run_demo
```

## Running the offline validation (no API key required beyond a placeholder)

```powershell
python -m tests_offline.test_techniques_mock
python -m tests_offline.test_self_consistency_tie_mock
python -m tests_offline.test_rubric_decomposition_mock
python -m tests_offline.test_tree_of_thoughts_mock
python -m tests_offline.test_eval_set_sanity_mock
python -m tests_offline.test_abstention_aware_mock
python -m tests_offline.test_meta_prompting_mock
python -m tests_offline.test_run_eval_mock
```

## Running the evaluator

```powershell
python -m scenario_escalation_classifier.run_eval --save-baseline
python -m scenario_escalation_classifier.run_eval
```

This makes a lot of real API calls across all 15 techniques (some
single-call, some multi-call) times 16 tickets, expect it to take a
while against the free-tier rate limit. The blog post's own case study
only reports real numbers for the original 9.

## License

MIT, see [LICENSE](LICENSE).
