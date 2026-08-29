# prompt-engineering-lab

Companion repo for the "Prompt Engineering" post (#12) on
[karthiksolution.wordpress.com](https://karthiksolution.wordpress.com/),
in the AI/agents series. Builds on the guardrail/tracer/drift-monitor
infra established across earlier posts, most directly
[`planning-agents`](https://github.com/keyan1603/planning-agents).

## What's in here

One scenario, `scenario_escalation_classifier`: given a support ticket,
decide whether it needs immediate escalation to a human responder. The
same underlying question, asked nine different ways:

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

All nine are scored against `scenario_escalation_classifier/eval_set.py`:
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

Several other named prompting techniques (ReAct, Retrieval Augmented
Generation, Program-Aided Language Models, Multimodal CoT, Automatic
Prompt Engineer) are deliberately not implemented here, see the
comment at the top of `techniques.py` for why each doesn't fit this
task.

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
python -m tests_offline.test_run_eval_mock
```

## Running the evaluator

```powershell
python -m scenario_escalation_classifier.run_eval --save-baseline
python -m scenario_escalation_classifier.run_eval
```

This makes a lot of real API calls, roughly 20 per ticket summed across
all 9 techniques (some single-call, some multi-call) times 16 tickets,
expect it to take a while against the free-tier rate limit.

## License

MIT, see [LICENSE](LICENSE).
