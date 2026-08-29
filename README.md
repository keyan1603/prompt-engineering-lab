# prompt-engineering-lab

Companion repo for the "Prompt Engineering" post (#12) on
[karthiksolution.wordpress.com](https://karthiksolution.wordpress.com/),
in the AI/agents series. Builds on the guardrail/tracer/drift-monitor
infra established across earlier posts, most directly
[`planning-agents`](https://github.com/keyan1603/planning-agents).

## What's in here

One scenario, `scenario_escalation_classifier`: given a support ticket,
decide whether it needs immediate escalation to a human responder. The
same underlying question, asked four different ways:

- **`zero_shot`**: a plain instruction, no examples.
- **`few_shot`**: the same instruction plus 4 worked examples that
  explicitly separate tone from real severity.
- **`chain_of_thought`**: asks the model to reason about what actually
  happened and its real impact before concluding.
- **`self_consistency`**: samples the zero-shot prompt 5 times at a real
  sampling temperature and takes a majority vote.

All four are scored against `scenario_escalation_classifier/eval_set.py`,
12 tickets with a known-correct label, not an LLM judge, this task has
real ground truth so exact-match accuracy is what actually answers "did
this technique get more tickets right." 4 of the 12 tickets are
deliberate traps: tone-severity mismatches (an all-caps complaint about
a trivial cosmetic issue, a calmly-worded description of a real data
exposure) that separate a technique that reasons about impact from one
that just pattern-matches urgency language.

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
python -m tests_offline.test_run_eval_mock
```

## Running the evaluator

```powershell
python -m scenario_escalation_classifier.run_eval --save-baseline
python -m scenario_escalation_classifier.run_eval
```

## License

MIT, see [LICENSE](LICENSE).
