# scenario_escalation_classifier/run_eval.py
# Runs all 15 techniques against the full 16-ticket labeled eval set and
# reports accuracy. This is a lot of real API calls (the multi-call
# techniques alone are 5+4+2+2 calls per ticket on top of 9 single-call
# techniques, times 16 tickets, plus one one-time meta_prompting
# template-revision call), expect this to take a while against the
# free-tier rate limit. The blog post's own case study only reports the
# original 9, run this if you want real numbers for the other 6 too.

import argparse
from pathlib import Path
from common.drift_monitor import save_baseline, check_drift
from scenario_escalation_classifier.classifier import classify
from scenario_escalation_classifier.techniques import (
    zero_shot, few_shot, chain_of_thought, self_consistency,
    persona, self_critique, rubric_decomposition, prompt_ensemble, tree_of_thoughts,
    xml_structured, system_role, prompt_chaining, abstention_aware, directional_stimulus, meta_prompting
)
from scenario_escalation_classifier.eval_set import EVAL_SET
from common.evaluator import TechniqueEvaluator

TECHNIQUES = {
    "zero_shot": (zero_shot, Path("baselines/escalation_zero_shot.json")),
    "few_shot": (few_shot, Path("baselines/escalation_few_shot.json")),
    "chain_of_thought": (chain_of_thought, Path("baselines/escalation_chain_of_thought.json")),
    "self_consistency": (self_consistency, Path("baselines/escalation_self_consistency.json")),
    "persona": (persona, Path("baselines/escalation_persona.json")),
    "self_critique": (self_critique, Path("baselines/escalation_self_critique.json")),
    "rubric_decomposition": (rubric_decomposition, Path("baselines/escalation_rubric_decomposition.json")),
    "prompt_ensemble": (prompt_ensemble, Path("baselines/escalation_prompt_ensemble.json")),
    "tree_of_thoughts": (tree_of_thoughts, Path("baselines/escalation_tree_of_thoughts.json")),
    "xml_structured": (xml_structured, Path("baselines/escalation_xml_structured.json")),
    "system_role": (system_role, Path("baselines/escalation_system_role.json")),
    "prompt_chaining": (prompt_chaining, Path("baselines/escalation_prompt_chaining.json")),
    "abstention_aware": (abstention_aware, Path("baselines/escalation_abstention_aware.json")),
    "directional_stimulus": (directional_stimulus, Path("baselines/escalation_directional_stimulus.json")),
    "meta_prompting": (meta_prompting, Path("baselines/escalation_meta_prompting.json")),
}


def _make_pipeline_fn(name, technique_fn):
    def pipeline_fn(ticket):
        result = classify(name, technique_fn, ticket)
        return {
            "predicted_escalate": result["predicted_escalate"],
            "guardrail_blocks": result["guardrail_blocks"]
        }
    return pipeline_fn


def _run_one(name: str, technique_fn, baseline_path: Path, save: bool):
    print(f"\n=== {name} ===")
    results = TechniqueEvaluator(EVAL_SET).run(_make_pipeline_fn(name, technique_fn))
    print(f"  accuracy: {results['accuracy']} ({results['correct_count']}/{results['total_count']})")
    print(f"  avg_guardrail_blocks: {results['avg_guardrail_blocks']}")
    print(f"  avg_latency_ms: {results['avg_latency_ms']}")
    for t in results["per_ticket"]:
        mark = "OK" if t["correct"] else "WRONG"
        print(f"    [{mark}] expected={t['expected_escalate']} predicted={t['predicted_escalate']}: {t['ticket'][:60]}")

    if save:
        save_baseline(results, baseline_path)
        print(f"  saved baseline -> {baseline_path}")
    else:
        drift = check_drift(results, baseline_path)
        print(f"  drift check: {drift['status']}")
        if drift["status"] != "no_baseline":
            print(f"  deltas: {drift['deltas']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 15 prompting techniques against the labeled escalation eval set.")
    parser.add_argument("--save-baseline", action="store_true",
                         help="Save this run's results as the new baseline instead of checking drift against the existing one.")
    args = parser.parse_args()

    for name, (fn, path) in TECHNIQUES.items():
        _run_one(name, fn, path, args.save_baseline)


if __name__ == "__main__":
    main()
