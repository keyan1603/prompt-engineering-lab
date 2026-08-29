# scenario_escalation_classifier/run_eval.py
# Runs all 4 techniques against the full 12-ticket labeled eval set and
# reports accuracy, the actual point of this repo: which technique gets
# more tickets right, not which one sounds more sophisticated.

import argparse
from pathlib import Path
from common.drift_monitor import save_baseline, check_drift
from scenario_escalation_classifier.classifier import classify
from scenario_escalation_classifier.techniques import zero_shot, few_shot, chain_of_thought, self_consistency
from scenario_escalation_classifier.eval_set import EVAL_SET
from common.evaluator import TechniqueEvaluator

TECHNIQUES = {
    "zero_shot": (zero_shot, Path("baselines/escalation_zero_shot.json")),
    "few_shot": (few_shot, Path("baselines/escalation_few_shot.json")),
    "chain_of_thought": (chain_of_thought, Path("baselines/escalation_chain_of_thought.json")),
    "self_consistency": (self_consistency, Path("baselines/escalation_self_consistency.json")),
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
    parser = argparse.ArgumentParser(description="Evaluate all 4 prompting techniques against the labeled escalation eval set.")
    parser.add_argument("--save-baseline", action="store_true",
                         help="Save this run's results as the new baseline instead of checking drift against the existing one.")
    args = parser.parse_args()

    for name, (fn, path) in TECHNIQUES.items():
        _run_one(name, fn, path, args.save_baseline)


if __name__ == "__main__":
    main()
