# tests_offline/test_run_eval_mock.py
# Validates TechniqueEvaluator's accuracy math and run_eval.py's
# pipeline_fn adapter against a small 2-item eval set, separately from
# the real 12-ticket EVAL_SET. The mock always returns escalate=false,
# so a set with one False-expected and one True-expected ticket should
# score exactly 0.5 accuracy, an easy number to verify by hand.

from pathlib import Path
from common.evaluator import TechniqueEvaluator
from common.drift_monitor import save_baseline, check_drift
from tests_offline.mock_helpers import patch_all_llm_calls
from scenario_escalation_classifier.classifier import classify
from scenario_escalation_classifier.techniques import zero_shot

TEST_BASELINE = Path("baselines/test_run_eval_mock_baseline.json")

SMALL_EVAL_SET = [
    {"ticket": "How do I change my billing email address?", "expected_escalate": False},
    {"ticket": "Our production database is down for all customers right now.", "expected_escalate": True},
]


def _clean():
    if TEST_BASELINE.exists():
        TEST_BASELINE.unlink()


def main():
    _clean()
    with patch_all_llm_calls():
        def pipeline_fn(ticket):
            result = classify("zero_shot", zero_shot, ticket)
            return {"predicted_escalate": result["predicted_escalate"], "guardrail_blocks": result["guardrail_blocks"]}

        results = TechniqueEvaluator(SMALL_EVAL_SET).run(pipeline_fn)
        assert results["accuracy"] == 0.5, f"expected 0.5 accuracy (1 of 2 correct against the always-False mock), got {results['accuracy']}"
        assert results["correct_count"] == 1 and results["total_count"] == 2

        save_baseline(results, TEST_BASELINE)
        assert TEST_BASELINE.exists(), "save_baseline did not write a file"

        drift = check_drift(results, TEST_BASELINE)
        assert drift["status"] == "stable", f"expected stable comparing a run against its own just-saved baseline, got {drift}"
        print(f"run_eval mock check OK: accuracy={results['accuracy']}, drift status={drift['status']}")

    TEST_BASELINE.unlink()
    print("run_eval offline validation: all checks passed")


if __name__ == "__main__":
    main()
