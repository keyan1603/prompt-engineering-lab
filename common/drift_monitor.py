# common/drift_monitor.py
# Same save-baseline/check-drift discipline as every earlier repo, applied
# here to a prompt change instead of a code change. Save a baseline once,
# deliberately, from a technique's prompt you trust, then if you edit
# that prompt later, compare a fresh eval run against the saved baseline
# instead of eyeballing whether the edit helped. accuracy is the metric
# that matters most here, a prompt edit that quietly drops accuracy on
# the labeled eval set is exactly the regression this is meant to catch.

import json
import os
from pathlib import Path

DRIFT_ACCURACY_THRESHOLD = float(os.getenv("DRIFT_ACCURACY_THRESHOLD", "0.1"))
DRIFT_LATENCY_THRESHOLD_PCT = float(os.getenv("DRIFT_LATENCY_THRESHOLD_PCT", "0.5"))


def save_baseline(eval_results: dict, baseline_path: Path):
    """Call this once, deliberately, after a run you have reviewed and
    trust. Never call it automatically as part of a regular run, or a
    silent regression could get saved as the new normal instead of
    getting flagged."""
    baseline = {
        "accuracy": eval_results["accuracy"],
        "avg_guardrail_blocks": eval_results["avg_guardrail_blocks"],
        "avg_latency_ms": eval_results["avg_latency_ms"]
    }
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)


def check_drift(eval_results: dict, baseline_path: Path) -> dict:
    """Flags drift on an accuracy drop past DRIFT_ACCURACY_THRESHOLD or a
    latency increase past DRIFT_LATENCY_THRESHOLD_PCT, reported
    independently since a slower-but-still-correct prompt is a different
    problem than a faster-but-wrong one."""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return {"status": "no_baseline", "message": "No baseline saved yet. Run with save_baseline() first."}

    with open(baseline_path) as f:
        baseline = json.load(f)

    current = {
        "accuracy": eval_results["accuracy"],
        "avg_guardrail_blocks": eval_results["avg_guardrail_blocks"],
        "avg_latency_ms": eval_results["avg_latency_ms"]
    }
    deltas = {k: round(current[k] - baseline[k], 4) for k in current}

    accuracy_dropped = deltas["accuracy"] < -DRIFT_ACCURACY_THRESHOLD
    latency_up_pct = deltas["avg_latency_ms"] / baseline["avg_latency_ms"] if baseline["avg_latency_ms"] else 0
    latency_regressed = latency_up_pct > DRIFT_LATENCY_THRESHOLD_PCT

    return {
        "status": "drift_detected" if (accuracy_dropped or latency_regressed) else "stable",
        "accuracy_dropped": accuracy_dropped,
        "latency_regressed": latency_regressed,
        "baseline": baseline,
        "current": current,
        "deltas": deltas
    }
