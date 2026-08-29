# common/tracer.py
# Ported from earlier repos in this series, same shape: one span per
# stage, a shared trace_id tying them together, duration plus arbitrary
# metadata. Here a "stage" is one technique's call against one ticket
# (or, for self-consistency, one sample of several against the same
# ticket), so one trace_id reconstructs everything that happened for a
# single ticket across a single technique run.

import json
import time
import uuid
from pathlib import Path
from contextlib import contextmanager

LOG_PATH = Path("traces/traces.jsonl")


class Tracer:
    """One Tracer per technique-run-against-a-ticket. Every span logged
    through it shares the same trace_id."""

    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        LOG_PATH.parent.mkdir(exist_ok=True)

    @contextmanager
    def span(self, name: str, **metadata):
        start = time.monotonic()
        record = {"trace_id": self.trace_id, "span": name, "metadata": dict(metadata)}
        error = None
        try:
            yield record["metadata"]
        except Exception as e:
            error = str(e)
            raise
        finally:
            record["duration_ms"] = round((time.monotonic() - start) * 1000, 2)
            record["timestamp"] = time.time()
            if error:
                record["error"] = error
            self._write(record)

    def _write(self, record: dict):
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


def read_traces() -> list:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def spans_for_trace(trace_id: str) -> list:
    return sorted(
        (r for r in read_traces() if r["trace_id"] == trace_id),
        key=lambda r: r["timestamp"]
    )
