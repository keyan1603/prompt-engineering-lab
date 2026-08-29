# tests_offline/test_eval_set_sanity_mock.py
# No LLM calls, no mocking needed, just checks EVAL_SET's own shape. This
# exists because a real bug slipped past every other offline test: a
# second, stale `EVAL_SET = [...]` assignment sat below the first,
# correct one in eval_set.py, and Python silently let the second
# assignment win, shadowing the 4 newly-added tickets back down to the
# original 12 with no error anywhere. No test imported EVAL_SET and
# checked its length, so nothing caught it until a real eval run showed
# "(12/12)" instead of the expected "(16/16)". This test exists so that
# specific failure mode, and any future one shaped like it (a ticket
# silently duplicated or dropped), fails loudly and immediately instead
# of silently understating the eval set to whoever reads the console
# output.

from scenario_escalation_classifier.eval_set import EVAL_SET


def main():
    assert len(EVAL_SET) == 16, f"expected 16 tickets, got {len(EVAL_SET)}, check for a duplicate or stale EVAL_SET assignment"

    true_count = sum(1 for t in EVAL_SET if t["expected_escalate"] is True)
    false_count = sum(1 for t in EVAL_SET if t["expected_escalate"] is False)
    assert true_count == 7, f"expected 7 True-labeled tickets, got {true_count}"
    assert false_count == 9, f"expected 9 False-labeled tickets, got {false_count}"
    assert true_count + false_count == len(EVAL_SET), "expected_escalate should only ever be True or False, no other values"

    ticket_texts = [t["ticket"] for t in EVAL_SET]
    assert len(ticket_texts) == len(set(ticket_texts)), "found a duplicate ticket text in EVAL_SET"

    print(f"  OK: EVAL_SET has {len(EVAL_SET)} unique tickets, {true_count} True / {false_count} False")
    print("eval_set sanity offline validation: all checks passed")


if __name__ == "__main__":
    main()
