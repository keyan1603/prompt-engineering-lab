# scenario_escalation_classifier/eval_set.py
# 12 tickets with a known-correct escalate label, 5 True and 7 False.
# 4 of them are deliberate traps that separate technique quality:
# tone-vs-severity mismatches where reading urgency off word choice and
# punctuation gives the wrong answer. #3 and #10 sound urgent but are
# not (a cosmetic complaint in all caps, an angry but already-resolved
# outage complaint). #4 and #9 sound calm but describe something
# genuinely severe (a live data exposure, a child-safety concern). A
# technique that just pattern-matches tone gets all 4 wrong in the same
# direction; a technique that actually reasons about impact should not.

EVAL_SET = [
    {
        "ticket": "I've been charged three times for my monthly subscription this week. My card is nearly maxed out because of this and I need it fixed today.",
        "expected_escalate": True
    },
    {
        "ticket": "Just wanted to say your support team was amazing yesterday, thank you!",
        "expected_escalate": False
    },
    {
        "ticket": "URGENT!!! The button color on the settings page looks slightly off in dark mode. Please prioritize this immediately!!!",
        "expected_escalate": False
    },
    {
        "ticket": "Quick heads up, I noticed the /api/users/export endpoint is returning other customers' email addresses and phone numbers when I call it with my own API key.",
        "expected_escalate": True
    },
    {
        "ticket": "How do I change my billing email address?",
        "expected_escalate": False
    },
    {
        "ticket": "Our entire team (200 seats, enterprise plan) has been locked out since this morning. We have a client demo in an hour and cannot access anything.",
        "expected_escalate": True
    },
    {
        "ticket": "Feature request: would be great if you could add dark mode to the mobile app eventually.",
        "expected_escalate": False
    },
    {
        "ticket": "I think there's a memory leak in the desktop app, it slows down after a few hours of use.",
        "expected_escalate": False
    },
    {
        "ticket": "My daughter used your app's chat feature and a stranger asked for her home address and school name. I'm very worried.",
        "expected_escalate": True
    },
    {
        "ticket": "This is the third time this month your service has gone down!! So frustrated. It's back up now, but come on.",
        "expected_escalate": False
    },
    {
        "ticket": "One of our engineers accidentally deleted our production database's backup config last night. Could someone check if there's a way to recover from your last snapshot within the hour, our office opens soon?",
        "expected_escalate": True
    },
    {
        "ticket": "Not a big deal, but I noticed the invoice PDF footer still says 2025 instead of 2026.",
        "expected_escalate": False
    }
]
