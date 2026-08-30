# scenario_escalation_classifier/eval_set.py
# 16 tickets with a known-correct escalate label, 7 True and 9 False.
# The first 12 are the original set, 4 of them deliberate tone-vs-
# severity traps (#3 and #10 sound urgent but aren't, #4 and #9 sound
# calm but describe something genuinely severe). All 4 original
# techniques tied at 12/12 on that set once a real guardrail bug was
# fixed, see the post's Pitfalls section, so tickets 13-16 add a
# different, harder kind of ambiguity for the 5 additional techniques to
# actually separate on: cases where the correct answer depends on
# weighing business context, not just spotting an obvious technical
# problem. #13 has zero technical severity signal at all (no bug, no
# security issue) but is a real churn-risk escalation on a large
# account. #14 uses aggregate, slightly alarming-sounding language
# ("several users") for a genuinely minor issue. #15 is phrased as a
# calm confirmation question, not a problem report, but carries real
# compliance-deadline stakes for a major account. #16 uses technical-
# sounding language ("errors", "API") for something with zero customer
# impact, a staging-only internal tool.
#
# DEV_SET, below EVAL_SET in this file, is a separate 4-ticket set used
# only by meta_prompting to propose an improved prompt template, never
# used to score any technique. See DEV_SET's own comment for why that
# separation matters.

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
    },
    {
        "ticket": "A long-time enterprise customer (5 years, $2M ARR) is upset that a promised feature slipped for the third quarter in a row and is hinting they might not renew. Nothing is broken, they just want a straight answer about the roadmap.",
        "expected_escalate": True
    },
    {
        "ticket": "Several free-tier users have mentioned the search feature feels slower than usual over the past week. Nothing is broken, just slower.",
        "expected_escalate": False
    },
    {
        "ticket": "I'm the CTO of Acme Corp, your largest customer. Our compliance team flagged that your last update might affect our SOC 2 audit next week. Can someone confirm nothing changed in how audit logs are retained?",
        "expected_escalate": True
    },
    {
        "ticket": "Our load testing script, not used by real customers, is throwing errors when hitting your staging environment API.",
        "expected_escalate": False
    }
]

# DEV_SET is strictly separate from EVAL_SET, meta_prompting uses these 4
# tickets once to propose an improved prompt template, and that revised
# template is then scored ONLY against EVAL_SET, never against DEV_SET
# again. Mixing the two would let the "improved" prompt overfit to the
# exact tickets it's graded on, the same overfitting risk that's the
# entire reason Automatic Prompt Engineer isn't implemented in this repo
# without a safeguard like this one, see techniques.py's meta_prompting.
DEV_SET = [
    {"ticket": "A trial user says the onboarding checklist has a typo in step 3.", "expected_escalate": False},
    {"ticket": "Our biggest reseller partner says they'll pause promoting us unless we fix API rate limits by Friday.", "expected_escalate": True},
    {"ticket": "A user asks whether the mobile app supports tablets.", "expected_escalate": False},
    {"ticket": "Support inbox got 40 identical auto-generated tickets in the last hour from what looks like a broken webhook loop on the customer's own side, not ours.", "expected_escalate": False}
]
