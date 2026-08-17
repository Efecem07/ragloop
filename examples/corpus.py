"""A small labelled corpus, built to expose the failure modes hybrid retrieval
exists to fix.

Every query below is deliberately one of three kinds:

  LEXICAL   the answer contains a literal the query names (an error code, a
            paragraph reference, a plan name). Dense retrieval tends to return
            something merely on-topic.
  SEMANTIC  the query and the answer share no meaningful terms. BM25 returns
            nothing at all.
  BOTH      an ordinary question where either retriever would do.

A retrieval eval set is only as good as its labels, so the relevant chunk ids
here were assigned by reading the documents, not by asking a model.
"""
from __future__ import annotations

DOCUMENTS: dict[str, str] = {
    "errors.md": """
Error TS2345 is returned when the webhook payload does not match the expected
schema. Check the field types before retrying. The request is not queued, so a
corrected payload can be sent immediately.

Error RL429 means the account exceeded its rate limit for the current minute.
Requests are not lost; they are rejected and must be retried with backoff. The
limit resets at the start of the next minute.

Error AUTH401 indicates an expired or revoked API token. Tokens issued before a
key rotation stop working immediately and cannot be refreshed.
""",
    "billing.md": """
Payment authorisation can fail for reasons outside our control. The most common
is that the issuing bank rejects the transaction, which we surface without
modification because only the bank can explain it.

Invoices are generated on the first working day of each month and cover the
previous month. A failed payment does not stop an invoice from being issued.

Refunds are processed to the original payment method and take between five and
ten working days to appear, depending on the bank.
""",
    "plans.md": """
The Starter plan includes one workspace, three seats and email support. It is
intended for teams evaluating the product.

The Growth plan includes ten workspaces, unlimited seats, the audit log and
priority support. Most customers move to Growth once more than one team is
using the product.

Downgrading takes effect at the end of the billing period. Data above the lower
plan's limits is retained in read-only form for ninety days.
""",
    "onboarding.md": """
A new workspace starts empty. The fastest way to see the product working is to
import an existing dataset from a CSV file, which maps columns to fields
automatically and reports anything it could not match.

Team members are added by email invitation. An invitation expires after seven
days and can be reissued from the same screen.

Single sign-on is available on the Growth plan and is configured per domain.
""",
    "security.md": """
API tokens are shown once, at creation. We store only a hash, so a lost token
cannot be recovered and must be rotated.

All data is encrypted in transit and at rest. Backups are taken nightly and
retained for thirty days in the same region as the primary database.

Access to production is limited to on-call engineers and every session is
recorded in the audit log, which is available to customers on the Growth plan.
""",
    "office.md": """
Our office is open from nine to five on weekdays. Visitors should register at
reception on the ground floor.

Team lunches are on the second Thursday of the month. Parking spaces can be
booked through the intranet up to a week in advance.
""",
}


# (query, relevant chunk ids, kind)
# Chunk ids assume chunk_documents(DOCUMENTS, size=400, overlap=60); the demo
# asserts that assumption rather than trusting it.
QUERIES: list[tuple[str, set[str], str]] = [
    ("TS2345", {"errors.md#0"}, "lexical"),
    ("RL429 rate limit", {"errors.md#0"}, "lexical"),
    ("AUTH401", {"errors.md#1"}, "lexical"),
    ("my card was declined", {"billing.md#0"}, "semantic"),
    ("when do I get my money back", {"billing.md#1"}, "semantic"),
    ("how long until a refund shows up", {"billing.md#1"}, "both"),
    ("what do I get for upgrading", {"plans.md#0", "plans.md#1"}, "semantic"),
    ("Growth plan audit log", {"plans.md#0", "security.md#1"}, "lexical"),
    ("what happens to my data if I move to a cheaper plan", {"plans.md#1"}, "semantic"),
    ("import a CSV", {"onboarding.md#0"}, "lexical"),
    ("how do I get colleagues into the workspace", {"onboarding.md#0"}, "semantic"),
    ("I lost my API token", {"security.md#0"}, "both"),
    ("is my data encrypted", {"security.md#0"}, "lexical"),
    ("how often are backups taken", {"security.md#0"}, "both"),
    ("single sign-on", {"onboarding.md#1"}, "lexical"),
    ("can I park at the office", {"office.md#0"}, "both"),
]
