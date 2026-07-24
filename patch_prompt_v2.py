#!/usr/bin/env python3
"""
patch_prompt_v2.py — add a version-selectable few-shot example to the planner.

WHY (evidence, not speculation):
  Across the easy and hard Exp 2 batteries, EMPTY_TRIGGERS accounted for 10 of 15 flagged
  failures, and ALL THREE successful self-heals in the entire suite were repairing the same
  defect: the entry state's trigger was too narrow, so the workflow dead-ended at state 1 when
  the browser did not happen to start on chrome://new-tab-page/. Every time, the repair applied
  the same fix -- broaden the entry trigger. At ~3,100 output tokens and ~14s per repair call,
  the loop has been paying repeatedly to rediscover one static fact.

WHAT CHANGES (one conceptual change: "triggers must be permissive enough to match reality"):
  v2 differs from v1 in exactly two triggers of the same worked example --
    1. entry state    : chrome://new-tab-page/ ONLY  ->  + about:blank, data:, no-page-loaded
    2. wikipedia state: .../wiki/Main_Page (over-specific)  ->  en.wikipedia.org
  Nothing else changes. Same task, same state count, same Terminate convention, same structure.

WHERE IT LIVES: the few-shot example, NOT _SYSTEM_RULES. The rules are shared by generation AND
repair, so putting the fix there would hand the same knowledge to the repair loop and blur the
v1-vs-v2 ablation. Keeping it in the example preserves the generation/repair separation.

HOW TO SELECT: environment variable, so no experiment script needs editing.
    PLANNER_PROMPT_VERSION=v1   (default -- current behaviour, unchanged)
    PLANNER_PROMPT_VERSION=v2   (broadened triggers)

USAGE:  python3 patch_prompt_v2.py [--dir .] [--dry-run] [--revert]
"""
import argparse
import os
import sys

TARGET = "netgent_planner.py"

# --------------------------------------------------------------------------- #
IMPORT_OLD = """import json
import re
import time
import logging"""

IMPORT_NEW = """import json
import os
import re
import time
import logging"""

FEWSHOT_END_OLD = '''    "actions": ["Terminate: task complete, article reached and scrolled"],
    "end_state": "Action completed"
  }
]
\'\'\'.strip()'''

FEWSHOT_END_NEW = '''    "actions": ["Terminate: task complete, article reached and scrolled"],
    "end_state": "Action completed"
  }
]
\'\'\'.strip()

# --------------------------------------------------------------------------- #
#  PROMPT VERSIONING  (see patch_prompt_v2.py for the evidence behind v2)      #
#                                                                             #
#  v1 = the example above, unchanged. Kept so the v1-vs-v2 ablation can run    #
#       from a single harness by flipping PLANNER_PROMPT_VERSION.              #
#  v2 = identical task/structure, but two triggers broadened. Motivation:      #
#       EMPTY_TRIGGERS was 10/15 of observed failures and 3/3 of successful    #
#       self-heals; the repair loop kept rediscovering this same fix.          #
# --------------------------------------------------------------------------- #

_FEWSHOT_V1 = _FEWSHOT

_FEWSHOT_V2 = r\'\'\'
User sentence:
"go to wikipedia, search for bezier curves, and scroll all the way down to the end of the article"

Workflow:
[
  {
    "name": "On Browser Home Page",
    "description": "Start the process from a fresh browser tab. The starting page is not guaranteed, so this state's triggers stay deliberately broad -- if none of them match, the whole workflow dead-ends here.",
    "triggers": [
      "If the current URL is chrome://new-tab-page/",
      "If the current URL is about:blank",
      "If the current URL starts with data:",
      "If no page has been loaded yet"
    ],
    "actions": ["Navigate to https://en.wikipedia.org/"]
  },
  {
    "name": "On Wikipedia Homepage",
    "description": "Search for a specific topic on Wikipedia. Match the site broadly rather than pinning one exact landing path, which may differ between runs.",
    "triggers": [
      "If the current URL contains en.wikipedia.org",
      "If a search box with placeholder 'Search Wikipedia' is visible"
    ],
    "actions": [
      "Click the search box",
      "Type 'Bezier curve' into the search box",
      "Press Enter"
    ]
  },
  {
    "name": "On Search Results Or Article Page",
    "description": "Confirm the search navigated to the Bezier curve article page.",
    "triggers": [
      "If the current URL contains en.wikipedia.org/wiki/B",
      "If the current URL contains en.wikipedia.org/wiki/Special:Search"
    ],
    "actions": [
      "If on a search results list, click the first search result link",
      "Scroll down 600 pixels on the article page"
    ]
  },
  {
    "name": "Article Loaded And Scrolled",
    "description": "The target article is open and has been scrolled, confirming successful load.",
    "triggers": [
      "If the current URL contains en.wikipedia.org/wiki/",
      "If the page has scrolled away from the top"
    ],
    "actions": ["Terminate: task complete, article reached and scrolled"],
    "end_state": "Action completed"
  }
]
\'\'\'.strip()

PROMPT_VERSION = os.environ.get("PLANNER_PROMPT_VERSION", "v1").strip().lower()
if PROMPT_VERSION not in ("v1", "v2"):
    logger.warning("Unknown PLANNER_PROMPT_VERSION=%r; falling back to v1", PROMPT_VERSION)
    PROMPT_VERSION = "v1"
if PROMPT_VERSION == "v2":
    _FEWSHOT = _FEWSHOT_V2
logger.info("Planner prompt version: %s", PROMPT_VERSION)'''

# `sentinel` decides whether the patch is already applied. It must NOT be a substring of the
# unpatched file -- note that FEWSHOT_END_NEW *begins with* FEWSHOT_END_OLD, so a naive
# "is new in src" test would report a false positive in both directions.
PATCHES = [
    {"name": "add `import os`",
     "old": IMPORT_OLD, "new": IMPORT_NEW, "sentinel": IMPORT_NEW},
    {"name": "add _FEWSHOT_V2 + env-based version selection",
     "old": FEWSHOT_END_OLD, "new": FEWSHOT_END_NEW,
     "sentinel": "_FEWSHOT_V2 = r'''"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true", help="undo this patch")
    args = ap.parse_args()

    path = os.path.join(args.dir, TARGET)
    if not os.path.exists(path):
        print(f"[MISS] {TARGET} not found in {args.dir}")
        return 1

    src = open(path, encoding="utf-8").read()
    applied = skipped = failed = 0

    for p in (list(reversed(PATCHES)) if args.revert else PATCHES):
        is_applied = p["sentinel"] in src
        want_applied = not args.revert
        if is_applied == want_applied:
            print(f"[SKIP] {p['name']} (already {'applied' if is_applied else 'reverted'})")
            skipped += 1
            continue
        old, new = (p["new"], p["old"]) if args.revert else (p["old"], p["new"])
        if old not in src:
            print(f"[FAIL] {p['name']} -- anchor not found; patch by hand")
            failed += 1
            continue
        if src.count(old) > 1:
            print(f"[FAIL] {p['name']} -- anchor ambiguous ({src.count(old)}x)")
            failed += 1
            continue
        src = src.replace(old, new, 1)
        print(f"[ OK ] {p['name']} {'reverted' if args.revert else 'applied'}")
        applied += 1

    if not args.dry_run and applied:
        open(path, "w", encoding="utf-8").write(src)

    print(f"\napplied={applied} skipped={skipped} failed={failed}"
          f"{'  (dry run -- nothing written)' if args.dry_run else ''}")

    if not args.revert and failed == 0:
        print("""
VERIFY:
  PLANNER_PROMPT_VERSION=v1 python3 -c "import netgent_planner as P; print(P.PROMPT_VERSION, len(P._FEWSHOT))"
  PLANNER_PROMPT_VERSION=v2 python3 -c "import netgent_planner as P; print(P.PROMPT_VERSION, len(P._FEWSHOT))"
  -> v2 must report a LONGER _FEWSHOT than v1. If the lengths match, the swap did not take.

SCOPE DISCIPLINE:
  This is the ONLY planned prompt change. Freeze it here. If v2 underperforms, report that
  honestly -- a negative ablation result is still a result, and re-opening the prompt after
  seeing the numbers turns an experiment into fitting.

TOKEN COST:
  v2 adds roughly 45-60 input tokens per generation call. Measure it from the telemetry
  (input_tokens, GENERATE stage) and report it alongside any accuracy gain.
""")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
