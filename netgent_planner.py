"""
netgent_planner.py
------------------
A lightweight, NetGent-specific "planner" that removes two human dependencies:

  Task 2  generate_workflow(sentence, llm)
          one plain-language sentence  ->  a NetGent NFA workflow (list of StatePrompt dicts)

  Task 3  diagnose_and_repair(sentence, workflow, run_result, llm)
          a failed NetGent run  ->  a more specific workflow that should get further

This is deliberately NOT the Plan-and-Act pipeline (no 70B fine-tuning, no synthetic-data
stack). It is a single prompted call to Gemini that emits NetGent's own state format, which
NetGent then compiles + caches as usual. See plan_and_act_limitations.md for why we went
this route.

The planner is model-agnostic: pass any LangChain chat model (we use Gemini via the RMP key).
Later this call can be swapped for a small local model without changing anything downstream.
"""

import json
import os
import re
import time
import logging

logger = logging.getLogger(__name__)

# Telemetry is optional: if planner_telemetry.py isn't importable the planner still runs,
# it just records nothing. Set PLANNER_TELEMETRY_DIR to turn recording on.
try:
    from planner_telemetry import get_telemetry
except Exception:  # pragma: no cover
    def get_telemetry():
        return None

# Soft import so this file is testable without the full netgent install.
try:
    from netgent.utils.message import StatePrompt
except Exception:  # pragma: no cover
    StatePrompt = None


# --------------------------------------------------------------------------- #
#  Debug printing                                                             #
# --------------------------------------------------------------------------- #
# These print directly to stdout (not just logging.debug) so they show up in the
# foreground `docker run` / `netgent -e ...` output you're already watching, with no
# extra flags needed. Every line is prefixed "[PLANNER DEBUG]" and tagged with a stage
# name so it's obvious which step produced it when reading a long log.

DEBUG = True  # flip to False to silence all [PLANNER DEBUG] output


def _debug(stage, msg):
    if not DEBUG:
        return
    print(f"[PLANNER DEBUG][{stage}] {msg}", flush=True)


def _debug_block(stage, title, text):
    """Print a clearly-delimited multi-line block (raw LLM output, JSON, etc.)."""
    if not DEBUG:
        return
    bar = "-" * 70
    print(f"[PLANNER DEBUG][{stage}] {title}\n{bar}\n{text}\n{bar}", flush=True)


# --------------------------------------------------------------------------- #
#  Prompts                                                                     #
# --------------------------------------------------------------------------- #

_FEWSHOT = r'''
User sentence:
"go to wikipedia, search for bezier curves, and scroll all the way down to the end of the article"

Workflow:
[
  {
    "name": "On Browser Home Page",
    "description": "Start the process from a fresh browser tab.",
    "triggers": ["If the current URL is chrome://new-tab-page/"],
    "actions": ["Navigate to https://en.wikipedia.org/"]
  },
  {
    "name": "On Wikipedia Homepage",
    "description": "Search for a specific topic on Wikipedia.",
    "triggers": [
      "If the current URL contains en.wikipedia.org/wiki/Main_Page",
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
'''.strip()

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

_FEWSHOT_V2 = r'''
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
'''.strip()

PROMPT_VERSION = os.environ.get("PLANNER_PROMPT_VERSION", "v1").strip().lower()
if PROMPT_VERSION not in ("v1", "v2"):
    logger.warning("Unknown PLANNER_PROMPT_VERSION=%r; falling back to v1", PROMPT_VERSION)
    PROMPT_VERSION = "v1"
if PROMPT_VERSION == "v2":
    _FEWSHOT = _FEWSHOT_V2
logger.info("Planner prompt version: %s", PROMPT_VERSION)

# Shared hard rule about output format, reused by both generation and repair so the repair
# model is held to the same "valid JSON, no comments" contract as the generator.
_JSON_OUTPUT_CONTRACT = (
    "Output ONLY a single JSON array and nothing else. It MUST be strictly valid JSON:\n"
    "  - No markdown fences, no prose before or after.\n"
    "  - NO comments of any kind (no // ... and no /* ... */).\n"
    "  - No trailing commas.\n"
    "  - Every string in double quotes; use single quotes for literals inside a string\n"
    "    (e.g. \"Type 'hello' into the box\")."
)

_SYSTEM_RULES = r"""
You convert one plain-language browsing task into a NetGent workflow.

A NetGent workflow is a JSON array of STATES. NetGent runs like a non-deterministic finite
automaton: on every page it checks each state's TRIGGERS, and when one matches it runs that
state's ACTIONS. So each state must be recognizable from the page it applies to.

Rules for a good workflow:
1. Each state is an object with keys: "name", "description", "triggers", "actions".
   The FINAL state must also include "end_state" (a short completion string).
2. TRIGGERS describe how to recognize the page the state applies to. Prefer stable signals:
   a URL substring, a visible unique text/placeholder/button label. Give 1-2 triggers per state.
   Do NOT rely on brittle exact URLs for pages that change (e.g. article slugs) -- use a
   substring or a visible element instead.
3. ACTIONS are concrete, ordered, one instruction each: navigate, click X, type 'Y' into Z,
   press Enter, scroll down N pixels, etc. Keep them specific enough to execute without guessing.
4. First state should start from a fresh tab (trigger on chrome://new-tab-page/ ) and navigate
   to the starting site, unless the sentence implies you are already somewhere.
5. The last state confirms the goal was reached and terminates with an "end_state".
6. Keep it minimal: one state per meaningful page transition. Don't invent steps the sentence
   didn't ask for.
7. A state's "actions" list MAY be empty ([]) if the state exists purely to wait/observe --
   e.g. "waiting for email verification to complete" or "waiting for a redirect after login."
   NetGent will just keep re-checking that state's triggers on every poll until a DIFFERENT
   state's trigger matches (such as the page after verification succeeds), so an empty-action
   state is how you express "sit here until something external changes." Use this sparingly --
   only for genuine external waits (2FA, email confirmation links, redirects) -- and always add
   the NEXT state that recognizes what the page looks like once the wait is over.

""" + _JSON_OUTPUT_CONTRACT


_REPAIR_RULES = r"""
A NetGent workflow you generated did not finish. NetGent stalled or failed before reaching the
final end_state. Your job is to produce a BETTER, MORE SPECIFIC version of the workflow that is
more likely to get past the point where it stalled.

Most stalls are caused by a trigger that never matched (too strict / wrong URL / element not
present) or an action that was too vague. Fix the workflow by:
  - Loosening or correcting the trigger of the state where it got stuck (use a URL substring or
    a visible text/label instead of an exact match), and/or adding an alternative trigger.
  - Making the stuck state's actions more explicit and granular (e.g. split "search for X" into
    click search box -> type X -> press Enter; add an intermediate "click first result" step).
  - Adding a recovery/intermediate state if a page in between was missed (cookie banner, consent
    dialog, redirect, results page before the article).
Reminder: a state's "actions" can legitimately be [] if it's a pure wait/observe state for
something external (2FA, email verification, a redirect) -- that's fine and not a bug, as long
as a later state recognizes the page once the wait resolves.
Keep the parts that already worked. Do NOT add explanatory notes or caveats as comments -- if you
must note an assumption, fold it into a state's "description" field instead. Return the full
corrected workflow as one JSON array, same format as before (final state keeps its "end_state").

""" + _JSON_OUTPUT_CONTRACT


# --------------------------------------------------------------------------- #
#  JSON extraction + validation                                               #
# --------------------------------------------------------------------------- #

def _sanitize_jsonish(text):
    """Make near-JSON from an LLM parseable: strip // and /* */ comments and trailing commas,
    in a STRING-AWARE way so that '//' inside a value (e.g. https://...) is preserved.

    This exists because repair-model output intermittently contains JS-style comments and
    trailing commas even when told not to. We remove them rather than fail the run.
    """
    out = []
    i, n = 0, len(text)
    in_str = False
    esc = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        # --- outside any string ---
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        # line comment: // ... end-of-line
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        # block comment: /* ... */
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        # trailing comma: a comma whose next significant char (skipping ws/comments) is } or ]
        if c == ",":
            j = i + 1
            while j < n:
                if text[j] in " \t\r\n":
                    j += 1
                elif text[j] == "/" and j + 1 < n and text[j + 1] == "/":
                    while j < n and text[j] != "\n":
                        j += 1
                elif text[j] == "/" and j + 1 < n and text[j + 1] == "*":
                    j += 2
                    while j + 1 < n and not (text[j] == "*" and text[j + 1] == "/"):
                        j += 1
                    j += 2
                else:
                    break
            if j < n and text[j] in "}]":
                i += 1          # drop the trailing comma
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_json_array(text):
    """Pull the first top-level JSON array out of an LLM response, tolerating fences,
    surrounding prose, // and /* */ comments, and trailing commas."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # remove comments / trailing commas (string-aware) before locating the array
    text = _sanitize_jsonish(text)

    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("Unterminated JSON array in model output")


def _validate(workflow):
    """Ensure the workflow matches NetGent's StatePrompt schema; coerce/repair light issues."""
    if not isinstance(workflow, list) or not workflow:
        raise ValueError("Workflow must be a non-empty JSON array of states")
    cleaned = []
    for i, st in enumerate(workflow):
        if not isinstance(st, dict):
            raise ValueError(f"State {i} is not an object")
        name = st.get("name") or f"State {i+1}"
        triggers = st.get("triggers") or []
        actions = st.get("actions") or []
        if isinstance(triggers, str):
            triggers = [triggers]
        if isinstance(actions, str):
            actions = [actions]
        if not actions:
            _debug("VALIDATE", f"State '{name}' has NO actions (0 actions) -- treating this as "
                                f"a legitimate wait/observe state. NetGent's executor no-ops on "
                                f"an empty actions list and just re-checks triggers on the next "
                                f"poll, so this is fine, not an error. If this wasn't intentional, "
                                f"check the state's triggers/description above in the debug log.")
        cleaned.append({
            "name": name,
            "description": st.get("description", ""),
            "triggers": triggers,
            "actions": actions,
            "end_state": st.get("end_state", ""),
        })
    # guarantee exactly one terminal end_state (on the last state) if none provided
    if not any(s["end_state"] for s in cleaned):
        cleaned[-1]["end_state"] = "Action completed"
    return cleaned


# --------------------------------------------------------------------------- #
#  Static workflow linting (Task 2 validation -- no browser/NetGent needed)   #
# --------------------------------------------------------------------------- #
# _validate() above only checks SCHEMA (right keys, right types) -- it will happily
# accept a workflow that is schema-valid but practically broken, like a state with an
# empty triggers list (which NetGent silently treats as "matches every page forever",
# per controller.py's _check_state -- see the bug this caught in the Gmail run).
#
# lint_workflow() checks the things that actually caused real failures, so you can catch
# them the moment a workflow is generated, before ever touching Docker/NetGent/a browser.

_WAIT_KEYWORDS = ("wait", "2fa", "verif", "otp", "redirect", "captcha", "confirm",
                  "check your", "authenticat", "loading", "processing")


def lint_workflow(workflow, sentence=None):
    """Static checks on a generated workflow. Returns a list of issue dicts:
        {"level": "ERROR"|"WARN"|"INFO", "state": <name or None>, "message": <str>}

    ERROR   = will break NetGent or is structurally invalid (would have crashed before,
              or will silently misbehave like the empty-triggers trap).
    WARN    = probably a problem, worth a human glance, not guaranteed to fail.
    INFO    = just a note (e.g. an intentional-looking wait state).

    This does NOT run a browser. It's pure static analysis of the JSON, so it's fast
    enough to run on every single generation in a test loop.
    """
    issues = []

    def err(msg, state=None):
        issues.append({"level": "ERROR", "state": state, "message": msg})

    def warn(msg, state=None):
        issues.append({"level": "WARN", "state": state, "message": msg})

    def info(msg, state=None):
        issues.append({"level": "INFO", "state": state, "message": msg})

    if not workflow:
        err("Workflow is empty.")
        return issues

    names = [s.get("name", "") for s in workflow]
    dupes = {n for n in names if names.count(n) > 1 and n}
    for d in dupes:
        err(f"State name '{d}' is used more than once. NetGent may not be able to "
            f"tell these states apart.", state=d)

    for i, state in enumerate(workflow):
        name = state.get("name") or f"<state {i}>"
        triggers = state.get("triggers") or []
        actions = state.get("actions") or []

        # --- THE critical check: empty triggers = matches every page forever ---
        if not triggers:
            err("Empty triggers list. NetGent's _check_state() treats a state with no "
                "checks as ALWAYS matching, so this state will fire on every page and "
                "never let the workflow advance -- this is the exact bug that caused the "
                "Gmail run to time out. Every state MUST have at least one trigger.",
                state=name)

        # --- empty actions: only OK if it reads like an intentional wait state ---
        if not actions:
            haystack = f"{name} {state.get('description', '')}".lower()
            if any(k in haystack for k in _WAIT_KEYWORDS):
                info("Empty actions with a wait/verification-sounding name -- looks "
                     "like an intentional 'sit here until something external changes' "
                     "state. Confirm that's what you meant.", state=name)
            else:
                warn("Empty actions list, but the name/description doesn't read like "
                     "a wait state. If this wasn't intentional, NetGent will do "
                     "nothing here and just re-poll the same triggers.", state=name)

        if i == 0:
            joined = " ".join(triggers).lower()
            if "chrome://new-tab-page" not in joined and "new-tab-page" not in joined:
                warn("First state doesn't trigger on chrome://new-tab-page/. This is "
                     "fine ONLY if the task explicitly starts from an existing page; "
                     "otherwise NetGent may never enter the workflow from a fresh tab.",
                     state=name)

        if len(triggers) > 4:
            warn(f"{len(triggers)} triggers on one state -- unusually many. Triggers "
                 f"are OR'd together in NetGent, so this may match more pages than "
                 f"intended.", state=name)

    # --- terminal state checks ---
    last = workflow[-1]
    last_name = last.get("name", "<last state>")
    if not last.get("end_state"):
        err("Last state has no 'end_state'. detect_failure() uses the terminal "
            "state's NAME to judge success, but a missing end_state usually means "
            "the model never actually closed out the task.", state=last_name)
    if not any("terminat" in a.lower() for a in (last.get("actions") or [])):
        # Promoted WARN -> ERROR on Exp 5 evidence: zero-shot generation omitted the
        # Terminate action in 35/35 workflows (few-shot 4/36, chi2=56.6) while
        # struct_valid_rate still read 0.97. A workflow that cannot signal completion
        # is functionally broken, so structural validity must not pass it.
        err("Last state's actions don't mention 'Terminate'. NetGent's web agent "
            "needs an explicit terminate instruction to stop cleanly.", state=last_name)

    # --- coarse size sanity ---
    if len(workflow) == 1:
        warn("Only one state in the whole workflow. Unless the task is trivially "
             "one page, this usually means the model under-decomposed the task.")
    elif len(workflow) > 12:
        warn(f"{len(workflow)} states is a lot for one task -- check the model didn't "
             f"invent steps the sentence never asked for (see rule 6 in the prompt).")

    return issues


def format_lint_report(issues):
    """Human-readable rendering of lint_workflow() output."""
    if not issues:
        return "  [LINT] No issues found."
    lines = []
    for iss in issues:
        tag = iss["state"] and f" ({iss['state']})" or ""
        lines.append(f"  [LINT][{iss['level']}]{tag} {iss['message']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Verifier <-> Planner interface                                             #
# --------------------------------------------------------------------------- #
# SCOPE NOTE: detecting/verifying that a run actually failed is the VERIFIER's job
# (Oliver's component). This planner's job is only to REPAIR the workflow once it's been
# told something went wrong. The two connect through the VerificationReport dict below.
#
# Keeping this boundary explicit matters for two reasons:
#   1. Practical: you and Oliver can build and test independently against a fixed schema.
#   2. Design: it matches the consensus in the papers citing Plan-and-Act -- recovery
#      should be conditioned on an *inferred failure class*, not a blind "regenerate and
#      hope" (see arXiv:2606.01416 on conditioning recovery on the observed failure
#      signal + failure class, and arXiv:2509.25238 / PALADIN on needing an explicit
#      failure taxonomy). The verifier supplies the class; the planner picks the fix.

# Failure taxonomy. The verifier should classify into one of these; the planner maps each
# class to a specific repair strategy (REPAIR_STRATEGY below).
FAILURE_CLASSES = (
    "TRIGGER_NEVER_MATCHED",   # a state's trigger never fired; workflow couldn't advance
    "EMPTY_TRIGGERS",          # state cached with no checks -> matches every page, loops forever
    "UNEXPECTED_PAGE",         # an interstitial appeared (consent/cookie/login/redirect)
    "ACTION_FAILED",           # an action referenced an element that wasn't there
    "ACTION_TOO_VAGUE",        # action executed but did the wrong/ambiguous thing
    "TIMEOUT",                 # state_timeout hit without advancing
    "WRONG_END_STATE",         # terminated, but not at the intended goal
    "BLOCKED",                 # captcha / bot-detection / hard block -- OUT OF SCOPE to repair
    "TOKEN_LIMIT",             # LLM output truncated
    "UNKNOWN",                 # verifier is sure it failed but can't classify
)

# Per-class repair guidance handed to the model. This is the "apply the smallest fix that
# addresses the observed failure class" principle, rather than always regenerating the
# whole workflow from scratch.
REPAIR_STRATEGY = {
    "TRIGGER_NEVER_MATCHED":
        "The trigger for the stuck state never fired. Loosen it: replace an exact URL match "
        "with a URL SUBSTRING, or key off a visible text/label/placeholder on the page "
        "instead. Add a second alternative trigger for the same state. Do not touch states "
        "that already worked.",
    "EMPTY_TRIGGERS":
        "The stuck state was cached with an EMPTY trigger list, which makes NetGent match it "
        "against every page and loop forever. Give this state at least one CONCRETE, "
        "synthesizable trigger: a URL substring, or a specific visible text string. Never "
        "emit a state with an empty triggers list.",
    "UNEXPECTED_PAGE":
        "An unexpected interstitial page appeared (cookie/consent banner, login wall, or a "
        "redirect). Insert ONE new state that recognizes that interstitial and dismisses or "
        "passes through it, then continue to the state that was already working. Keep the "
        "rest of the workflow unchanged.",
    "ACTION_FAILED":
        "An action referred to an element that wasn't present. Make the action reference "
        "something stable and visible (button label, placeholder text, link text), and split "
        "compound actions into single explicit steps.",
    "ACTION_TOO_VAGUE":
        "The action was ambiguous and the agent did the wrong thing. Rewrite the stuck "
        "state's actions to be granular and unambiguous: one instruction per action, naming "
        "exactly what to click/type, in order.",
    "TIMEOUT":
        "The workflow sat on a state without advancing. Either the trigger is matching a page "
        "it shouldn't, or the following state's trigger is too strict to fire. Tighten the "
        "stuck state's trigger and loosen the NEXT state's trigger.",
    "WRONG_END_STATE":
        "The run terminated somewhere other than the intended goal. Check the terminal "
        "state's trigger actually describes the goal page, and make sure no earlier state "
        "terminates prematurely.",
    "BLOCKED":
        "(Out of scope) The run was blocked by a captcha or bot detection. Do not attempt to "
        "solve or evade it. Return the workflow unchanged.",
    "TOKEN_LIMIT":
        "The previous generation was truncated. Return the SAME workflow but more concisely: "
        "shorter descriptions, fewer redundant triggers.",
    "UNKNOWN":
        "The failure could not be classified. Make the stuck state's triggers more permissive "
        "and its actions more explicit, and add an intermediate state if a page transition "
        "looks like it was skipped.",
}

# Repair is pointless (or actively wrong) for these classes -- the planner should decline
# rather than burn API calls pretending it can fix them.
UNREPAIRABLE_CLASSES = ("BLOCKED",)


def make_report(failed, failure_class="UNKNOWN", reason="", stuck_state=None,
                reached_states=None, evidence=None, source="verifier", confidence=None):
    """Build a VerificationReport -- the contract between the verifier and this planner.

    This is the ONLY thing diagnose_and_repair() needs from the verifier. Oliver's agent
    should return exactly this shape (or something that can be adapted into it).

    Fields:
      failed         bool   -- did the run actually fail? (verifier's core judgement)
      failure_class  str    -- one of FAILURE_CLASSES
      reason         str    -- human-readable explanation, passed to the repair model
      stuck_state    str    -- name of the state where it broke (None if unknown)
      reached_states list   -- state names that ran successfully before the failure
      evidence       dict   -- optional freeform: {"url":..., "page_text":..., "screenshot":...}
      source         str    -- "verifier" (Oliver's agent) or "stopgap_heuristic" (the
                              placeholder below). Recorded in telemetry so results from the
                              real verifier are never silently mixed with placeholder ones.
      confidence     float  -- optional 0-1 from the verifier
    """
    if failure_class not in FAILURE_CLASSES:
        logger.warning("Unknown failure_class %r; coercing to UNKNOWN", failure_class)
        failure_class = "UNKNOWN"
    return {
        "failed": bool(failed),
        "failure_class": failure_class,
        "reason": reason or "",
        "stuck_state": stuck_state,
        "reached_states": list(reached_states or []),
        "evidence": evidence or {},
        "source": source,
        "confidence": confidence,
    }


def validate_report(report):
    """Raise if a report doesn't meet the contract. Call this on whatever the verifier
    hands over, so a schema drift between the two components fails loudly and early
    rather than silently producing a nonsense repair."""
    if not isinstance(report, dict):
        raise TypeError(f"VerificationReport must be a dict, got {type(report).__name__}")
    if "failed" not in report:
        raise ValueError("VerificationReport missing required key 'failed'")
    fc = report.get("failure_class", "UNKNOWN")
    if fc not in FAILURE_CLASSES:
        raise ValueError(f"failure_class {fc!r} not in FAILURE_CLASSES {FAILURE_CLASSES}")
    return True


def _invoke_text(llm, system, user, stage="LLM_CALL"):
    """Call a LangChain chat model and return raw text (handles str or .content).

    Also records the full call to telemetry: both prompts, the raw output, wall-clock
    latency, and token usage. Returns just the text so callers are unchanged; the
    per-call record is retrievable via _LAST_CALL for the caller to aggregate.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    _debug_block(stage, "SYSTEM PROMPT SENT", system)
    _debug_block(stage, "USER PROMPT SENT", user)

    tel = get_telemetry()
    model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    temperature = getattr(llm, "temperature", None)

    t0 = time.time()
    resp, err, text = None, None, ""
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        latency = time.time() - t0
        rec = None
        if tel is not None:
            rec = tel.log_llm_call(stage=stage, system_prompt=system, user_prompt=user,
                                   raw_output=text, latency_s=latency, resp=resp,
                                   model=model, temperature=temperature, error=err)
        global _LAST_CALL
        _LAST_CALL = rec or {"latency_s": latency, "input_tokens": None,
                             "output_tokens": None, "total_tokens": None,
                             "finish_reason": None, "error": err}

    _debug_block(stage, "RAW LLM OUTPUT (unparsed, straight from the model)", text)
    _debug(stage, f"latency={_LAST_CALL.get('latency_s')}s  "
                  f"tokens in/out/total={_LAST_CALL.get('input_tokens')}/"
                  f"{_LAST_CALL.get('output_tokens')}/{_LAST_CALL.get('total_tokens')}  "
                  f"finish_reason={_LAST_CALL.get('finish_reason')}")
    return text


# Populated by _invoke_text with the telemetry record of the most recent call, so
# _generate_validated can roll per-call token/latency numbers up into the attempt record.
_LAST_CALL = {}


def _generate_validated(system, user, llm, stage="GENERATE", sentence=None, extra=None):
    """Invoke the model and parse; on a JSON error, do ONE self-correcting re-ask before
    giving up. Cheap insurance against the occasional malformed array.

    Records an `attempt` telemetry row covering the whole generate-and-parse cycle,
    including whether the JSON fixup retry was needed (a key quality metric: how often
    does the model emit unparseable JSON on the first try?).
    """
    tel = get_telemetry()
    t_start = time.time()
    tok = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    n_calls = 0

    def _accumulate():
        nonlocal n_calls
        n_calls += 1
        for k in tok:
            v = _LAST_CALL.get(k)
            if v is not None and tok[k] is not None:
                tok[k] += v
            elif v is None:
                tok[k] = tok[k]  # leave running total; None means provider didn't report

    raw = _invoke_text(llm, system, user, stage=stage)
    _accumulate()

    sanitized = _sanitize_jsonish(
        re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    )
    if sanitized != raw.strip():
        _debug_block(stage, "AFTER STRIPPING FENCES/COMMENTS/TRAILING COMMAS", sanitized)
    else:
        _debug(stage, "Sanitizer made no changes (output was already clean JSON).")

    needed_fixup, parse_error, parsed = False, None, None
    try:
        parsed = _validate(_extract_json_array(raw))
        _debug(stage, f"Parsed + validated OK on first try -> {len(parsed)} states: "
                       f"{[s['name'] for s in parsed]}")
    except (ValueError, json.JSONDecodeError) as e:
        needed_fixup = True
        parse_error = f"{type(e).__name__}: {e}"
        _debug(stage, f"PARSE FAILED: {e}")
        logger.warning("First parse failed (%s); asking the model to fix its JSON.", e)
        fix_user = (
            "Your previous output was not valid JSON and failed to parse:\n\n"
            f"{raw}\n\nError: {e}\n\n"
            "Return the SAME workflow as one strictly valid JSON array. "
            "No comments, no trailing commas, no markdown fences, nothing else."
        )
        try:
            raw2 = _invoke_text(llm, _JSON_OUTPUT_CONTRACT, fix_user, stage=f"{stage}_JSON_FIXUP")
            _accumulate()
            parsed = _validate(_extract_json_array(raw2))
            _debug(f"{stage}_JSON_FIXUP", f"Parsed + validated OK after fixup -> {len(parsed)} "
                                           f"states: {[s['name'] for s in parsed]}")
        except (ValueError, json.JSONDecodeError) as e2:
            parse_error = f"fixup also failed: {type(e2).__name__}: {e2}"
            if tel is not None:
                tel.log_attempt(sentence=sentence, stage=stage, workflow=None, issues=[],
                                latency_s=time.time() - t_start, parse_ok=False,
                                needed_fixup=True, parse_error=parse_error,
                                n_llm_calls=n_calls, tokens=tok, extra=extra)
            raise

    if tel is not None:
        tel.log_attempt(sentence=sentence, stage=stage, workflow=parsed,
                        issues=lint_workflow(parsed, sentence=sentence),
                        latency_s=time.time() - t_start, parse_ok=True,
                        needed_fixup=needed_fixup, parse_error=parse_error,
                        n_llm_calls=n_calls, tokens=tok, extra=extra)
    return parsed


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #

def generate_workflow(sentence, llm, as_stateprompts=False):
    """Task 2: plain-language sentence -> validated NetGent workflow (list of dicts).

    Set as_stateprompts=True to get StatePrompt objects ready for agent.run(state_prompts=...).
    """
    _debug("GENERATE", f'Task sentence: "{sentence.strip()}"')
    user = f"{_FEWSHOT}\n\nUser sentence:\n\"{sentence.strip()}\"\n\nWorkflow:"
    workflow = _generate_validated(_SYSTEM_RULES, user, llm, stage="GENERATE", sentence=sentence)
    logger.info("Generated workflow with %d states for: %s", len(workflow), sentence)
    return _as_prompts(workflow) if as_stateprompts else workflow


def diagnose_and_repair(sentence, workflow, report, llm, as_stateprompts=False):
    """Repair a workflow given a VerificationReport. THIS IS THE PLANNER'S HALF.

    `report` is a VerificationReport dict from the verifier (see make_report). Detection is
    NOT done here -- the verifier already decided the run failed and classified why. This
    function only picks the smallest fix for that failure class and regenerates.

    Backwards compatible: if `report` looks like a raw NetGent run_result instead of a
    report, it is passed through the stopgap heuristic and a warning is logged. That path
    exists only so this keeps working until the verifier lands -- results generated that
    way are tagged source="stopgap_heuristic" in telemetry so they are never confused with
    real verifier output.
    """
    # --- accept either a report or (temporarily) a raw run_result ---
    if not isinstance(report, dict) or "failed" not in report:
        logger.warning("diagnose_and_repair() got something that isn't a VerificationReport; "
                       "falling back to the stopgap heuristic. Once the verifier exists, "
                       "pass its report here instead.")
        report = stopgap_verify(report, workflow)
    validate_report(report)

    if not report["failed"]:
        _debug("REPAIR", "Verifier reports no failure -- nothing to repair. Returning workflow unchanged.")
        return _as_prompts(workflow) if as_stateprompts else workflow

    fclass = report.get("failure_class", "UNKNOWN")
    reason = report.get("reason", "")
    stuck = report.get("stuck_state")
    reached = report.get("reached_states") or []

    _debug("REPAIR", f"source={report.get('source')}  class={fclass}  "
                     f"confidence={report.get('confidence')}")
    _debug("REPAIR", f"reason: {reason}")
    _debug("REPAIR", f"stuck state: {stuck}   reached: {', '.join(reached) if reached else 'none'}")

    # --- decline classes we deliberately don't handle (scope decision) ---
    if fclass in UNREPAIRABLE_CLASSES:
        _debug("REPAIR", f"failure_class {fclass} is out of scope for workflow repair "
                         f"(no captcha/bot-detection evasion). Returning workflow unchanged.")
        logger.info("Declining to repair out-of-scope failure class: %s", fclass)
        return _as_prompts(workflow) if as_stateprompts else workflow

    strategy = REPAIR_STRATEGY.get(fclass, REPAIR_STRATEGY["UNKNOWN"])
    evidence = report.get("evidence") or {}
    ev_lines = "\n".join(f"  {k}: {str(v)[:300]}" for k, v in evidence.items()) or "  (none provided)"

    user = (
        f"Original task:\n\"{sentence.strip() if sentence else ''}\"\n\n"
        f"Workflow that failed:\n{json.dumps(workflow, indent=2)}\n\n"
        f"FAILURE CLASS: {fclass}\n"
        f"What the verifier observed: {reason}\n"
        f"State where it got stuck: {stuck or 'unknown'}\n"
        f"States that ran successfully first: {', '.join(reached) if reached else 'none'}\n"
        f"Evidence:\n{ev_lines}\n\n"
        f"HOW TO FIX THIS SPECIFIC FAILURE:\n{strategy}\n\n"
        f"Produce the corrected workflow now."
    )
    repaired = _generate_validated(
        _REPAIR_RULES, user, llm, stage="REPAIR", sentence=sentence,
        extra={"failure_class": fclass,
               "verifier_source": report.get("source"),
               "verifier_confidence": report.get("confidence"),
               "stuck_state": stuck,
               "states_before_repair": len(workflow)})

    # --- lint guard: the repair must not REGRESS into a broken workflow ---
    # In real runs the repair pass has itself emitted an EMPTY_TRIGGERS workflow (the
    # infinite-loop trap), making attempt N+1 worse than attempt N. Since lint_workflow()
    # already detects that class of defect statically, re-ask ONCE if the repair introduced
    # an ERROR-level issue, quoting the specific lint errors back to the model.
    repaired_dicts = repaired if isinstance(repaired[0], dict) else \
        [{"name": p.name, "description": getattr(p, "description", ""),
          "triggers": p.triggers, "actions": p.actions,
          "end_state": getattr(p, "end_state", "")} for p in repaired]
    lint = lint_workflow(repaired_dicts, sentence=sentence)
    errs = [i for i in lint if i["level"] == "ERROR"]
    if errs:
        _debug("REPAIR", f"repair introduced {len(errs)} lint ERROR(s); re-asking once.")
        err_text = "\n".join(f"  - {i.get('state','?')}: {i['message']}" for i in errs)
        guard_user = (
            user + "\n\nYOUR PREVIOUS REPAIR HAD THESE STRUCTURAL ERRORS -- fix them:\n"
            + err_text +
            "\n\nEvery state MUST have at least one concrete trigger (URL substring or visible "
            "text). Produce a corrected workflow with no such errors."
        )
        repaired = _generate_validated(
            _REPAIR_RULES, guard_user, llm, stage="REPAIR_GUARD", sentence=sentence,
            extra={"failure_class": fclass, "verifier_source": report.get("source"),
                   "guard_triggered": True, "states_before_repair": len(workflow)})

    logger.info("Repaired workflow now has %d states (was %d) [class=%s]",
                len(repaired), len(workflow), fclass)
    return _as_prompts(repaired) if as_stateprompts else repaired


# --------------------------------------------------------------------------- #
#  STOPGAP verifier -- placeholder for Oliver's verifier agent                #
# --------------------------------------------------------------------------- #

def stopgap_verify(run_result, workflow):
    """PLACEHOLDER. Returns a VerificationReport from a raw NetGent run_result.

    This is NOT the real verifier -- detection/verification is Oliver's component. This
    exists only so the planner is testable end-to-end before that lands. It is deliberately
    crude: a few heuristics over the run_result dict. Everything it emits is tagged
    source="stopgap_heuristic" so placeholder results never get mistaken for real ones in
    the telemetry or the paper.

    When the verifier is ready, delete this and pass its report to diagnose_and_repair().

    Known-correct facts baked in here (learned from real runs, worth telling Oliver):
      - NetGent does NOT echo our `end_state` back; cached states come back end_state="".
        So success must be judged by terminal state NAME, not end_state.
      - A state cached with an empty `checks` list matches EVERY page (controller.py
        _check_state loops over checks and returns True on an empty list) -> infinite loop.
    """
    if not isinstance(run_result, dict):
        return make_report(True, "UNKNOWN", "Run returned no result object.",
                           source="stopgap_heuristic")

    if run_result.get("error"):
        err = str(run_result["error"])
        low = err.lower()
        if "token" in low and ("limit" in low or "exceed" in low):
            fclass = "TOKEN_LIMIT"
        elif "captcha" in low or "blocked" in low:
            fclass = "BLOCKED"
        elif "timeout" in low or "timed out" in low:
            fclass = "TIMEOUT"
        else:
            fclass = "UNKNOWN"
        return make_report(True, fclass, f"Run raised an error: {err}",
                           source="stopgap_heuristic")

    goal_name = workflow[-1].get("name", "") if workflow else ""
    repo = run_result.get("state_repository") or []
    passed = run_result.get("passed_states") or []
    executed = run_result.get("executed_states") or []
    last_passed = run_result.get("last_passed_state_name") or ""

    def _names(states):
        return [s.get("name", "?") for s in states if isinstance(s, dict)]

    reached = _names(executed) or _names(passed) or _names(repo)

    # success: reached the terminal state BY NAME (end_state is never echoed back)
    if goal_name and (last_passed == goal_name
                      or goal_name in _names(passed)
                      or goal_name in _names(executed)):
        return make_report(False, "UNKNOWN", f"Reached terminal state '{goal_name}'.",
                           reached_states=reached, source="stopgap_heuristic")

    # the always-match/infinite-loop trap
    empty_checks = [s.get("name", "?") for s in repo
                    if isinstance(s, dict) and not s.get("checks")]
    if empty_checks:
        return make_report(
            True, "EMPTY_TRIGGERS",
            f"State(s) [{', '.join(empty_checks)}] were cached with an empty checks list, so "
            f"they matched every page and the workflow never advanced.",
            stuck_state=empty_checks[0], reached_states=reached, source="stopgap_heuristic")

    if not reached:
        first = workflow[0].get("name") if workflow else None
        return make_report(True, "TRIGGER_NEVER_MATCHED",
                           "No states matched at all; the first state's trigger never fired.",
                           stuck_state=first, reached_states=[], source="stopgap_heuristic")

    return make_report(
        True, "TIMEOUT",
        f"Passed {len(reached)} state(s) ({', '.join(reached)}) but never reached the "
        f"terminal state '{goal_name}'.",
        stuck_state=reached[-1] if reached else None,
        reached_states=reached, source="stopgap_heuristic")


def detect_failure(run_result, workflow):
    """DEPRECATED shim. Detection now belongs to the verifier (Oliver's component).

    Kept so existing callers (run_planner.py) don't break. Returns the old
    (failed, reason, reached) tuple by adapting a stopgap_verify() report.
    """
    r = stopgap_verify(run_result, workflow)
    return r["failed"], r["reason"], r["reached_states"]


def _as_prompts(workflow):
    if StatePrompt is None:
        raise RuntimeError("netgent not importable; cannot build StatePrompt objects")
    return [StatePrompt(**s) for s in workflow]
