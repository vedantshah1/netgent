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
import re
import logging

logger = logging.getLogger(__name__)

# Soft import so this file is testable without the full netgent install.
try:
    from netgent.utils.message import StatePrompt
except Exception:  # pragma: no cover
    StatePrompt = None


# --------------------------------------------------------------------------- #
#  Prompts                                                                     #
# --------------------------------------------------------------------------- #

# One worked example (the Wikipedia / Bezier task) anchors the format. NetGent states are an
# NFA: each state has triggers (when am I here?) and actions (what do I do?). The final state
# carries an `end_state` so NetGent knows the workflow is done.
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

_SYSTEM_RULES = r"""
You convert one plain-language browsing task into a NetGent workflow.

A NetGent workflow is a JSON array of STATES. NetGent runs like a non-deterministic finite
automaton: on every page it checks each state's TRIGGERS, and when one matches it runs that
state's ACTIONS. So each state must be recognizable from the page it applies to.

Rules for a good workflow:
1. Output ONLY a JSON array. No prose, no markdown fences, no comments.
2. Each state is an object with keys: "name", "description", "triggers", "actions".
   The FINAL state must also include "end_state" (a short completion string).
3. TRIGGERS describe how to recognize the page the state applies to. Prefer stable signals:
   a URL substring, a visible unique text/placeholder/button label. Give 1-2 triggers per state.
   Do NOT rely on brittle exact URLs for pages that change (e.g. article slugs) — use a
   substring or a visible element instead.
4. ACTIONS are concrete, ordered, one instruction each: navigate, click X, type 'Y' into Z,
   press Enter, scroll down N pixels, etc. Keep them specific enough to execute without guessing.
5. First state should start from a fresh tab (trigger on chrome://new-tab-page/ ) and navigate
   to the starting site, unless the sentence implies you are already somewhere.
6. The last state confirms the goal was reached and terminates with an "end_state".
7. Keep it minimal: one state per meaningful page transition. Don't invent steps the sentence
   didn't ask for.

Return strictly valid JSON.
"""


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
Keep the parts that already worked. Output ONLY the full corrected JSON array, same format as
before (final state keeps its "end_state").
"""


# --------------------------------------------------------------------------- #
#  JSON extraction + validation                                               #
# --------------------------------------------------------------------------- #

def _extract_json_array(text):
    """Pull the first top-level JSON array out of an LLM response."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
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
            raise ValueError(f"State '{name}' has no actions")
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


def _invoke_text(llm, system, user):
    """Call a LangChain chat model and return raw text (handles str or .content)."""
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if hasattr(resp, "content") else str(resp)


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #

def generate_workflow(sentence, llm, as_stateprompts=False):
    """Task 2: plain-language sentence -> validated NetGent workflow (list of dicts).

    Set as_stateprompts=True to get StatePrompt objects ready for agent.run(state_prompts=...).
    """
    user = f"{_FEWSHOT}\n\nUser sentence:\n\"{sentence.strip()}\"\n\nWorkflow:"
    raw = _invoke_text(llm, _SYSTEM_RULES, user)
    workflow = _validate(_extract_json_array(raw))
    logger.info("Generated workflow with %d states for: %s", len(workflow), sentence)
    return _as_prompts(workflow) if as_stateprompts else workflow


def diagnose_and_repair(sentence, workflow, run_result, llm, as_stateprompts=False):
    """Task 3: given a stalled/failed run, produce a more specific workflow.

    `workflow`   : the list-of-dicts workflow that was run
    `run_result` : dict returned by NetGent's agent.run(...) (or {'error': str} on exception)
    """
    failed, reason, reached = detect_failure(run_result, workflow)
    reached_names = ", ".join(reached) if reached else "none"
    user = (
        f"Original task:\n\"{sentence.strip()}\"\n\n"
        f"Workflow that stalled:\n{json.dumps(workflow, indent=2)}\n\n"
        f"What happened: {reason}\n"
        f"States that ran before stalling: {reached_names}\n\n"
        f"Produce the corrected workflow now."
    )
    raw = _invoke_text(llm, _REPAIR_RULES, user)
    repaired = _validate(_extract_json_array(raw))
    logger.info("Repaired workflow now has %d states (was %d)", len(repaired), len(workflow))
    return _as_prompts(repaired) if as_stateprompts else repaired


def detect_failure(run_result, workflow):
    """Heuristic failure detector for a NetGent generation run.

    Returns (failed: bool, reason: str, reached_state_names: list[str]).

    Success = the run produced/executed a state carrying the workflow's terminal end_state.
    NOTE: tune this against real run_result shapes on the VM; the executed_states / end_state
    signals below match agent.py at the time of writing but are the most likely thing to drift.
    """
    if not isinstance(run_result, dict):
        return True, "Run returned no result object.", []

    if run_result.get("error"):
        return True, f"Run raised an error: {run_result['error']}", []

    goal = ""
    for s in workflow:
        if s.get("end_state"):
            goal = s["end_state"]

    executed = run_result.get("executed_states") or []
    repo = run_result.get("state_repository") or []
    reached = [s.get("name", "?") for s in executed] or [s.get("name", "?") for s in repo]

    def _has_goal(states):
        for s in states:
            es = s.get("end_state")
            if es and (es == goal or (goal == "" and es != "")):
                return True
        return False

    if _has_goal(executed) or _has_goal(repo):
        return False, "Reached terminal end_state.", reached

    # didn't reach the end_state
    if not reached:
        return True, "No states executed at all (likely the first trigger never matched).", []
    return (True,
            f"Stalled after {len(reached)} state(s) without reaching end_state '{goal}'.",
            reached)


def _as_prompts(workflow):
    if StatePrompt is None:
        raise RuntimeError("netgent not importable; cannot build StatePrompt objects")
    return [StatePrompt(**s) for s in workflow]
