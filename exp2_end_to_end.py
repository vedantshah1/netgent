#!/usr/bin/env python3
"""
exp2_end_to_end.py
==================
EXPERIMENT 2 — The one that measures TRUE TASK ACCURACY, not just structural validity.

For each prompt it does the full journey and reports at every stage:
    1. GENERATE   sentence -> workflow (raw LLM output captured)
    2. LINT       static validation report (before any browser)
    3. RUN        actually execute the workflow in the browser via NetGent
    4. VERIFY     check a PER-PROMPT success condition against the final browser state
    5. (optional) SELF-HEAL + retry if it failed

This is the honest accuracy number: does the generated workflow actually DO the task?
As you predicted, expect the initial planner to fail several of these -- that failure data
is exactly what tells us which planner-prompt edits are needed.

WHY the success condition matters: "structural validity" (exp on report1/report2) only checks
the workflow is well-formed. It cannot tell you the workflow WORKS. Here each prompt ships with
an explicit, checkable success condition (final URL contains X, or page text contains Y), so
"success" is grounded in the real end state, not the workflow looking plausible.

Each prompt is defined as:
    {
      "sentence":  the natural-language task,
      "success":   {"type": "url_contains"|"text_contains"|"url_regex", "value": ...},
      "notes":     what we expect to be hard about it
    }

USAGE (must run in the Docker/NetGent environment with a display, like run_planner.py):
    python3 exp2_end_to_end.py --api-keys /keys.json --outdir /out/exp2
    python3 exp2_end_to_end.py --api-keys /keys.json --outdir /out/exp2 --max-repairs 2
    python3 exp2_end_to_end.py --api-keys /keys.json --prompts-json my_prompts.json

OUTPUT: /out/exp2/report.md (human-readable per-prompt journey), per-prompt JSON with the
workflow + lint + run result + verdict, and telemetry.
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netgent_planner import (generate_workflow, diagnose_and_repair, lint_workflow,
                             format_lint_report, stopgap_verify)
from planner_telemetry import Telemetry, set_telemetry
import netgent_planner as P


# --------------------------------------------------------------------------- #
#  The battery: 5 prompts of graded difficulty, each with a success condition  #
# --------------------------------------------------------------------------- #
# Chosen to be POSSIBLE (unlike the Scholar run that asked for a non-existent link) and to
# have UNAMBIGUOUS, checkable end states. Ordered easy -> hard so you can see where it breaks.
DEFAULT_BATTERY = [
    {
        "sentence": "go to example.com",
        "success": {"type": "url_contains", "value": "example.com"},
        "notes": "trivial single navigation -- sanity check the whole pipeline works.",
    },
    {
        "sentence": "go to en.wikipedia.org, search for Bezier curve, and open the article",
        "success": {"type": "url_contains", "value": "/wiki/B"},
        "notes": "search + result navigation; the wikipedia consistency fork lives here.",
    },
    {
        "sentence": "go to the-internet.herokuapp.com/login and log in with username tomsmith and password SuperSecretPassword!",
        "success": {"type": "url_contains", "value": "/secure"},
        "notes": "a real login flow with a stable success page (/secure area).",
    },
    {
        "sentence": "go to the-internet.herokuapp.com/dynamic_loading/2, click Start, and wait for the hidden text to appear",
        "success": {"type": "text_contains", "value": "Hello World"},
        "notes": "async wait -- needs a wait/observe state, tests the empty-actions pattern.",
    },
    {
        "sentence": "go to the-internet.herokuapp.com/checkboxes and make sure both checkboxes are checked",
        "success": {"type": "text_contains", "value": "checkboxes"},
        "notes": "post-condition task (both checked) -- hardest; structural consistency was low here.",
    },
]


def check_success(condition, driver, run_result):
    """Evaluate a prompt's success condition against the real browser end state.

    Returns (ok: bool, detail: str). Reads live from the driver where possible so the verdict
    reflects what actually happened, not what the workflow intended.
    """
    ctype = condition.get("type")
    val = condition.get("value", "")
    try:
        cur_url = driver.current_url
    except Exception:
        cur_url = ""
    try:
        page_text = driver.find_element("tag name", "body").text
    except Exception:
        page_text = ""

    if ctype == "url_contains":
        ok = val.lower() in (cur_url or "").lower()
        return ok, f"final_url={cur_url!r} contains {val!r}? {ok}"
    if ctype == "url_regex":
        ok = bool(re.search(val, cur_url or ""))
        return ok, f"final_url={cur_url!r} matches /{val}/? {ok}"
    if ctype == "text_contains":
        ok = val.lower() in (page_text or "").lower()
        return ok, f"page text contains {val!r}? {ok}  (url={cur_url!r})"
    return False, f"unknown success condition type: {ctype}"


def run_workflow_and_verify(workflow, condition, llm, use_human, user_data_dir):
    """Run one workflow in the browser and check its success condition.
    Returns (run_result, success_bool, detail, driver_final_url)."""
    from netgent import NetGent
    from netgent.utils.message import StatePrompt
    from netgent.browser.session import BrowserSession

    prompts = [StatePrompt(**s) for s in workflow]
    driver = BrowserSession(user_data_dir=user_data_dir).driver
    controller = None
    if use_human:
        from human_controller import HumanController
        controller = HumanController(driver)

    agent = NetGent(driver=driver, controller=controller, llm=llm, llm_enabled=True)
    result, ok, detail, final_url = {}, False, "", ""
    try:
        result = agent.run(state_prompts=prompts, state_repository=[])
    except Exception as e:
        result = {"error": str(e)}
    finally:
        try:
            ok, detail = check_success(condition, driver, result)
            final_url = driver.current_url
        except Exception as e:
            detail = f"success-check raised: {e}"
        try:
            agent.controller.quit()
        except Exception:
            pass
    return result, ok, detail, final_url


def process_prompt(item, llm, args, report_lines):
    sentence = item["sentence"]
    print("\n" + "#" * 78)
    print(f"# {sentence}")
    print("#" * 78)
    report_lines.append(f"\n## Task: {sentence}\n")
    report_lines.append(f"*Expected difficulty:* {item.get('notes','')}\n")

    record = {"sentence": sentence, "success_condition": item["success"], "stages": {}}

    # ---- 1. GENERATE ----
    t0 = time.time()
    workflow = generate_workflow(sentence, llm)
    gen_t = time.time() - t0
    print(f"[1/4 GENERATE] {len(workflow)} states in {gen_t:.1f}s")
    report_lines.append(f"**1. Generate:** {len(workflow)} states in {gen_t:.1f}s — "
                        f"{' -> '.join(s['name'] for s in workflow)}\n")
    record["stages"]["generate"] = {"workflow": workflow, "latency_s": gen_t}

    # ---- 2. LINT ----
    issues = lint_workflow(workflow, sentence=sentence)
    n_err = sum(1 for i in issues if i["level"] == "ERROR")
    print(f"[2/4 LINT] {n_err} errors, {len(issues)} total issues")
    report_lines.append(f"**2. Lint:** {'clean' if not issues else f'{n_err} errors / {len(issues)} issues'}\n")
    if issues:
        report_lines.append("```\n" + format_lint_report(issues) + "\n```\n")
    record["stages"]["lint"] = {"issues": issues, "n_errors": n_err}

    if args.dry_run:
        report_lines.append("*(dry-run: browser skipped)*\n")
        return record

    # ---- 3. RUN + 4. VERIFY, with optional self-heal ----
    attempt = 0
    while True:
        print(f"[3/4 RUN] attempt {attempt+1}")
        result, ok, detail, final_url = run_workflow_and_verify(
            workflow, item["success"], llm, not args.no_human, args.user_data_dir)
        report = stopgap_verify(result, workflow)
        print(f"[4/4 VERIFY] success={ok} | {detail}")
        print(f"           structural verdict (stopgap): failed={report['failed']} class={report['failure_class']}")

        record.setdefault("attempts", []).append({
            "attempt": attempt + 1, "task_success": ok, "success_detail": detail,
            "final_url": final_url, "stopgap_failed": report["failed"],
            "stopgap_class": report["failure_class"],
        })
        report_lines.append(f"**3-4. Run attempt {attempt+1}:** "
                            f"task_success=**{ok}** — {detail}\n")

        if ok:
            report_lines.append("Task VERIFIED SUCCESSFUL.\n")
            break
        if attempt >= args.max_repairs:
            report_lines.append(f"Failed after {attempt+1} attempt(s); self-heal budget exhausted.\n")
            break
        print(f"[heal] repairing ({report['failure_class']})...")
        workflow = diagnose_and_repair(sentence, workflow, report, llm)
        report_lines.append(f"*Self-heal ({report['failure_class']}): regenerated to "
                            f"{len(workflow)} states.*\n")
        attempt += 1

    record["final_task_success"] = ok
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-keys", required=True)
    ap.add_argument("--prompts-json", default=None, help="JSON list of {sentence,success,notes}")
    ap.add_argument("--outdir", default="results/exp2_end_to_end")
    ap.add_argument("--max-repairs", type=int, default=1)
    ap.add_argument("--no-human", action="store_true")
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache")
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="generation + lint only, no browser")
    args = ap.parse_args()

    battery = DEFAULT_BATTERY
    if args.prompts_json:
        with open(args.prompts_json) as f:
            battery = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    set_telemetry(Telemetry(outdir=os.path.join(args.outdir, "telemetry"),
                            meta={"experiment": "end_to_end", "n_prompts": len(battery)}))
    P.DEBUG = False

    from run_planner import build_llm
    llm = build_llm(args.api_keys, thinking_budget=args.thinking_budget)

    report_lines = [f"# Experiment 2 — End-to-end task accuracy\n",
                    f"{len(battery)} prompts, max_repairs={args.max_repairs}, "
                    f"human={'no' if args.no_human else 'yes'}\n"]
    records = []
    for item in battery:
        try:
            records.append(process_prompt(item, llm, args, report_lines))
        except Exception as e:
            print(f"  PROMPT CRASHED: {e}")
            report_lines.append(f"\n## {item['sentence']}\n**CRASHED:** {e}\n")

    # summary
    if not args.dry_run:
        n = len(records)
        succ = sum(1 for r in records if r.get("final_task_success"))
        first_try = sum(1 for r in records
                        if r.get("attempts") and r["attempts"][0]["task_success"])
        healed = sum(1 for r in records if r.get("final_task_success")
                     and r.get("attempts") and not r["attempts"][0]["task_success"])
        summary = (f"\n---\n# SUMMARY\n"
                   f"- Task success (final): **{succ}/{n}**\n"
                   f"- Success on first try (no self-heal): {first_try}/{n}\n"
                   f"- Recovered by self-healing: {healed}/{n}\n"
                   f"- Structural validity (lint clean at generation): "
                   f"{sum(1 for r in records if r['stages']['lint']['n_errors']==0)}/{n}\n")
        print(summary)
        report_lines.append(summary)
        report_lines.append("\n**Note:** self-heal here is driven by the STOPGAP verifier, "
                            "not Oliver's real one — label accordingly in the paper.\n")

    with open(os.path.join(args.outdir, "report.md"), "w") as f:
        f.write("\n".join(report_lines))
    with open(os.path.join(args.outdir, "records.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[done] {args.outdir}/report.md  and  records.json")


if __name__ == "__main__":
    main()
