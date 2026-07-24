#!/usr/bin/env python3
"""
run_planner.py
--------------
End-to-end driver for the NetGent planner.

    sentence (typed by user)
        -> generate_workflow                      (sentence -> NetGent NFA workflow)
        -> lint_workflow                           (static check before ever touching a browser)
        -> run NetGent generation                  (with the human-imitation controller)
        -> if it fails: stopgap_verify + diagnose_and_repair + rerun   (placeholder self-healing)
        -> save prompts.json + the cached state repository + telemetry

SCOPE NOTE: detection/verification is Oliver's component (not built yet). This script uses
stopgap_verify() as a placeholder so the pipeline is runnable end-to-end -- every result it
produces is tagged source="stopgap_heuristic" in telemetry. Once Oliver's verifier exists,
swap the stopgap_verify() call below for his agent's report and nothing else needs to change,
since diagnose_and_repair() already speaks the VerificationReport interface.

Usage on the VM (inside the netgent repo / container, foreground so output streams):

    python3 run_planner.py --api-keys api_keys.json -o out/planner_result.json
    python3 run_planner.py --api-keys api_keys.json --sentence "go to wikipedia, search bezier curves, scroll to the end" -o out/planner_result.json
    python3 run_planner.py --api-keys api_keys.json --no-human --max-repairs 3 -o out/planner_result.json
    python3 run_planner.py --api-keys api_keys.json --dry-run --sentence "..."   # generation only, no browser

api_keys.json must contain: {"google_api_key": "AQ.Ab8RN6..."}   (same key NetGent -g uses)

Recording (set before running, or pass -e in Docker):
    export PLANNER_TELEMETRY_DIR=/out/telemetry     # every LLM call + attempt, JSONL
    export HUMAN_TRACE=/out/human_trace.jsonl       # every mouse/key sample from HumanController

Put this file (plus netgent_planner.py, human_controller.py, planner_telemetry.py) somewhere
importable, e.g. the repo root, and scp to the VM with:
    scp -P 2202 run_planner.py netgent_planner.py human_controller.py planner_telemetry.py \
        student@128.111.5.230:~/netgent/
"""

import argparse
import json
import os
import sys

from netgent_planner import (generate_workflow, diagnose_and_repair, lint_workflow,
                             format_lint_report, stopgap_verify)
from planner_telemetry import Telemetry, set_telemetry


def build_llm(api_keys_path, model="gemini-2.5-flash", temperature=0.2, thinking_budget=None):
    """Build the Gemini chat model.

    thinking_budget: gemini-2.5-flash reasoning control (an int token cap).
        None -> library default (dynamic thinking)
        0    -> attempt to DISABLE thinking (some langchain-google-genai versions ignore 0 and
                still spend reasoning tokens -- issue #928 -- so ALWAYS verify against measured
                reasoning tokens in telemetry, never assume the setting took effect)
        -1   -> explicit dynamic thinking
    """
    with open(api_keys_path) as f:
        keys = json.load(f)
    from langchain_google_genai import ChatGoogleGenerativeAI
    kwargs = dict(model=model, temperature=temperature, api_key=keys["google_api_key"])
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    return ChatGoogleGenerativeAI(**kwargs)


def run_netgent_once(workflow_dicts, llm, use_human=True, user_data_dir=None):
    """Run one NetGent generation pass over a workflow. Returns the raw run_result dict
    (or {'error': ...} if it raised) for stopgap_verify() to interpret."""
    from netgent import NetGent
    from netgent.utils.message import StatePrompt
    from netgent.browser.session import BrowserSession

    prompts = [StatePrompt(**s) for s in workflow_dicts]
    driver = BrowserSession(user_data_dir=user_data_dir).driver

    if use_human:
        from human_controller import HumanController
        controller = HumanController(driver)
    else:
        controller = None  # NetGent falls back to the stock PyAutoGUIController

    agent = NetGent(driver=driver, controller=controller, llm=llm, llm_enabled=True)
    try:
        return agent.run(state_prompts=prompts, state_repository=[])
    except Exception as e:                       # crash / token limit / captcha bail-out
        return {"error": str(e)}
    finally:
        try:
            agent.controller.quit()
        except Exception:
            pass


def run_with_self_healing(sentence, llm, max_repairs=2, use_human=True, user_data_dir=None):
    """Generate a workflow, lint it, run it, and repair-and-retry on failure using the
    STOPGAP verifier (see module docstring -- swap for the real verifier when it lands)."""
    workflow = generate_workflow(sentence, llm)
    print(f"\n[planner] generated {len(workflow)} states")
    for s in workflow:
        print(f"   - {s['name']}")

    issues = lint_workflow(workflow, sentence=sentence)
    if issues:
        print("\n[lint] issues found in the generated workflow BEFORE running it:")
        print(format_lint_report(issues))
        if any(i["level"] == "ERROR" for i in issues):
            print("[lint] ERROR-level issue present -- this will very likely fail. "
                  "Continuing anyway so you can see what happens.")
    else:
        print("[lint] no issues found.")

    attempt = 0
    while True:
        print(f"\n[run] attempt {attempt + 1} (self-heal budget left: {max_repairs - attempt})")
        result = run_netgent_once(workflow, llm, use_human=use_human, user_data_dir=user_data_dir)

        # PLACEHOLDER: real detection/verification is Oliver's component. Tagged
        # source="stopgap_heuristic" in telemetry so it's never confused with his output.
        report = stopgap_verify(result, workflow)
        print(f"[verify:stopgap] failed={report['failed']}  class={report['failure_class']}  "
              f"reason={report['reason']}")

        if not report["failed"]:
            print(f"[run] SUCCESS")
            return workflow, result

        if attempt >= max_repairs:
            print("[run] self-heal budget exhausted; returning last (failed) result.")
            return workflow, result

        print(f"[heal] repairing for failure class {report['failure_class']}...")
        workflow = diagnose_and_repair(sentence, workflow, report, llm)
        print(f"[heal] new workflow has {len(workflow)} states")
        attempt += 1


def main():
    ap = argparse.ArgumentParser(description="NetGent planner driver")
    ap.add_argument("--api-keys", required=True, help="JSON file with {'google_api_key': ...}")
    ap.add_argument("--sentence", default=None, help="Task sentence (prompted if omitted)")
    ap.add_argument("-o", "--output", default="out/planner_result.json",
                    help="Where to write the resulting state repository")
    ap.add_argument("--prompts-out", default="out/generated_prompts.json",
                    help="Where to write the generated workflow (NetGent prompts.json)")
    ap.add_argument("--max-repairs", type=int, default=2, help="Self-heal retry budget")
    ap.add_argument("--no-human", action="store_true", help="Use stock controller, not HumanController")
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache", help="Browser cache dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only generate + lint + print the workflow, do not launch a browser")
    ap.add_argument("--telemetry-dir", default=None,
                    help="Override PLANNER_TELEMETRY_DIR for this run")
    args = ap.parse_args()

    sentence = args.sentence or input("Describe the task in one sentence:\n> ").strip()
    if not sentence:
        print("No task given."); sys.exit(1)

    tel_dir = args.telemetry_dir or os.environ.get("PLANNER_TELEMETRY_DIR")
    tel = set_telemetry(Telemetry(outdir=tel_dir, meta={"driver": "run_planner.py",
                                                        "sentence": sentence}))
    if tel_dir:
        print(f"[telemetry] recording to {tel_dir}/")
    else:
        print("[telemetry] PLANNER_TELEMETRY_DIR not set -- nothing will be recorded. "
              "Set it (or pass --telemetry-dir) to keep prompts/outputs/tokens/latency.")

    llm = build_llm(args.api_keys)

    if args.dry_run:
        wf = generate_workflow(sentence, llm)
        print(json.dumps(wf, indent=2))
        issues = lint_workflow(wf, sentence=sentence)
        print("\n" + (format_lint_report(issues) if issues else "[lint] no issues found."))
        _write(args.prompts_out, wf)
        print(f"\n[dry-run] workflow written to {args.prompts_out}")
        tel.close(summary={"mode": "dry_run", "state_count": len(wf)})
        return

    workflow, result = run_with_self_healing(
        sentence, llm,
        max_repairs=args.max_repairs,
        use_human=not args.no_human,
        user_data_dir=args.user_data_dir,
    )

    _write(args.prompts_out, workflow)
    repo = result.get("state_repository", result) if isinstance(result, dict) else result
    _write(args.output, repo)
    print(f"\n[done] workflow -> {args.prompts_out}")
    print(f"[done] state repository -> {args.output}")
    print("You can now replay this deterministically with:  netgent -e "
          f"{args.output} -o out/replay.json -s")

    tel.close(summary={"mode": "self_healing", "final_state_count": len(workflow)})


def _write(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    main()
