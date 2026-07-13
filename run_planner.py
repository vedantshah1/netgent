#!/usr/bin/env python3
"""
run_planner.py
--------------
End-to-end driver for the self-healing NetGent planner.

    sentence (typed by user)
        -> generate_workflow            (Task 2: no human writes the workflow)
        -> run NetGent generation        (with the human-imitation controller, Task 4)
        -> if it stalls: diagnose_and_repair + rerun   (Task 3: self-healing)
        -> save prompts.json + the cached state repository

Usage on the VM (inside the netgent repo / container, foreground so output streams):

    python3 run_planner.py --api-keys api_keys.json -o out/planner_result.json
    python3 run_planner.py --api-keys api_keys.json --sentence "go to wikipedia, search bezier curves, scroll to the end" -o out/planner_result.json
    python3 run_planner.py --api-keys api_keys.json --no-human --max-repairs 3 -o out/planner_result.json

api_keys.json must contain: {"google_api_key": "AQ.Ab8RN6..."}   (same key NetGent -g uses)

Put this file (plus netgent_planner.py and human_controller.py) somewhere importable, e.g. the
repo root or alongside your other scripts, and scp them to the VM with:
    scp -P 2202 run_planner.py netgent_planner.py human_controller.py student@128.111.5.230:~/netgent/
"""

import argparse
import json
import sys
import langchain_google_genai

from netgent_planner import generate_workflow, diagnose_and_repair, detect_failure


def build_llm(api_keys_path):
    with open(api_keys_path) as f:
        keys = json.load(f)
    from langchain_google_genai import ChatGoogleGenerativeAI
    # temperature 0.2 matches NetGent's own generation default (cli.py)
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.2, api_key=keys["google_api_key"]
    )


def run_netgent_once(workflow_dicts, llm, use_human=True, user_data_dir=None):
    """Run one NetGent generation pass over a workflow. Returns the run result dict
    (or {'error': ...} if it raised) so the self-healing layer can react to failures."""
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
    """Generate a workflow from `sentence`, run it, and repair-and-retry on failure."""
    workflow = generate_workflow(sentence, llm)
    print(f"\n[planner] generated {len(workflow)} states")
    for s in workflow:
        print(f"   - {s['name']}")

    attempt = 0
    while True:
        print(f"\n[run] attempt {attempt + 1} (self-heal budget left: {max_repairs - attempt})")
        result = run_netgent_once(workflow, llm, use_human=use_human, user_data_dir=user_data_dir)
        failed, reason, reached = detect_failure(result, workflow)

        if not failed:
            print(f"[run] SUCCESS — {reason}")
            return workflow, result

        print(f"[run] FAILED — {reason}")
        if attempt >= max_repairs:
            print("[run] self-heal budget exhausted; returning last (failed) result.")
            return workflow, result

        print("[heal] diagnosing and regenerating a more specific workflow...")
        workflow = diagnose_and_repair(sentence, workflow, result, llm)
        print(f"[heal] new workflow has {len(workflow)} states")
        attempt += 1


def main():
    ap = argparse.ArgumentParser(description="Self-healing NetGent planner")
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
                    help="Only generate + print the workflow, do not launch a browser")
    args = ap.parse_args()

    sentence = args.sentence or input("Describe the task in one sentence:\n> ").strip()
    if not sentence:
        print("No task given."); sys.exit(1)

    llm = build_llm(args.api_keys)

    if args.dry_run:
        wf = generate_workflow(sentence, llm)
        print(json.dumps(wf, indent=2))
        _write(args.prompts_out, wf)
        print(f"\n[dry-run] workflow written to {args.prompts_out}")
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


def _write(path, obj):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    main()
