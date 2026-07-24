#!/usr/bin/env python3
"""
exp7_author_baseline.py
========================
Interactive tool for authoring Exp 7's hand-authored baseline workflows THE WAY YOU ACTUALLY
AUTHOR THEM -- write, run, see it fail, fix the JSON, run again -- while measuring the real cost
of that process instead of asking you to estimate it afterward.

WHY THIS EXISTS: a hand-authored NetGent workflow is not a single guess that succeeds or fails.
It's arrived at by iteration. Treating the baseline as "run it once, check task_success" measures
almost nothing -- you keep editing until it works, so task_success is close to 100% by
construction. The number that actually matters for the paper's Q1 ("does this remove the human
dependency of authoring workflows?") is the COST of that iteration: how many edit-run cycles, and
how many real minutes, it took a human to reach a working workflow. That's what this logs.

WORKFLOW:
    1. Loads (or creates a stub for) the workflow for one sentence in baseline_workflows.json
    2. Runs it in a real browser, shows you exactly what happened
    3. Prompts you to edit the JSON file directly, then press Enter to re-run -- or type 'done'
    4. Every attempt is timestamped and logged
    5. On 'done', computes authoring_minutes and iteration_count from the log and writes them
       back into baseline_workflows.json automatically

You can also type 'giveup' if a task turns out to be one a human wouldn't solve either (e.g. an
action the executor doesn't support) -- that is a legitimate, reportable outcome, not a failure
of the tool.

USAGE (run with -it -- this is interactive, it needs your terminal):
    python3 exp7_author_baseline.py --api-keys keys.json \
        --baseline-json baseline_workflows.json \
        --sentence "go to saucedemo.com, log in with..." \
        --user-data-dir /tmp/browser-cache

    # or iterate through every entry in the file that isn't finished yet:
    python3 exp7_author_baseline.py --api-keys keys.json --baseline-json baseline_workflows.json --all

OUTPUT: updates baseline_workflows.json in place (authoring_minutes, iteration_count,
        authoring_log, authoring_status) and appends to <baseline-json>.authoring_log.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import run_and_verify


def log_path(baseline_json):
    return baseline_json + ".authoring_log.jsonl"


def append_log(baseline_json, record):
    with open(log_path(baseline_json), "a") as f:
        f.write(json.dumps(record) + "\n")


def load_battery(baseline_json):
    return json.load(open(baseline_json))


def save_battery(baseline_json, battery):
    tmp = baseline_json + ".tmp"
    json.dump(battery, open(tmp, "w"), indent=2)
    os.replace(tmp, baseline_json)


def author_one(item, llm, uddir, baseline_json, battery):
    sent = item["sentence"]
    print("\n" + "#" * 78)
    print(f"# AUTHORING: {sent}")
    print("#" * 78)
    print("\nSUCCESS CONDITION (what your workflow must make true):")
    print("  " + json.dumps(item.get("success")))
    print("\nEdit this entry's \"workflow\" field directly in the JSON file between runs.")
    print(f"  file: {baseline_json}")
    print("Commands after each run: [Enter]=re-run same workflow after you've edited it, "
          "'done'=mark finished, 'giveup'=record as unsolved, 'skip'=leave for later\n")

    start = time.time()
    attempt = 0
    status = "in_progress"

    while True:
        # reload from disk every time, so your hand edits are picked up
        battery = load_battery(baseline_json)
        item = next(b for b in battery if b["sentence"] == sent)
        wf = item.get("workflow") or []
        if not wf:
            print("  [!] workflow is currently EMPTY -- edit the JSON to add at least one state, "
                  "then press Enter.")
            cmd = input("  > ").strip().lower()
            if cmd == "giveup":
                status = "gave_up"; break
            if cmd == "skip":
                status = "skipped"; break
            continue

        attempt += 1
        elapsed_min = (time.time() - start) / 60.0
        print(f"\n  --- attempt {attempt}  (elapsed {elapsed_min:.1f} min) ---")
        try:
            res = run_and_verify(wf, item["success"], llm, use_human=True, user_data_dir=uddir)
        except Exception as e:
            print(f"  RUN ERROR: {str(e)[:200]}")
            append_log(baseline_json, {"sentence": sent, "attempt": attempt,
                                       "t": time.time(), "error": str(e)[:300]})
            cmd = input("  Fix and press Enter to retry, or 'giveup'/'skip': ").strip().lower()
            if cmd == "giveup":
                status = "gave_up"; break
            if cmd == "skip":
                status = "skipped"; break
            continue

        print(f"  task_success = {res['task_success']}")
        print(f"  detail       = {res['success_detail']}")
        print(f"  states       = {len(wf)}   latency = {res['latency_s']:.1f}s")

        append_log(baseline_json, {
            "sentence": sent, "attempt": attempt, "t": time.time(),
            "elapsed_min": round(elapsed_min, 2), "task_success": res["task_success"],
            "success_detail": res["success_detail"], "state_count": len(wf),
        })

        if res["task_success"]:
            print("\n  ✓ SUCCEEDED. Type 'done' to lock this in, or Enter to keep refining "
                  "(e.g. if you want to simplify it), or 'giveup' is not applicable now.")
        cmd = input("  > ").strip().lower()
        if cmd == "done":
            status = "success" if res["task_success"] else "success_but_marked_done_while_failing"
            break
        if cmd == "giveup":
            status = "gave_up"; break
        if cmd == "skip":
            status = "skipped"; break
        # anything else (including blank Enter): loop, re-read the file, re-run

    total_min = round((time.time() - start) / 60.0, 2)
    battery = load_battery(baseline_json)
    item = next(b for b in battery if b["sentence"] == sent)
    item["authoring_minutes"] = total_min
    item["iteration_count"] = attempt
    item["authoring_status"] = status
    save_battery(baseline_json, battery)

    print(f"\n  >>> {sent[:50]}")
    print(f"  >>> status={status}  attempts={attempt}  authoring_minutes={total_min}")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-keys", required=True)
    ap.add_argument("--baseline-json", required=True)
    ap.add_argument("--sentence", default=None, help="author just this one entry")
    ap.add_argument("--all", action="store_true",
                    help="iterate every entry not already marked success/gave_up")
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache")
    args = ap.parse_args()

    if not args.sentence and not args.all:
        print("Specify --sentence \"...\" for one entry, or --all to go through every "
              "unfinished entry.")
        return 1

    from run_planner import build_llm
    llm = build_llm(args.api_keys)

    battery = load_battery(args.baseline_json)
    battery = [b for b in battery if not b["sentence"].startswith("_TEMPLATE")]

    if args.sentence:
        targets = [b for b in battery if b["sentence"] == args.sentence]
        if not targets:
            print("No entry with that exact sentence in the baseline file.")
            return 1
    else:
        targets = [b for b in battery
                  if b.get("authoring_status") not in ("success", "gave_up")]
        if not targets:
            print("Every entry is already marked success or gave_up. Nothing to do.")
            return 0

    print(f"{len(targets)} entr{'y' if len(targets)==1 else 'ies'} to author.\n")
    results = {}
    for item in targets:
        status = author_one(item, llm, args.user_data_dir, args.baseline_json, battery)
        results[item["sentence"][:50]] = status

    print("\n\n" + "=" * 78)
    print("  SESSION SUMMARY")
    print("=" * 78)
    for s, st in results.items():
        print(f"  {st:12} {s}")

    battery = load_battery(args.baseline_json)
    total_min = sum(b.get("authoring_minutes", 0) or 0 for b in battery)
    total_iter = sum(b.get("iteration_count", 0) or 0 for b in battery)
    n_success = sum(1 for b in battery if b.get("authoring_status") == "success")
    n_gave_up = sum(1 for b in battery if b.get("authoring_status") == "gave_up")
    print(f"\n  totals so far: {total_min:.1f} human-minutes, {total_iter} iterations, "
          f"{n_success} authored successfully, {n_gave_up} given up on")
    print(f"\n  {args.baseline_json} updated in place.")
    print(f"  full attempt log: {log_path(args.baseline_json)}")


if __name__ == "__main__":
    main()
