#!/usr/bin/env python3
"""
exp7_try_once.py -- run ONE attempt at a hand-authored baseline workflow, report, and log.

Replaces the interactive exp7_author_baseline.py. That version needed a TTY (-it), a long-lived
container, and an input() loop; too many moving parts, and when any of them broke the failure was
silent. This does exactly one thing per invocation and then exits.

THE LOOP YOU ACTUALLY RUN:
    1. edit the "workflow" field for your task in baseline_workflows.json
    2. run this script  -> it runs that workflow once and tells you what happened
    3. repeat until it passes
    4. run it once more with --done  -> records authoring_minutes + iteration_count

Timing and iteration count are still measured for you: every attempt is appended to
baseline_workflows.json.authoring_log.jsonl with a timestamp, and --done computes elapsed
minutes from the FIRST attempt to the last. Nothing to estimate by hand.

USAGE:
    python3 exp7_try_once.py --api-keys /keys.json --baseline-json /baseline.json --list
    python3 exp7_try_once.py --api-keys /keys.json --baseline-json /baseline.json --task 1
    python3 exp7_try_once.py --api-keys /keys.json --baseline-json /baseline.json --task 1 --done
    python3 exp7_try_once.py --api-keys /keys.json --baseline-json /baseline.json --task 1 --giveup

--task takes an INDEX (from --list), so you never have to paste the long sentence or worry
about matching it byte-for-byte.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def log_path(baseline_json):
    return baseline_json + ".authoring_log.jsonl"


def read_log(baseline_json, sentence):
    p = log_path(baseline_json)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("sentence") == sentence:
                out.append(r)
    return out


SESSION_GAP_SEC = 20 * 60   # gaps longer than this = a break (sleep, meal, etc.), not active work


def active_minutes(records, manual_minutes):
    """Sum only the time BETWEEN consecutive attempts, and only when that gap looks like real
    editing time rather than a break. A naive (last - first) span silently counts overnight
    gaps as authoring time, which is wrong the moment a session spans a sleep."""
    ts = sorted(r["t"] for r in records)
    active_sec = 0.0
    for i in range(1, len(ts)):
        gap = ts[i] - ts[i - 1]
        if gap <= SESSION_GAP_SEC:
            active_sec += gap
        # else: treated as a break, not counted
    return round(active_sec / 60.0 + manual_minutes, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-json", required=True)
    ap.add_argument("--api-keys")
    ap.add_argument("--task", type=int, help="task index from --list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--done", action="store_true", help="mark this task finished")
    ap.add_argument("--giveup", action="store_true", help="mark this task unsolved by hand")
    ap.add_argument("--add-session-minutes", type=float, default=None,
                    help="log real minutes worked THIS session by hand -- use this when the "
                         "automated attempt log under-counts (e.g. you spent most of the time "
                         "editing/debugging JSON without a single successful script invocation "
                         "to log a timestamp). Does NOT mark the task done or gave_up.")
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache")
    args = ap.parse_args()

    battery = json.load(open(args.baseline_json))
    battery = [b for b in battery if not b["sentence"].startswith("_TEMPLATE")]

    # ---------------- --list ---------------- #
    if args.list or (args.task is None and args.add_session_minutes is None):
        print(f"\nTasks in {args.baseline_json}:\n")
        for i, b in enumerate(battery, 1):
            wf = b.get("workflow") or []
            st = b.get("authoring_status", "not_started")
            n_att = len(read_log(args.baseline_json, b["sentence"]))
            manual = b.get("manual_minutes", 0.0)
            print(f"  [{i}] {st:12}  states={len(wf):<2} attempts_so_far={n_att}"
                  f"  manual_minutes_logged={manual}")
            print(f"      {b['sentence'][:88]}")
            print(f"      success: {json.dumps(b.get('success'))}")
            print()
        if args.task is None:
            print("Pick one with --task N")
        return 0

    if not (1 <= args.task <= len(battery)):
        print(f"--task must be between 1 and {len(battery)}. Use --list to see them.")
        return 1
    item = battery[args.task - 1]
    sent = item["sentence"]

    # ---------------- --add-session-minutes (no browser, no display needed) ---------------- #
    if args.add_session_minutes is not None:
        full = json.load(open(args.baseline_json))
        for b in full:
            if b["sentence"] == sent:
                b["manual_minutes"] = round(b.get("manual_minutes", 0.0)
                                            + args.add_session_minutes, 2)
                total_manual = b["manual_minutes"]
        json.dump(full, open(args.baseline_json, "w"), indent=2)
        print(f"\n  logged {args.add_session_minutes} manual minutes for task {args.task}")
        print(f"  total manual_minutes so far: {total_manual}")
        print(f"  status unchanged -- this task is still open, not marked done/gave_up.")
        print(f"  This will be added to the log-derived active time when you eventually")
        print(f"  run --done or --giveup.")
        return 0

    # ---------------- validate ---------------- #
    cond = item.get("success")
    if not isinstance(cond, dict) or "type" not in cond:
        print(f"[X] CONFIG ERROR: task {args.task} has no usable \"success\" field (found "
              f"{cond!r}). Nothing was tested.")
        return 1

    prior = read_log(args.baseline_json, sent)

    # ---------------- --done / --giveup ---------------- #
    if args.done or args.giveup:
        manual = item.get("manual_minutes", 0.0)
        if not prior and not manual:
            print("[X] No attempts logged and no manual minutes recorded for this task yet --")
            print("    run at least one attempt, or use --add-session-minutes, first.")
            return 1
        minutes = active_minutes(prior, manual)
        full = json.load(open(args.baseline_json))
        for b in full:
            if b["sentence"] == sent:
                b["authoring_minutes"] = minutes
                b["iteration_count"] = len(prior)
                b["authoring_status"] = "gave_up" if args.giveup else "success"
        json.dump(full, open(args.baseline_json, "w"), indent=2)
        print(f"\n  marked {'GAVE UP' if args.giveup else 'DONE'}")
        print(f"  iteration_count   = {len(prior)}")
        print(f"  authoring_minutes = {minutes}  "
              f"(active-session time; breaks longer than {SESSION_GAP_SEC//60}min excluded"
              f"{f', includes {manual} manual min' if manual else ''})")
        return 0

    # ---------------- run one attempt ---------------- #
    wf = item.get("workflow") or []
    print("=" * 72)
    print(f"  TASK {args.task}  (attempt {len(prior) + 1})")
    print("=" * 72)
    print(f"  {sent[:200]}")
    print(f"\n  success condition: {json.dumps(cond)}")
    print(f"  workflow states:   {len(wf)}")

    if not wf:
        print(f"\n[X] The \"workflow\" field is empty. Edit {args.baseline_json}, add your states,")
        print(f"    then run this again. Nothing was tested.")
        return 1

    print("\n  running in a real browser (this takes 10-60s)...\n")
    t0 = time.time()
    try:
        from exp_common import run_and_verify
        from run_planner import build_llm
        llm = build_llm(args.api_keys) if args.api_keys else None
        res = run_and_verify(wf, cond, llm, use_human=True, user_data_dir=args.user_data_dir)
        ok = bool(res["task_success"])
        detail = res["success_detail"]
        err = None
    except Exception as e:
        ok, detail, err = False, None, f"{type(e).__name__}: {e}"
        print("[X] RUN CRASHED -- nothing was tested this attempt.")
        traceback.print_exc()

    dt = time.time() - t0
    rec = {"sentence": sent, "t": time.time(), "attempt": len(prior) + 1,
           "task_success": ok, "detail": detail, "error": err,
           "state_count": len(wf), "run_seconds": round(dt, 1)}
    with open(log_path(args.baseline_json), "a") as f:
        f.write(json.dumps(rec) + "\n")

    print("\n" + "-" * 72)
    if err:
        print(f"  RESULT: CRASHED ({dt:.1f}s)")
        print(f"  {err}")
    elif ok:
        print(f"  RESULT: SUCCESS ({dt:.1f}s)")
        print(f"  {detail}")
        print(f"\n  Lock it in:  --task {args.task} --done")
    else:
        print(f"  RESULT: FAILED ({dt:.1f}s)")
        print(f"  {detail}")
        print(f"\n  Edit the workflow in {args.baseline_json} and run this again.")
    print(f"  attempts so far: {len(prior) + 1}")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
