#!/usr/bin/env python3
"""
exp7_baseline_comparison.py
===========================
EXPERIMENT 7 — The headline comparison: NetGent as-published vs. this project.

MOTIVATION: every other experiment measures the planner against ITSELF (different budgets,
different prompt designs). None of them answers the question the paper actually has to answer:
*what does this project add to NetGent?* NetGent consumes hand-authored StatePrompt state
machines. This project generates them from one English sentence and repairs them when they fail.

THREE CONDITIONS, same tasks, same browser, same success conditions:

  A) baseline    -- hand-authored workflow, run once. This is NetGent as published.
                    Authoring cost = human effort. LLM generation tokens = 0.
  B) planner     -- NL sentence -> generated workflow, run once, NO repair.
                    Isolates contribution #1 (the planner) on its own.
  C) planner+heal-- NL sentence -> generated -> repair loop up to --max-repairs.
                    Isolates contribution #2 (self-healing) as the delta B -> C.

WHAT THIS BUYS THE PAPER:
  - task success:  baseline (ceiling) vs planner vs planner+heal
  - authoring cost: human minutes vs LLM tokens + seconds
  - the honest claim is a TRADE: the planner gives up some accuracy for ~zero authoring cost,
    and self-healing buys part of that accuracy back. Quantify all three numbers.

IMPORTANT — the baseline is only as good as the hand-authored workflows in --baseline-json.
Those define the ceiling, so author them carefully and record how long each took you
(--authoring-minutes). Do NOT copy a generated workflow into the baseline file: that makes the
comparison circular and the result meaningless.

USAGE:
    python3 exp7_baseline_comparison.py --api-keys keys.json \
        --battery-json hard_prompts_exp2.json \
        --baseline-json baseline_workflows.json \
        --max-repairs 2 --repeat 2 -o /out/exp7

OUTPUT: exp7_summary.csv, exp7_per_task.csv, report printed.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netgent_planner import (generate_workflow, diagnose_and_repair, lint_workflow,
                             stopgap_verify, make_report)
from exp_common import run_and_verify
from planner_telemetry import Telemetry, set_telemetry


def _tok(rec):
    """Pull token counts out of the telemetry singleton for the last N attempts."""
    return (rec or {}).get("input_tokens") or 0, (rec or {}).get("output_tokens") or 0


def run_baseline(item, baseline_wf, llm, uddir, authoring_minutes, iteration_count,
                 authoring_status):
    """Condition A: hand-authored workflow, run the FINAL authored version once.

    task_success here is expected to be ~True by construction -- a human-authored workflow was
    iterated on (via exp7_author_baseline.py) until it worked, or explicitly marked gave_up. The
    number that matters is NOT this run's success; it's authoring_minutes and iteration_count,
    which were measured live during authoring, not estimated after the fact.
    """
    issues = lint_workflow(baseline_wf, sentence=item["sentence"])
    n_err = sum(1 for i in issues if i["level"] == "ERROR")
    res = run_and_verify(baseline_wf, item["success"], llm, use_human=True, user_data_dir=uddir)
    if authoring_status == "gave_up" and not res["task_success"]:
        # Expected and informative, not an error: the human couldn't solve this either. Keep
        # it in the data -- it bounds what's achievable at all, independent of who authors it.
        pass
    return {
        "condition": "baseline",
        "sentence": item["sentence"],
        "struct_valid": n_err == 0,
        "lint_errors": n_err,
        "task_success": res["task_success"],
        "success_detail": res["success_detail"],
        "attempts_used": iteration_count or 1,   # real edit-run cycles, from the authoring log
        "state_count": len(baseline_wf),
        "gen_latency_s": 0.0,
        "gen_input_tokens": 0,
        "gen_output_tokens": 0,
        "authoring_minutes": authoring_minutes,
        "authoring_status": authoring_status,
        "run_latency_s": res["latency_s"],
    }


def run_planner(item, llm, uddir, max_repairs, tele):
    """Conditions B and C: generate, run, optionally repair. max_repairs=0 gives B."""
    sent = item["sentence"]
    n_before = len(tele._attempts)

    t0 = time.time()
    wf = generate_workflow(sent, llm)
    gen_latency = time.time() - t0

    issues = lint_workflow(wf, sentence=sent)
    n_err = sum(1 for i in issues if i["level"] == "ERROR")

    attempt = 1
    res = run_and_verify(wf, item["success"], llm, use_human=True, user_data_dir=uddir)
    while not res["task_success"] and attempt <= max_repairs:
        report = make_report(
            failed=True,
            failure_class=res["stopgap"].get("failure_class", "UNKNOWN"),
            reason=res["stopgap"].get("reason", ""),
            stuck_state=res["stopgap"].get("stuck_state"),
        )
        wf = diagnose_and_repair(sent, wf, report, llm)
        attempt += 1
        res = run_and_verify(wf, item["success"], llm, use_human=True, user_data_dir=uddir)

    # sum token cost of every LLM attempt this task consumed (generate + repairs)
    new_attempts = tele._attempts[n_before:]
    in_tok = sum((a.get("input_tokens") or 0) for a in new_attempts)
    out_tok = sum((a.get("output_tokens") or 0) for a in new_attempts)
    total_llm_latency = sum((a.get("latency_s") or 0) for a in new_attempts)

    return {
        "condition": "planner+heal" if max_repairs > 0 else "planner",
        "sentence": sent,
        "struct_valid": n_err == 0,
        "lint_errors": n_err,
        "task_success": res["task_success"],
        "success_detail": res["success_detail"],
        "attempts_used": attempt,
        "state_count": len(wf),
        "gen_latency_s": round(total_llm_latency or gen_latency, 3),
        "gen_input_tokens": in_tok,
        "gen_output_tokens": out_tok,
        "authoring_minutes": 0.0,
        "run_latency_s": res["latency_s"],
    }


def summarize(rows, condition):
    g = [r for r in rows if r["condition"] == condition]
    if not g:
        return None
    n = len(g)
    return {
        "condition": condition,
        "n": n,
        "task_success_rate": sum(1 for r in g if r["task_success"]) / n,
        "struct_valid_rate": sum(1 for r in g if r["struct_valid"]) / n,
        "mean_attempts": statistics.mean([r["attempts_used"] for r in g]),
        "mean_states": statistics.mean([r["state_count"] for r in g]),
        "mean_gen_latency_s": statistics.mean([r["gen_latency_s"] for r in g]),
        "mean_input_tokens": statistics.mean([r["gen_input_tokens"] for r in g]),
        "mean_output_tokens": statistics.mean([r["gen_output_tokens"] for r in g]),
        "total_authoring_minutes": sum(r["authoring_minutes"] for r in g),
        "mean_run_latency_s": statistics.mean([r["run_latency_s"] for r in g]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-keys", required=True)
    ap.add_argument("--battery-json", required=True)
    ap.add_argument("--baseline-json", required=True,
                    help="hand-authored workflows keyed by sentence (see baseline_workflows.json)")
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--conditions", default="baseline,planner,planner+heal")
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache")
    ap.add_argument("-o", "--outdir", default="results/exp7_baseline")
    args = ap.parse_args()

    from run_planner import build_llm
    llm = build_llm(args.api_keys)
    os.makedirs(args.outdir, exist_ok=True)
    tele = Telemetry(outdir=os.path.join(args.outdir, "telemetry"),
                     meta={"experiment": "baseline_comparison"})
    set_telemetry(tele)

    battery = json.load(open(args.battery_json))
    baseline = json.load(open(args.baseline_json))
    bl_by_sentence = {b["sentence"]: b for b in baseline}

    conditions = [c.strip() for c in args.conditions.split(",")]
    rows = []

    for item in battery:
        sent = item["sentence"]
        print("\n" + "#" * 78)
        print(f"# {sent[:74]}")
        print("#" * 78)

        for rep in range(args.repeat):
            if "baseline" in conditions:
                bl = bl_by_sentence.get(sent)
                if bl is None:
                    print(f"  [baseline] SKIP -- no hand-authored workflow for this sentence")
                elif bl.get("authoring_status") not in ("success", "gave_up"):
                    print(f"  [baseline] SKIP -- not yet authored via exp7_author_baseline.py "
                          f"(authoring_status={bl.get('authoring_status')!r}). "
                          f"Run that tool first so authoring_minutes/iteration_count are real, "
                          f"not guessed.")
                else:
                    r = run_baseline(item, bl["workflow"], llm, args.user_data_dir,
                                     bl.get("authoring_minutes", 0.0),
                                     bl.get("iteration_count", 1),
                                     bl.get("authoring_status"))
                    r["repeat"] = rep
                    rows.append(r)
                    print(f"  [baseline]     task={r['task_success']}  {r['state_count']}st  "
                          f"({r['attempts_used']} human iterations, "
                          f"{r['authoring_minutes']:.1f} min, status={r['authoring_status']})")

            if "planner" in conditions:
                r = run_planner(item, llm, args.user_data_dir, 0, tele)
                r["repeat"] = rep
                rows.append(r)
                print(f"  [planner]      task={r['task_success']}  {r['state_count']}st  "
                      f"{r['gen_input_tokens']}in/{r['gen_output_tokens']}out tok")

            if "planner+heal" in conditions:
                r = run_planner(item, llm, args.user_data_dir, args.max_repairs, tele)
                r["repeat"] = rep
                rows.append(r)
                print(f"  [planner+heal] task={r['task_success']}  {r['state_count']}st  "
                      f"{r['attempts_used']} attempt(s)  "
                      f"{r['gen_input_tokens']}in/{r['gen_output_tokens']}out tok")

    summaries = [s for s in (summarize(rows, c) for c in conditions) if s]

    print("\n\n" + "=" * 78)
    print("  EXPERIMENT 7 — NetGent baseline vs planner")
    print("  ('att' = human edit-run cycles for baseline; automated repair attempts for planner+heal)")
    print("=" * 78)
    print(f"  {'condition':<14}{'TASK':>8}{'struct':>8}{'att':>6}{'states':>8}"
          f"{'gen_s':>8}{'in_tok':>9}{'out_tok':>9}{'author_min':>12}")
    for s in summaries:
        print(f"  {s['condition']:<14}{s['task_success_rate']*100:>7.1f}%"
              f"{s['struct_valid_rate']*100:>7.1f}%{s['mean_attempts']:>6.1f}"
              f"{s['mean_states']:>8.2f}{s['mean_gen_latency_s']:>8.2f}"
              f"{s['mean_input_tokens']:>9.0f}{s['mean_output_tokens']:>9.0f}"
              f"{s['total_authoring_minutes']:>12.1f}")

    b = next((s for s in summaries if s["condition"] == "baseline"), None)
    p = next((s for s in summaries if s["condition"] == "planner"), None)
    h = next((s for s in summaries if s["condition"] == "planner+heal"), None)
    print("\n  DELTAS")
    if b and p:
        print(f"    planner vs baseline      : {(p['task_success_rate']-b['task_success_rate'])*100:+.1f} pp task success")
    if p and h:
        print(f"    self-healing contribution: {(h['task_success_rate']-p['task_success_rate'])*100:+.1f} pp task success")
    if b and h:
        print(f"    planner+heal vs baseline : {(h['task_success_rate']-b['task_success_rate'])*100:+.1f} pp task success")
    if b:
        print(f"    authoring cost avoided   : {b['total_authoring_minutes']:.1f} human-minutes "
              f"-> {(h or p or {}).get('mean_output_tokens', 0):.0f} output tokens/task")

    with open(os.path.join(args.outdir, "exp7_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    with open(os.path.join(args.outdir, "exp7_per_task.csv"), "w", newline="") as f:
        cols = ["condition", "sentence", "repeat", "task_success", "struct_valid", "lint_errors",
                "attempts_used", "state_count", "gen_latency_s", "gen_input_tokens",
                "gen_output_tokens", "authoring_minutes", "run_latency_s"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n[csv] {args.outdir}/exp7_summary.csv")
    print(f"[csv] {args.outdir}/exp7_per_task.csv")


if __name__ == "__main__":
    main()
