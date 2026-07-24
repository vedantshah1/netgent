#!/usr/bin/env python3
"""
exp8_controller_ab.py
=====================
EXPERIMENT 8 — What does the human-imitation layer actually cost and buy?

MOTIVATION: Exp 4 compared stock vs HumanController using traces synthesized from the
controllers' own math, with no browser involved. That is close to tautological -- it confirms the
imitation code does what it was written to do. What has never been measured is the thing the
paper needs:

  1. Does the human layer change TASK SUCCESS on real pages?  (does it help, or just cost time?)
  2. What is the WALL-CLOCK OVERHEAD of the imitation?        (the honest cost of contribution #3)
  3. Do real captured traces reproduce the motor statistics?  (Exp 4 was synthetic; this is not)

DESIGN: identical workflows, identical success conditions, run twice -- once with NetGent's stock
controller, once with HumanController -- alternating order to avoid drift from site state or
caching. Traces are captured per-condition when HUMAN_TRACE is honoured, so the Exp 4 feature
extractor can be re-run on REAL movement instead of synthetic movement.

NOTE ON INTERPRETATION: if task success is identical between conditions, that is a GOOD result to
report honestly -- it means the imitation layer is free in accuracy terms and its justification
rests on detection-surface reduction, not task performance. Do not oversell a null.

USAGE:
    python3 exp8_controller_ab.py --api-keys keys.json \
        --battery-json hard_prompts_exp2.json --repeat 2 -o /out/exp8

    # reuse fixed workflows so generation variance does not contaminate the A/B
    python3 exp8_controller_ab.py --api-keys keys.json \
        --battery-json hard_prompts_exp2.json \
        --workflows-json baseline_workflows.json -o /out/exp8

OUTPUT: exp8_summary.csv, exp8_per_run.csv, trace files under <outdir>/traces/.
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netgent_planner import generate_workflow, lint_workflow
from exp_common import run_and_verify
from planner_telemetry import Telemetry, set_telemetry


def stock_factory(driver):
    """NetGent's stock controller -- the un-humanized baseline."""
    from netgent.browser.controller import PyAutoGUIController
    return PyAutoGUIController(driver)


def human_factory(driver):
    from human_controller import HumanController
    return HumanController(driver)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-keys", required=True)
    ap.add_argument("--battery-json", required=True)
    ap.add_argument("--workflows-json", default=None,
                    help="optional fixed workflows keyed by sentence; avoids generation variance")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache")
    ap.add_argument("-o", "--outdir", default="results/exp8_controller_ab")
    args = ap.parse_args()

    from run_planner import build_llm
    llm = build_llm(args.api_keys)
    os.makedirs(args.outdir, exist_ok=True)
    tracedir = os.path.join(args.outdir, "traces")
    os.makedirs(tracedir, exist_ok=True)
    set_telemetry(Telemetry(outdir=os.path.join(args.outdir, "telemetry"),
                            meta={"experiment": "controller_ab"}))

    battery = json.load(open(args.battery_json))
    fixed = {}
    if args.workflows_json:
        for b in json.load(open(args.workflows_json)):
            if b.get("workflow"):
                fixed[b["sentence"]] = b["workflow"]

    rows = []
    for item in battery:
        sent = item["sentence"]
        print("\n" + "#" * 78)
        print(f"# {sent[:74]}")
        print("#" * 78)

        # One workflow per task, shared by BOTH conditions, so the only variable is the
        # controller. Generating separately per condition would confound the comparison.
        if sent in fixed:
            wf = fixed[sent]
            print(f"  [workflow] fixed, {len(wf)} states")
        else:
            wf = generate_workflow(sent, llm)
            print(f"  [workflow] generated, {len(wf)} states")
        n_err = sum(1 for i in lint_workflow(wf, sentence=sent) if i["level"] == "ERROR")

        for rep in range(args.repeat):
            # alternate which controller goes first, so warm-cache / site-state drift
            # does not systematically favour one condition
            order = [("stock", stock_factory), ("human", human_factory)]
            if rep % 2 == 1:
                order.reverse()

            for cond, factory in order:
                trace_path = os.path.join(tracedir, f"{cond}_rep{rep}_{abs(hash(sent)) % 10**8}.jsonl")
                os.environ["HUMAN_TRACE"] = trace_path
                try:
                    res = run_and_verify(wf, item["success"], llm,
                                         user_data_dir=args.user_data_dir,
                                         controller_factory=factory)
                except Exception as e:
                    print(f"  [{cond:5}] rep{rep} EXCEPTION: {str(e)[:80]}")
                    continue
                rows.append({
                    "condition": cond,
                    "sentence": sent,
                    "repeat": rep,
                    "struct_valid": n_err == 0,
                    "task_success": res["task_success"],
                    "success_detail": res["success_detail"],
                    "run_latency_s": res["latency_s"],
                    "state_count": len(wf),
                    "trace_path": trace_path if os.path.exists(trace_path) else "",
                })
                print(f"  [{cond:5}] rep{rep} task={res['task_success']} "
                      f"{res['latency_s']:.1f}s")

    # ---------------- summary ---------------- #
    summaries = []
    for cond in ("stock", "human"):
        g = [r for r in rows if r["condition"] == cond]
        if not g:
            continue
        summaries.append({
            "condition": cond,
            "n": len(g),
            "task_success_rate": sum(1 for r in g if r["task_success"]) / len(g),
            "mean_run_latency_s": statistics.mean([r["run_latency_s"] for r in g]),
            "median_run_latency_s": statistics.median([r["run_latency_s"] for r in g]),
            "traces_captured": sum(1 for r in g if r["trace_path"]),
        })

    print("\n\n" + "=" * 78)
    print("  EXPERIMENT 8 — stock vs HumanController")
    print("=" * 78)
    print(f"  {'condition':<12}{'n':>5}{'TASK':>9}{'mean_s':>10}{'median_s':>11}{'traces':>9}")
    for s in summaries:
        print(f"  {s['condition']:<12}{s['n']:>5}{s['task_success_rate']*100:>8.1f}%"
              f"{s['mean_run_latency_s']:>10.2f}{s['median_run_latency_s']:>11.2f}"
              f"{s['traces_captured']:>9}")

    if len(summaries) == 2:
        st, hu = summaries[0], summaries[1]
        d_task = (hu["task_success_rate"] - st["task_success_rate"]) * 100
        ovh = (hu["mean_run_latency_s"] / st["mean_run_latency_s"] - 1) * 100 \
            if st["mean_run_latency_s"] else float("nan")
        print(f"\n  task-success delta (human - stock): {d_task:+.1f} pp")
        print(f"  wall-clock overhead of imitation  : {ovh:+.1f}%")
        print("\n  Reminder: a task-success delta near zero is a legitimate finding. It means the")
        print("  imitation layer is accuracy-neutral and must be justified on detection grounds.")

    with open(os.path.join(args.outdir, "exp8_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    with open(os.path.join(args.outdir, "exp8_per_run.csv"), "w", newline="") as f:
        cols = ["condition", "sentence", "repeat", "struct_valid", "task_success",
                "run_latency_s", "state_count", "trace_path"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n[csv] {args.outdir}/exp8_summary.csv")
    print(f"[csv] {args.outdir}/exp8_per_run.csv")
    print(f"[traces] {tracedir}/  -> feed these to exp4 with --from-traces for REAL motor stats")


if __name__ == "__main__":
    main()
