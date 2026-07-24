#!/usr/bin/env python3
"""
exp1_thinking_budget.py
=======================
EXPERIMENT 1 — Does Gemini 2.5's "thinking" actually help workflow generation, and what
does it cost?

MOTIVATION (from RESULTS_MATRIX.md, Table 3b): on the real run, 70-86% of output tokens were
internal reasoning, not workflow text -- i.e. most of the latency and cost is thinking. If
turning thinking down/off doesn't hurt structural validity or consistency, that's a large,
cheap efficiency win and a clean paper result.

WHAT THIS TESTS (and why each piece is here):
  - INDEPENDENT variable: thinking_budget in {0, 512, 2048, -1(dynamic/default)}.
  - DEPENDENT variables, per budget:
      * structural validity rate      (does quality hold?)          <- lint, no browser
      * JSON-fixup rate               (does bad-JSON get worse?)
      * consistency (state-count + Jaccard over repeats)            <- does thinking stabilize it?
      * MEASURED reasoning tokens     (did the budget actually take effect? issue #928)
      * output tokens, total tokens, cost
      * latency (mean/median/p95)
  - CONTROLS: identical prompt battery, identical temperature, identical repeats across all
    budgets. Only thinking_budget changes. Same battery is reused so differences are attributable
    to the budget, not the prompts.

IMPORTANT: This measures GENERATION quality only (structural), NOT task success -- no browser.
That's deliberate: it isolates the effect of thinking on the planner's output, one variable at
a time. Task-success under different budgets is a follow-up (exp is browser-bound and slower).

USAGE:
    python3 exp1_thinking_budget.py --api-keys api_keys.json --prompts-file hard_prompts.txt --repeat 3
    python3 exp1_thinking_budget.py --api-keys api_keys.json --budgets 0 2048 -1 --repeat 5

OUTPUT: results/exp1_thinking/<budget>/telemetry/... plus a printed comparison table and
exp1_summary.csv you can drop straight into the paper.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netgent_planner import generate_workflow, lint_workflow
from planner_telemetry import Telemetry, set_telemetry
import netgent_planner as P


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def run_budget(budget, prompts, repeat, api_keys, outroot, model, browser=False,
               battery=None, user_data_dir="/tmp/browser-cache"):
    """Run the full battery at one thinking_budget. Returns an aggregate stats dict.

    If browser=True, also RUNS each generated workflow and checks real task success, so the
    thinking-budget effect is measured on task accuracy (ground truth), not only structural
    validity. `battery` (list of {sentence,success,...}) is required for browser mode so each
    task has a checkable success condition.
    """
    from run_planner import build_llm   # reuse the single source of truth for model construction
    llm = build_llm(api_keys, model=model, thinking_budget=budget)

    label = "default" if budget is None else ("dynamic" if budget == -1 else str(budget))
    tel_dir = os.path.join(outroot, f"budget_{label}", "telemetry")
    tel = set_telemetry(Telemetry(outdir=tel_dir, meta={"experiment": "thinking_budget",
                                                        "thinking_budget": budget, "model": model,
                                                        "browser": browser}))
    print(f"\n{'='*70}\nTHINKING BUDGET = {label}  ({'BROWSER' if browser else 'structural-only'})\n{'='*70}")

    per_attempt = []
    groups = defaultdict(list)   # sentence -> list of state_name lists
    P.DEBUG = False

    # in browser mode we iterate the battery (has success conditions); else plain sentences
    items = battery if browser else [{"sentence": s} for s in prompts]

    for item in items:
        sent = item["sentence"]
        for r in range(repeat):
            t0 = time.time()
            try:
                wf = generate_workflow(sent, llm)
                issues = lint_workflow(wf, sentence=sent)
                n_err = sum(1 for i in issues if i["level"] == "ERROR")
                rec = {"parse_ok": True, "n_err": n_err, "state_count": len(wf),
                       "latency": time.time() - t0, "task_success": None}

                if browser:
                    from exp_common import run_and_verify
                    res = run_and_verify(wf, item["success"], llm,
                                         use_human=False, user_data_dir=user_data_dir)
                    rec["task_success"] = res["task_success"]
                    print(f"  [{label}] struct={'OK' if n_err==0 else 'ERR'} "
                          f"task_success={res['task_success']} {len(wf)}st \"{sent[:38]}\"")
                else:
                    print(f"  [{label}] {'PASS' if n_err==0 else 'FAIL'} "
                          f"{len(wf)} states {time.time()-t0:.1f}s \"{sent[:44]}\"")

                per_attempt.append(rec)
                groups[sent].append([s["name"] for s in wf])
            except Exception as e:
                per_attempt.append({"parse_ok": False, "n_err": 0, "state_count": None,
                                    "latency": time.time() - t0, "task_success": False})
                print(f"  [{label}] EXCEPTION: {str(e)[:80]}")

    tel.close()

    # pull the token/latency truth from telemetry (measured, not assumed)
    calls = [json.loads(l) for l in open(os.path.join(tel_dir, "llm_calls.jsonl"))] \
        if os.path.exists(os.path.join(tel_dir, "llm_calls.jsonl")) else []

    def col(key):
        return [c[key] for c in calls if c.get(key) is not None]

    lat = [a["latency"] for a in per_attempt]
    reasoning = col("reasoning_tokens")
    out_tok = col("output_tokens")
    in_tok = col("input_tokens")

    # consistency: mean state-count stability + mean Jaccard, across repeated prompts
    sc_stable, jacc = [], []
    for sent, names in groups.items():
        if len(names) > 1:
            counts = [len(n) for n in names]
            sc_stable.append(1.0 if len(set(counts)) == 1 else 0.0)
            jacc.extend(jaccard(names[0], n) for n in names[1:])

    n = len(per_attempt)
    n_pass = sum(1 for a in per_attempt if a["parse_ok"] and a["n_err"] == 0)
    task_runs = [a for a in per_attempt if a.get("task_success") is not None]
    n_task_ok = sum(1 for a in task_runs if a["task_success"])
    return {
        "budget": label,
        "n": n,
        "pass_rate": n_pass / n if n else 0,
        "task_success_rate": (n_task_ok / len(task_runs)) if task_runs else None,
        "n_browser_runs": len(task_runs),
        "parse_rate": sum(1 for a in per_attempt if a["parse_ok"]) / n if n else 0,
        "mean_state_count": statistics.mean([a["state_count"] for a in per_attempt if a["state_count"]]) if any(a["state_count"] for a in per_attempt) else None,
        "statecount_stability": statistics.mean(sc_stable) if sc_stable else None,
        "mean_jaccard": statistics.mean(jacc) if jacc else None,
        "reasoning_tok_mean": statistics.mean(reasoning) if reasoning else None,
        "reasoning_measured": bool(reasoning),
        "output_tok_mean": statistics.mean(out_tok) if out_tok else None,
        "input_tok_mean": statistics.mean(in_tok) if in_tok else None,
        "latency_mean": statistics.mean(lat) if lat else None,
        "latency_median": statistics.median(lat) if lat else None,
        "latency_p95": sorted(lat)[int(0.95*(len(lat)-1))] if lat else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-keys", required=True)
    ap.add_argument("--prompts-file", default="hard_prompts.txt")
    ap.add_argument("--budgets", nargs="+", type=int, default=[0, 512, 2048, -1],
                    help="thinking_budget values to sweep (use -1 for dynamic)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--outroot", default="results/exp1_thinking_structural")
    ap.add_argument("--browser", action="store_true",
                    help="ALSO run each workflow in the browser and measure real task success "
                         "(needs Docker/display + a battery with success conditions)")
    ap.add_argument("--battery-json", default=None,
                    help="JSON battery [{sentence,success,...}] for --browser mode; "
                         "defaults to exp2's DEFAULT_BATTERY")
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache")
    args = ap.parse_args()

    battery = None
    if args.browser:
        if args.battery_json:
            with open(args.battery_json) as f:
                battery = json.load(f)
        else:
            from exp2_end_to_end import DEFAULT_BATTERY
            battery = DEFAULT_BATTERY
        prompts = [b["sentence"] for b in battery]
    else:
        with open(args.prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    os.makedirs(args.outroot, exist_ok=True)

    rows = []
    for b in args.budgets:
        rows.append(run_budget(b, prompts, args.repeat, args.api_keys, args.outroot, args.model,
                               browser=args.browser, battery=battery,
                               user_data_dir=args.user_data_dir))

    # comparison table
    print("\n\n" + "=" * 100)
    print("EXPERIMENT 1 RESULTS: thinking budget vs. quality / cost / latency")
    print("=" * 100)
    hdr = ["budget", "struct%", "TASK%", "parse%", "state#", "cnt-stbl", "jaccard",
           "reason-tok", "out-tok", "lat-mean"]
    print(("{:<9}{:>8}{:>7}{:>8}{:>8}{:>10}{:>9}{:>12}{:>9}{:>10}").format(*hdr))
    def f(x, nd=2):
        return "n/a" if x is None else f"{x:.{nd}f}"
    def pctf(x):
        return "n/a" if x is None else f"{x*100:.0f}"
    for r in rows:
        print(("{:<9}{:>8}{:>7}{:>8}{:>8}{:>10}{:>9}{:>12}{:>9}{:>10}").format(
            r["budget"], pctf(r["pass_rate"]), pctf(r.get("task_success_rate")),
            pctf(r["parse_rate"]), f(r["mean_state_count"],1),
            f(r["statecount_stability"],2), f(r["mean_jaccard"],2),
            f(r["reasoning_tok_mean"],0), f(r["output_tok_mean"],0), f(r["latency_mean"])))
    if any(r.get("task_success_rate") is not None for r in rows):
        print("\n  struct% = structural validity (lint-clean).  TASK% = REAL task success in browser.")
        print("  If struct% stays high but TASK% drops as thinking falls -> thinking IS buying")
        print("  real capability that structure alone hides. This is exactly why we run both.")
    else:
        print("\n  (structural-only run -- add --browser to also measure real task success)")

    # flag if the budget setting was ignored
    for r in rows:
        if r["budget"] == "0" and r["reasoning_measured"] and r["reasoning_tok_mean"] and r["reasoning_tok_mean"] > 5:
            print(f"\n  !! WARNING: thinking_budget=0 but measured reasoning tokens = "
                  f"{r['reasoning_tok_mean']:.0f}/call. The budget was IGNORED (see issue #928). "
                  f"Report the MEASURED reasoning tokens, not the requested budget.")

    import csv
    csv_path = os.path.join(args.outroot, "exp1_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[csv] {csv_path}")
    print("\nINTERPRETATION GUIDE:")
    print("  - If pass%/state#/cnt-stable hold flat as budget drops -> thinking isn't buying")
    print("    quality; disable it for a big latency/cost win. That's the headline.")
    print("  - If cnt-stable RISES with budget -> thinking stabilizes structure; there's a")
    print("    quality/cost tradeoff to report, not a free win.")
    print("  - Compare reason-tok to out-tok to quantify the reasoning overhead per setting.")


if __name__ == "__main__":
    main()
