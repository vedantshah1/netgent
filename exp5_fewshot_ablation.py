#!/usr/bin/env python3
"""
exp5_fewshot_ablation.py
========================
EXPERIMENT 5 — Does the few-shot example in the planner prompt actually matter?

MOTIVATION: generate_workflow() includes one worked example (the Wikipedia/Bezier workflow) in
the prompt. Standard practice, but is it doing any work? An ablation answers "does prompt design
matter here, or would zero-shot do just as well?" -- a cheap, defensible result, and it tells
you whether the few-shot example is worth its token cost.

WHAT THIS TESTS:
  Same battery, same everything, generated TWO ways:
    - few-shot   : the current prompt (with the worked example)
    - zero-shot  : the rules only, example removed
  Compares structural validity, JSON-fixup rate, consistency, and (optionally, --browser) real
  task success. If few-shot wins on validity/consistency, that justifies the example. If not,
  you can drop it and save tokens.

  This is a controlled A/B: ONLY the presence of the few-shot example changes.

USAGE:
    python3 exp5_fewshot_ablation.py --api-keys api_keys.json --prompts-file hard_prompts.txt --repeat 3
    python3 exp5_fewshot_ablation.py --api-keys api_keys.json --browser   # also measure task success

OUTPUT: printed side-by-side comparison + exp5_summary.csv.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netgent_planner as P
from netgent_planner import lint_workflow, _SYSTEM_RULES, _FEWSHOT, _generate_validated
from planner_telemetry import Telemetry, set_telemetry


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def generate_zeroshot(sentence, llm):
    """Same as generate_workflow but WITHOUT the few-shot example in the user message."""
    user = f"User sentence:\n\"{sentence.strip()}\"\n\nWorkflow:"
    return _generate_validated(_SYSTEM_RULES, user, llm, stage="GENERATE_ZEROSHOT", sentence=sentence)


def generate_fewshot(sentence, llm):
    user = f"{_FEWSHOT}\n\nUser sentence:\n\"{sentence.strip()}\"\n\nWorkflow:"
    return _generate_validated(_SYSTEM_RULES, user, llm, stage="GENERATE_FEWSHOT", sentence=sentence)


def run_condition(name, genfn, prompts, repeat, llm, browser, battery, uddir):
    P.DEBUG = False
    per, groups = [], defaultdict(list)
    items = battery if browser else [{"sentence": s} for s in prompts]
    print(f"\n{'='*66}\nCONDITION: {name}\n{'='*66}")
    for item in items:
        sent = item["sentence"]
        for _ in range(repeat):
            try:
                wf = genfn(sent, llm)
                issues = lint_workflow(wf, sentence=sent)
                n_err = sum(1 for i in issues if i["level"] == "ERROR")
                rec = {"parse_ok": True, "n_err": n_err, "state_count": len(wf), "task_success": None}
                if browser:
                    from exp_common import run_and_verify
                    res = run_and_verify(wf, item["success"], llm, use_human=False, user_data_dir=uddir)
                    rec["task_success"] = res["task_success"]
                    print(f"  [{name}] struct={'OK' if n_err==0 else 'ERR'} task={res['task_success']} \"{sent[:40]}\"")
                else:
                    print(f"  [{name}] {'PASS' if n_err==0 else 'FAIL'} {len(wf)}st \"{sent[:44]}\"")
                per.append(rec); groups[sent].append([s["name"] for s in wf])
            except Exception as e:
                # task_success must stay None when we are NOT measuring the browser --
                # otherwise a parse failure becomes the ONLY record in the denominator and
                # the summary reports a bogus 0.0 task-success rate (it did exactly that).
                per.append({"parse_ok": False, "n_err": 0, "state_count": None,
                            "task_success": (False if browser else None)})
                print(f"  [{name}] EXCEPTION: {str(e)[:70]}")

    n = len(per)
    sc_stable = []
    for sent, names in groups.items():
        if len(names) > 1:
            sc_stable.append(1.0 if len({len(x) for x in names}) == 1 else 0.0)
    task = [r for r in per if r.get("task_success") is not None]
    return {
        "condition": name, "n": n,
        "struct_valid_rate": sum(1 for r in per if r["parse_ok"] and r["n_err"]==0)/n if n else 0,
        "parse_rate": sum(1 for r in per if r["parse_ok"])/n if n else 0,
        "mean_states": statistics.mean([r["state_count"] for r in per if r["state_count"]]) if any(r["state_count"] for r in per) else None,
        "statecount_stability": statistics.mean(sc_stable) if sc_stable else None,
        "task_success_rate": (sum(1 for r in task if r["task_success"])/len(task)) if task else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-keys", required=True)
    ap.add_argument("--prompts-file", default="hard_prompts.txt")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--battery-json", default=None)
    ap.add_argument("--user-data-dir", default="/tmp/browser-cache")
    ap.add_argument("--outdir", default="results/exp5_ablation_structural")
    args = ap.parse_args()

    from run_planner import build_llm
    llm = build_llm(args.api_keys)
    os.makedirs(args.outdir, exist_ok=True)
    set_telemetry(Telemetry(outdir=os.path.join(args.outdir, "telemetry"),
                            meta={"experiment": "fewshot_ablation"}))

    battery = None
    if args.browser:
        if args.battery_json:
            battery = json.load(open(args.battery_json))
        else:
            from exp2_end_to_end import DEFAULT_BATTERY
            battery = DEFAULT_BATTERY
        prompts = [b["sentence"] for b in battery]
    else:
        prompts = [l.strip() for l in open(args.prompts_file) if l.strip()]

    rows = [
        run_condition("few-shot", generate_fewshot, prompts, args.repeat, llm, args.browser, battery, args.user_data_dir),
        run_condition("zero-shot", generate_zeroshot, prompts, args.repeat, llm, args.browser, battery, args.user_data_dir),
    ]

    print("\n\n" + "=" * 78)
    print("EXPERIMENT 5 RESULTS: few-shot vs zero-shot")
    print("=" * 78)
    print(f"  {'condition':<12}{'struct%':>9}{'TASK%':>8}{'parse%':>9}{'states':>8}{'cnt-stbl':>10}")
    def pf(x): return "n/a" if x is None else f"{x*100:.0f}"
    def ff(x): return "n/a" if x is None else f"{x:.1f}"
    for r in rows:
        print(f"  {r['condition']:<12}{pf(r['struct_valid_rate']):>9}{pf(r['task_success_rate']):>8}"
              f"{pf(r['parse_rate']):>9}{ff(r['mean_states']):>8}{ff(r['statecount_stability']):>10}")
    print("\n  If few-shot > zero-shot on struct%/TASK%/cnt-stbl, the example earns its tokens.")
    print("  If they're equal, you can drop it -- and that's a finding too.")

    import csv
    with open(os.path.join(args.outdir, "exp5_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n[csv] {args.outdir}/exp5_summary.csv")


if __name__ == "__main__":
    main()
