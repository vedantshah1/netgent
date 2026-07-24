#!/usr/bin/env python3
"""
analyze_planner_results.py
--------------------------
Turn the JSONL telemetry written by planner_telemetry.py into (a) a metrics report you
can hand your mentor and (b) figures/tables for the paper.

Usage:
    python3 analyze_planner_results.py /out/telemetry
    python3 analyze_planner_results.py /out/telemetry --csv /out/results.csv --figs /out/figs

Reports:
  RELIABILITY   pass rate, JSON-parse-failure rate, fixup rate, lint error taxonomy
  LATENCY       mean/median/p95/min/max per stage
  TOKENS        input/output/total per call and per attempt, plus a cost estimate
  WORKFLOW      state-count distribution, triggers/actions per state
  CONSISTENCY   for repeated identical prompts: does the planner produce the same thing?

Everything degrades gracefully: no matplotlib -> skips figures; no token data from the
provider -> reports "not reported" instead of inventing numbers.
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict

# gemini-2.5-flash public pricing at time of writing, USD per 1M tokens.
# VERIFY THIS before quoting any cost figure in the paper -- provider pricing changes and
# this is the one number here that isn't measured from your own runs.
DEFAULT_PRICE_IN = 0.30
DEFAULT_PRICE_OUT = 2.50


def load_jsonl(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt(x, nd=2, suffix=""):
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def report(calls, attempts, price_in, price_out):
    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 78)
    out("NETGENT PLANNER -- RESULTS REPORT")
    out("=" * 78)
    out(f"LLM calls recorded : {len(calls)}")
    out(f"Attempts recorded  : {len(attempts)}")
    if not attempts and not calls:
        out("\nNo telemetry found. Did you set PLANNER_TELEMETRY_DIR before the run?")
        return lines

    # ---------------- RELIABILITY ----------------
    out("\n" + "-" * 78)
    out("RELIABILITY")
    out("-" * 78)
    if attempts:
        n = len(attempts)
        n_pass = sum(1 for a in attempts if a.get("verdict") == "PASS")
        n_parse_ok = sum(1 for a in attempts if a.get("parse_ok"))
        n_fixup = sum(1 for a in attempts if a.get("needed_json_fixup"))
        out(f"  Attempts                : {n}")
        out(f"  PASS (parsed, 0 lint errors): {n_pass}/{n}  ({100.0*n_pass/n:.1f}%)")
        out(f"  Parsed successfully     : {n_parse_ok}/{n}  ({100.0*n_parse_ok/n:.1f}%)")
        out(f"  Needed JSON fixup retry : {n_fixup}/{n}  ({100.0*n_fixup/n:.1f}%)"
            f"   <- how often the model emits bad JSON first try")
        errs = [a for a in attempts if not a.get("parse_ok")]
        if errs:
            out(f"  Hard parse failures     : {len(errs)}")
            for e in errs[:5]:
                out(f"      - {str(e.get('parse_error'))[:100]}")

        # lint taxonomy
        codes = Counter()
        for a in attempts:
            for c in a.get("lint_codes") or []:
                codes[c] += 1
        if codes:
            out("\n  Lint issue frequency (across all attempts):")
            for code, cnt in codes.most_common():
                out(f"      {code:<22} {cnt:>4}")

        by_stage = Counter(a.get("stage") for a in attempts)
        out("\n  Attempts by stage: " + ", ".join(f"{k}={v}" for k, v in by_stage.items()))

    # ---------------- REPAIR / FAILURE CLASSES ----------------
    repairs = [a for a in attempts if a.get("stage") == "REPAIR"]
    if repairs:
        out("\n" + "-" * 78)
        out("REPAIR BY FAILURE CLASS  (planner half of self-healing)")
        out("-" * 78)
        by_class = defaultdict(list)
        for r in repairs:
            by_class[r.get("failure_class", "UNKNOWN")].append(r)
        out(f"  {'failure_class':<24}{'n':>4}{'parsed':>8}{'lint-clean':>12}{'mean lat':>10}")
        for fc, items in sorted(by_class.items()):
            n = len(items)
            ok = sum(1 for i in items if i.get("parse_ok"))
            clean = sum(1 for i in items if i.get("lint_errors") == 0 and i.get("parse_ok"))
            lat = [i["latency_s"] for i in items if i.get("latency_s") is not None]
            out(f"  {fc:<24}{n:>4}{ok:>8}{clean:>12}"
                f"{fmt(statistics.mean(lat)) if lat else 'n/a':>10}")
        srcs = Counter(r.get("verifier_source") for r in repairs if r.get("verifier_source"))
        if srcs:
            out("\n  Verifier source: " + ", ".join(f"{k}={v}" for k, v in srcs.items()))
            if "stopgap_heuristic" in srcs:
                out("  NOTE: some/all repairs were driven by the STOPGAP heuristic, not the real")
                out("        verifier. Don't report these as verifier-driven results.")
        grew = [r for r in repairs if r.get("states_before_repair") is not None
                and r.get("state_count") is not None]
        if grew:
            deltas = [r["state_count"] - r["states_before_repair"] for r in grew]
            out(f"\n  State-count change after repair: mean={fmt(statistics.mean(deltas),1)}  "
                f"(repairs that ADDED states: {sum(1 for d in deltas if d > 0)}/{len(deltas)})")

    # ---------------- LATENCY ----------------
    out("\n" + "-" * 78)
    out("LATENCY")
    out("-" * 78)
    if calls:
        by_stage = defaultdict(list)
        for c in calls:
            if c.get("latency_s") is not None:
                by_stage[c.get("stage", "?")].append(c["latency_s"])
        out(f"  {'stage':<22}{'n':>4}{'mean':>9}{'median':>9}{'p95':>9}{'min':>9}{'max':>9}")
        for stage, vs in sorted(by_stage.items()):
            out(f"  {stage:<22}{len(vs):>4}{fmt(statistics.mean(vs)):>9}"
                f"{fmt(statistics.median(vs)):>9}{fmt(pct(vs,95)):>9}"
                f"{fmt(min(vs)):>9}{fmt(max(vs)):>9}")
        allv = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
        if allv:
            out(f"  {'ALL CALLS':<22}{len(allv):>4}{fmt(statistics.mean(allv)):>9}"
                f"{fmt(statistics.median(allv)):>9}{fmt(pct(allv,95)):>9}"
                f"{fmt(min(allv)):>9}{fmt(max(allv)):>9}")
    if attempts:
        av = [a["latency_s"] for a in attempts if a.get("latency_s") is not None]
        if av:
            out(f"\n  End-to-end per attempt: mean={fmt(statistics.mean(av))}s  "
                f"median={fmt(statistics.median(av))}s  p95={fmt(pct(av,95))}s")

    # ---------------- TOKENS ----------------
    out("\n" + "-" * 78)
    out("TOKENS")
    out("-" * 78)
    ins = [c["input_tokens"] for c in calls if c.get("input_tokens") is not None]
    outs = [c["output_tokens"] for c in calls if c.get("output_tokens") is not None]
    if not ins and not outs:
        out("  Token usage not reported by the provider for these calls.")
        out("  (checked .usage_metadata and .response_metadata; if this persists, the")
        out("   langchain-google-genai version may not surface counts -- verify on the VM.)")
    else:
        out(f"  Input tokens  : total={sum(ins):,}  mean/call={fmt(statistics.mean(ins),1)}"
            f"  max={max(ins)}" if ins else "  Input tokens: not reported")
        out(f"  Output tokens : total={sum(outs):,}  mean/call={fmt(statistics.mean(outs),1)}"
            f"  max={max(outs)}" if outs else "  Output tokens: not reported")
        if ins and outs:
            cost = sum(ins) / 1e6 * price_in + sum(outs) / 1e6 * price_out
            out(f"  Total tokens  : {sum(ins)+sum(outs):,}")
            out(f"  Est. cost     : ${cost:.4f}  (at ${price_in}/1M in, ${price_out}/1M out"
                f" -- VERIFY current pricing before citing)")
            if attempts:
                out(f"  Est. cost per attempt: ${cost/len(attempts):.5f}")

    fr = Counter(c.get("finish_reason") for c in calls if c.get("finish_reason"))
    if fr:
        out("  Finish reasons: " + ", ".join(f"{k}={v}" for k, v in fr.items()))
        if any(k for k in fr if "MAX_TOKENS" in str(k).upper()):
            out("  WARNING: MAX_TOKENS seen -- output was TRUNCATED, which will look like a")
            out("           JSON parse error but needs a bigger max_output_tokens, not a prompt fix.")
    if any(c.get("safety_ratings") for c in calls):
        out("  NOTE: safety_ratings present on some calls -- check for SAFETY blocks.")

    # ---------------- WORKFLOW SHAPE ----------------
    out("\n" + "-" * 78)
    out("WORKFLOW SHAPE")
    out("-" * 78)
    ok = [a for a in attempts if a.get("parse_ok")]
    if ok:
        sc = [a["state_count"] for a in ok]
        out(f"  State count: mean={fmt(statistics.mean(sc),1)}  median={statistics.median(sc)}  "
            f"min={min(sc)}  max={max(sc)}")
        out(f"  Distribution: {dict(sorted(Counter(sc).items()))}")
        tr = [a["total_triggers"] for a in ok]
        ac = [a["total_actions"] for a in ok]
        if tr:
            out(f"  Triggers/workflow: mean={fmt(statistics.mean(tr),1)}   "
                f"Actions/workflow: mean={fmt(statistics.mean(ac),1)}")
        no_trig = sum(1 for a in ok if a.get("states_with_no_triggers"))
        if no_trig:
            out(f"  !! {no_trig} workflow(s) contained a state with NO triggers "
                f"(the always-match/infinite-loop trap)")

    # ---------------- CONSISTENCY ----------------
    out("\n" + "-" * 78)
    out("CONSISTENCY (identical prompt -> identical workflow?)")
    out("-" * 78)
    groups = defaultdict(list)
    for a in ok:
        if a.get("sentence") and a.get("stage") == "GENERATE":
            groups[a["sentence"]].append(a)
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    if not repeated:
        out("  No prompt was generated more than once -- run the test harness with")
        out("  --repeat N to measure this.")
    else:
        for sent, items in repeated.items():
            counts = [i["state_count"] for i in items]
            names = [i["state_names"] for i in items]
            sims = [jaccard(names[0], n) for n in names[1:]]
            identical = sum(1 for n in names[1:] if n == names[0]) + 1
            out(f"\n  \"{sent[:64]}{'...' if len(sent) > 64 else ''}\"  (n={len(items)})")
            out(f"    state counts     : {counts}")
            out(f"    identical naming : {identical}/{len(items)}")
            out(f"    mean Jaccard sim : {fmt(statistics.mean(sims), 3) if sims else 'n/a'}"
                f"   (1.0 = same state names every time)")
    out("\n" + "=" * 78)
    return lines


def write_csv(attempts, path):
    """Flat per-attempt CSV -- easiest thing to paste into a spreadsheet or pandas."""
    import csv
    cols = ["run_id", "attempt_id", "iso", "stage", "sentence", "verdict", "parse_ok",
            "needed_json_fixup", "parse_error", "latency_s", "n_llm_calls",
            "input_tokens", "output_tokens", "total_tokens", "state_count",
            "total_triggers", "total_actions", "lint_errors", "lint_warnings",
            "lint_infos", "state_names", "lint_codes"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for a in attempts:
            row = dict(a)
            row["state_names"] = "|".join(str(x) for x in (a.get("state_names") or []))
            row["lint_codes"] = "|".join(a.get("lint_codes") or [])
            w.writerow(row)
    print(f"[csv] wrote {path}  ({len(attempts)} rows)")


def make_figs(calls, attempts, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figs] matplotlib not installed; skipping. "
              "pip install matplotlib --break-system-packages")
        return
    os.makedirs(outdir, exist_ok=True)

    lat = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    if lat:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(lat, bins=20)
        ax.set_xlabel("LLM call latency (s)"); ax.set_ylabel("count")
        ax.set_title("Planner LLM call latency")
        fig.tight_layout(); fig.savefig(f"{outdir}/latency_hist.png", dpi=150)
        print(f"[figs] wrote {outdir}/latency_hist.png")

    ok = [a for a in attempts if a.get("parse_ok")]
    if ok:
        sc = Counter(a["state_count"] for a in ok)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(sc.keys()), list(sc.values()))
        ax.set_xlabel("states per generated workflow"); ax.set_ylabel("count")
        ax.set_title("Workflow size distribution")
        fig.tight_layout(); fig.savefig(f"{outdir}/state_count.png", dpi=150)
        print(f"[figs] wrote {outdir}/state_count.png")

    codes = Counter()
    for a in attempts:
        for c in a.get("lint_codes") or []:
            codes[c] += 1
    if codes:
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = [k for k, _ in codes.most_common()]
        vs = [v for _, v in codes.most_common()]
        ax.barh(ks[::-1], vs[::-1])
        ax.set_xlabel("occurrences"); ax.set_title("Lint issue frequency")
        fig.tight_layout(); fig.savefig(f"{outdir}/lint_taxonomy.png", dpi=150)
        print(f"[figs] wrote {outdir}/lint_taxonomy.png")

    ins = [c.get("input_tokens") for c in calls if c.get("input_tokens") is not None]
    outs = [c.get("output_tokens") for c in calls if c.get("output_tokens") is not None]
    if ins and outs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ins, outs, s=18, alpha=0.7)
        ax.set_xlabel("input tokens"); ax.set_ylabel("output tokens")
        ax.set_title("Token usage per LLM call")
        fig.tight_layout(); fig.savefig(f"{outdir}/tokens.png", dpi=150)
        print(f"[figs] wrote {outdir}/tokens.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry_dir")
    ap.add_argument("--csv", default=None, help="Write a flat per-attempt CSV here")
    ap.add_argument("--figs", default=None, help="Write figures to this directory")
    ap.add_argument("--report", default=None, help="Save the text report here")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN)
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT)
    a = ap.parse_args()

    calls = load_jsonl(os.path.join(a.telemetry_dir, "llm_calls.jsonl"))
    attempts = load_jsonl(os.path.join(a.telemetry_dir, "attempts.jsonl"))
    lines = report(calls, attempts, a.price_in, a.price_out)

    if a.report:
        with open(a.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[report] wrote {a.report}")
    if a.csv:
        write_csv(attempts, a.csv)
    if a.figs:
        make_figs(calls, attempts, a.figs)
