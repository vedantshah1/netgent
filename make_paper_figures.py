#!/usr/bin/env python3
"""
make_paper_figures.py
=====================
Builds every paper/poster figure from the experiment artifacts. Each figure is independent:
if its input files are missing the figure is skipped with a note, so this can be run at any
point in the experiment round and re-run as more results land.

EXPECTED LAYOUT (override with --results-dir):
    results/
      exp1_thinking/exp1_summary.csv
      exp2_easy/{records.json}          exp2_hard/{records.json}
      exp4_detect/exp4_features.csv
      exp5_ablation/exp5_summary.csv    exp5_ablation/telemetry/attempts.jsonl
      exp6_nshot/exp6_summary.csv
      exp7_baseline/exp7_summary.csv
      exp8_controller_ab/exp8_summary.csv

USAGE:
    python3 make_paper_figures.py --results-dir results -o figures
    python3 make_paper_figures.py --results-dir results -o figures --format pdf
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- house style (matches slide/poster deck) ------------------------------- #
TEAL, TEAL_LT, GREY, FAIL, ACCENT, INK = "#0E6E78", "#5FA8B0", "#B9C2C4", "#C0563B", "#9FCAD0", "#20302F"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#5b6a6a", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": "#ecf0f0", "grid.linewidth": 0.9, "axes.axisbelow": True,
})

OUT, FMT = "figures", "png"
MADE, SKIPPED = [], []


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in {FMT, "svg"}:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    MADE.append(name)
    print(f"  [ok]   {name}")


def _skip(name, why):
    SKIPPED.append((name, why))
    print(f"  [skip] {name}  ({why})")


def _bare(ax, left=True):
    for s in ("top", "right") + (("left",) if not left else ()):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


# --------------------------------------------------------------------------- #
# FIG 1 — task success: easy vs hard battery (the difficulty-scaling story)
# --------------------------------------------------------------------------- #
def fig_easy_vs_hard(rd):
    paths = [(os.path.join(rd, "exp2_easy", "records.json"), "easy (5 prompts)"),
             (os.path.join(rd, "exp2_hard", "records.json"), "hard (5 prompts)")]
    data = []
    for p, label in paths:
        if not os.path.exists(p):
            return _skip("fig1_easy_vs_hard", f"missing {p}")
        recs = json.load(open(p))
        first = sum(1 for r in recs if r["attempts"][0]["task_success"])
        final = sum(1 for r in recs if r["final_task_success"])
        data.append((label, first, final - first, len(recs) - final, len(recs)))

    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.grid(False)
    for i, (label, ft, hl, fl, n) in enumerate(data):
        y = len(data) - 1 - i
        ax.barh(y, ft, color=TEAL, edgecolor="white", height=0.55,
                label="first-try success" if i == 0 else None)
        ax.barh(y, hl, left=ft, color=TEAL_LT, edgecolor="white", height=0.55,
                label="recovered by self-heal" if i == 0 else None)
        ax.barh(y, fl, left=ft + hl, color=FAIL, edgecolor="white", height=0.55,
                label="failed" if i == 0 else None)
        for val, off in ((ft, 0), (hl, ft), (fl, ft + hl)):
            if val:
                ax.text(off + val / 2, y, str(val), ha="center", va="center",
                        color="white", fontweight="bold")
        ax.text(n + 0.12, y, f"{ft + hl}/{n}", va="center", ha="left",
                fontweight="bold", color=INK)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in reversed(data)])
    ax.set_xlim(0, max(d[4] for d in data) + 0.9)
    ax.set_xlabel("prompts")
    ax.set_title("Task success degrades with difficulty; self-healing absorbs part of it",
                 fontweight="bold", color=INK, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=9)
    _bare(ax, left=False)
    _save(fig, "fig1_easy_vs_hard")


# --------------------------------------------------------------------------- #
# FIG 2 — self-healing contribution (first-try vs final, both batteries)
# --------------------------------------------------------------------------- #
def fig_selfheal(rd):
    series = []
    for sub, label in (("exp2_easy", "easy"), ("exp2_hard", "hard")):
        p = os.path.join(rd, sub, "records.json")
        if not os.path.exists(p):
            continue
        recs = json.load(open(p))
        n = len(recs)
        series.append((label,
                       sum(1 for r in recs if r["attempts"][0]["task_success"]) / n * 100,
                       sum(1 for r in recs if r["final_task_success"]) / n * 100))
    if not series:
        return _skip("fig2_selfheal", "no exp2 records.json found")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = range(len(series))
    w = 0.36
    ax.bar([i - w / 2 for i in x], [s[1] for s in series], w,
           color=GREY, edgecolor="white", label="no self-heal (first try)")
    ax.bar([i + w / 2 for i in x], [s[2] for s in series], w,
           color=TEAL, edgecolor="white", label="with self-heal (final)")
    for i, s in enumerate(series):
        ax.text(i - w / 2, s[1] + 1.5, f"{s[1]:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(i + w / 2, s[2] + 1.5, f"{s[2]:.0f}%", ha="center", fontsize=9,
                fontweight="bold", color=TEAL)
        if s[2] > s[1]:
            ax.annotate("", xy=(i + w / 2, s[2]), xytext=(i - w / 2, s[1]),
                        arrowprops=dict(arrowstyle="->", color=FAIL, lw=1.4))
            ax.text(i, (s[1] + s[2]) / 2 + 3, f"+{s[2]-s[1]:.0f}pp",
                    ha="center", fontsize=9, color=FAIL, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([s[0] for s in series])
    ax.set_ylabel("task success (%)"); ax.set_ylim(0, 112)
    ax.set_title("Contribution of the self-healing repair loop", fontweight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _bare(ax)
    _save(fig, "fig2_selfheal_contribution")


# --------------------------------------------------------------------------- #
# FIG 3 — thinking budget: cost/consistency tradeoff
# --------------------------------------------------------------------------- #
def fig_thinking(rd):
    p = os.path.join(rd, "exp1_thinking", "exp1_summary.csv")
    if not os.path.exists(p):
        return _skip("fig3_thinking_budget", f"missing {p}")
    rows = _csv(p)
    labels = [r["budget"] for r in rows]
    lat = [_f(r.get("latency_mean"), 0) for r in rows]
    jac = [_f(r.get("mean_jaccard"), 0) for r in rows]
    out = [_f(r.get("output_tok_mean"), 0) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(labels))
    ax.bar(x, out, color=TEAL, edgecolor="white", width=0.6, label="output tokens")
    ax.set_ylabel("output tokens / workflow", color=TEAL)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_xlabel("thinking budget")
    _bare(ax)

    ax2 = ax.twinx()
    ax2.plot(list(x), jac, "o-", color=FAIL, lw=2, ms=7, label="consistency (Jaccard)")
    ax2.set_ylabel("state-name Jaccard", color=FAIL)
    ax2.set_ylim(0, 1.05); ax2.grid(False)
    for s in ("top",): ax2.spines[s].set_visible(False)
    for i, (o, j, l) in enumerate(zip(out, jac, lat)):
        ax.text(i, o + max(out) * 0.03, f"{o:.0f}\n{l:.1f}s", ha="center", fontsize=8, color="#6b7a7a")
        ax2.text(i, j + 0.04, f"{j:.2f}", ha="center", fontsize=9, color=FAIL, fontweight="bold")

    ax.set_title("More thinking costs tokens and latency while reducing consistency",
                 fontweight="bold", color=INK, pad=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
    _save(fig, "fig3_thinking_budget")


# --------------------------------------------------------------------------- #
# FIG 4 — few-shot ablation: the Terminate-action collapse
# --------------------------------------------------------------------------- #
def fig_fewshot(rd):
    p = os.path.join(rd, "exp5_ablation", "telemetry", "attempts.jsonl")
    if not os.path.exists(p):
        return _skip("fig4_fewshot_ablation", f"missing {p}")
    A = [json.loads(l) for l in open(p) if l.strip()]
    stages = {"GENERATE_FEWSHOT": "few-shot", "GENERATE_ZEROSHOT": "zero-shot"}
    bars = []
    for st, lab in stages.items():
        g = [a for a in A if a.get("stage") == st]
        if not g:
            continue
        n = len(g)
        bars.append((lab,
                     sum(1 for a in g if "NO_TERMINATE_ACTION" not in a.get("lint_codes", [])) / n * 100,
                     sum(1 for a in g if "EMPTY_ACTIONS" in a.get("lint_codes", [])) / n * 100,
                     sum(1 for a in g if a.get("parse_ok")) / n * 100, n))
    if not bars:
        return _skip("fig4_fewshot_ablation", "no GENERATE_* attempts found")

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    metrics = ["has Terminate\naction", "empty-action\nstates", "parsed\ncleanly"]
    x = range(len(metrics))
    w = 0.36
    for k, (lab, term, empty, parse, n) in enumerate(bars):
        vals = [term, empty, parse]
        off = (k - 0.5) * w
        col = TEAL if lab == "few-shot" else GREY
        ax.bar([i + off for i in x], vals, w, color=col, edgecolor="white", label=f"{lab} (n={n})")
        for i, v in enumerate(vals):
            ax.text(i + off, v + 2, f"{v:.0f}%", ha="center", fontsize=9,
                    fontweight="bold" if lab == "few-shot" else "normal", color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylabel("% of generations"); ax.set_ylim(0, 116)
    ax.set_title("The worked example is what teaches the Terminate convention",
                 fontweight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _bare(ax)
    _save(fig, "fig4_fewshot_ablation")


# --------------------------------------------------------------------------- #
# FIG 5 — n-shot scaling (exp6)
# --------------------------------------------------------------------------- #
def fig_nshot(rd):
    p = os.path.join(rd, "exp6_nshot", "exp6_summary.csv")
    if not os.path.exists(p):
        return _skip("fig5_nshot_scaling", f"missing {p}")
    rows = sorted(_csv(p), key=lambda r: int(r["n_shot"]))
    x = [int(r["n_shot"]) for r in rows]
    series = [("terminate_rate", "has Terminate action", TEAL, "o-"),
              ("struct_valid_rate", "structurally valid", TEAL_LT, "s--"),
              ("task_success_rate", "REAL task success", FAIL, "D-")]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for key, lab, col, style in series:
        ys = [_f(r.get(key)) for r in rows]
        if all(v is None for v in ys):
            continue
        xs = [xi for xi, v in zip(x, ys) if v is not None]
        vs = [v * 100 for v in ys if v is not None]
        ax.plot(xs, vs, style, color=col, lw=2, ms=7, label=lab)
        for xi, v in zip(xs, vs):
            ax.text(xi, v + 2.5, f"{v:.0f}%", ha="center", fontsize=8, color=col)
    ax.set_xticks(x); ax.set_xlabel("worked examples in prompt (n-shot)")
    ax.set_ylabel("%"); ax.set_ylim(-4, 115)
    ax.set_title("Returns to additional worked examples", fontweight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _bare(ax)
    _save(fig, "fig5_nshot_scaling")


# --------------------------------------------------------------------------- #
# FIG 6 — baseline vs planner vs planner+heal (exp7): the headline comparison
# --------------------------------------------------------------------------- #
def fig_baseline(rd):
    p = os.path.join(rd, "exp7_baseline", "exp7_summary.csv")
    if not os.path.exists(p):
        return _skip("fig6_baseline_comparison", f"missing {p}")
    rows = _csv(p)
    order = ["baseline", "planner", "planner+heal"]
    rows = [r for k in order for r in rows if r["condition"] == k]
    labels = [r["condition"].replace("baseline", "NetGent\n(hand-authored)")
                            .replace("planner+heal", "planner\n+ self-heal")
              for r in rows]
    task = [_f(r["task_success_rate"], 0) * 100 for r in rows]
    outtok = [_f(r.get("mean_output_tokens"), 0) for r in rows]
    author = [_f(r.get("total_authoring_minutes"), 0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ax = axes[0]
    cols = [GREY, TEAL_LT, TEAL]
    ax.bar(range(len(rows)), task, color=cols[:len(rows)], edgecolor="white", width=0.6)
    for i, v in enumerate(task):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", color=INK)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("task success (%)"); ax.set_ylim(0, 115)
    ax.set_title("Accuracy", fontweight="bold", color=INK)
    _bare(ax)

    ax = axes[1]
    ax.bar(range(len(rows)), outtok, color=cols[:len(rows)], edgecolor="white", width=0.6)
    for i, (v, a) in enumerate(zip(outtok, author)):
        note = f"{v:.0f} tok" + (f"\n{a:.0f} human-min" if a else "\n0 human-min")
        ax.text(i, v + max(outtok or [1]) * 0.03, note, ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("output tokens / task")
    ax.set_title("Authoring cost", fontweight="bold", color=INK)
    _bare(ax)

    fig.suptitle("What the planner adds to NetGent: accuracy traded for authoring cost",
                 fontweight="bold", color=INK, y=1.04)
    _save(fig, "fig6_baseline_comparison")


# --------------------------------------------------------------------------- #
# FIG 7 — stock vs human controller (exp8)
# --------------------------------------------------------------------------- #
def fig_controller(rd):
    p = os.path.join(rd, "exp8_controller_ab", "exp8_summary.csv")
    if not os.path.exists(p):
        return _skip("fig7_controller_ab", f"missing {p}")
    rows = _csv(p)
    labels = [r["condition"] for r in rows]
    task = [_f(r["task_success_rate"], 0) * 100 for r in rows]
    lat = [_f(r["mean_run_latency_s"], 0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    cols = [GREY if l == "stock" else TEAL for l in labels]
    for ax, vals, ylab, title, fmt in (
            (axes[0], task, "task success (%)", "Accuracy is unaffected", "{:.0f}%"),
            (axes[1], lat, "mean run time (s)", "Cost of the imitation layer", "{:.1f}s")):
        ax.bar(range(len(labels)), vals, color=cols, edgecolor="white", width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals or [1]) * 0.03, fmt.format(v), ha="center",
                    fontweight="bold", color=INK)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold", color=INK)
        ax.set_ylim(0, max(vals or [1]) * 1.25)
        _bare(ax)
    fig.suptitle("Human-imitation controller: what it costs and what it changes",
                 fontweight="bold", color=INK, y=1.04)
    _save(fig, "fig7_controller_ab")


# --------------------------------------------------------------------------- #
# FIG 8 — stopgap verifier reliability (motivates the real verifier)
# --------------------------------------------------------------------------- #
def fig_verifier(rd):
    tp = fp = tn = fn = 0
    found = False
    for sub in ("exp2_easy", "exp2_hard"):
        p = os.path.join(rd, sub, "records.json")
        if not os.path.exists(p):
            continue
        found = True
        for r in json.load(open(p)):
            for a in r["attempts"]:
                s, f = a["task_success"], a["stopgap_failed"]
                if not s and f: tp += 1
                elif s and f: fp += 1
                elif s and not f: tn += 1
                else: fn += 1
    if not found:
        return _skip("fig8_stopgap_verifier", "no exp2 records.json found")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.grid(False)
    succ, fail = tn + fp, tp + fn
    ax.barh(1, tn, color=TEAL, edgecolor="white", height=0.5, label="verifier correct")
    ax.barh(1, fp, left=tn, color=FAIL, edgecolor="white", height=0.5, label="verifier wrong")
    ax.barh(0, tp, color=TEAL, edgecolor="white", height=0.5)
    ax.barh(0, fn, left=tp, color=FAIL, edgecolor="white", height=0.5)
    for y, (ok, bad, tot) in ((1, (tn, fp, succ)), (0, (tp, fn, fail))):
        if ok: ax.text(ok / 2, y, str(ok), ha="center", va="center", color="white", fontweight="bold")
        if bad: ax.text(ok + bad / 2, y, str(bad), ha="center", va="center", color="white", fontweight="bold")
        ax.text(tot + 0.15, y, f"n={tot}", va="center", fontsize=9, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["runs that FAILED", "runs that SUCCEEDED"])
    ax.set_xlabel("browser runs")
    ax.set_xlim(0, max(succ, fail) + 1.4)
    fprate = fp / succ * 100 if succ else 0
    ax.set_title(f"Stopgap verifier mislabels {fprate:.0f}% of successful runs as failures",
                 fontweight="bold", color=INK, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=9)
    _bare(ax, left=False)
    _save(fig, "fig8_stopgap_verifier")


# --------------------------------------------------------------------------- #
# FIG 9 — motor-model features, stock vs human (exp4)
# --------------------------------------------------------------------------- #
def fig_motor(rd):
    p = os.path.join(rd, "exp4_detect", "exp4_features.csv")
    if not os.path.exists(p):
        return _skip("fig9_motor_features", f"missing {p}")
    rows = _csv(p)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    feats = [("straightness", "straightness", 1.0),
             ("dev_ratio", "path deviation", None),
             ("peak_mean_v_ratio", "peak/mean velocity", 1.875)]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, (key, lab, human_ref) in zip(axes, feats):
        names, vals, errs, cols = [], [], [], []
        for g in ("stock", "human"):
            if g not in groups:
                continue
            v = [_f(r[key], 0) for r in groups[g]]
            names.append("stock" if g == "stock" else "HumanCtrl")
            vals.append(sum(v) / len(v))
            errs.append((max(v) - min(v)) / 2)
            cols.append(GREY if g == "stock" else TEAL)
        ax.bar(range(len(vals)), vals, yerr=errs, capsize=4, color=cols,
               edgecolor="white", width=0.55)
        if human_ref is not None:
            ax.axhline(human_ref, color=FAIL, ls="--", lw=1.5)
            ax.text(len(vals) - 0.5, human_ref, f" human ref {human_ref}", color=FAIL,
                    fontsize=8, va="bottom", ha="right")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
        ax.set_title(lab, fontweight="bold", color=INK, fontsize=11)
        _bare(ax)
    fig.suptitle("Motor-model features: the imitation removes the trivial tells "
                 "but under-shoots the human velocity profile",
                 fontweight="bold", color=INK, y=1.06)
    _save(fig, "fig9_motor_features")


# --------------------------------------------------------------------------- #
def main():
    global OUT, FMT
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("-o", "--outdir", default="figures")
    ap.add_argument("--format", default="png", choices=["png", "pdf"])
    args = ap.parse_args()
    OUT, FMT = args.outdir, args.format

    print(f"reading artifacts from: {args.results_dir}\nwriting figures to:    {OUT}\n")
    for fn in (fig_easy_vs_hard, fig_selfheal, fig_thinking, fig_fewshot,
               fig_nshot, fig_baseline, fig_controller, fig_verifier, fig_motor):
        try:
            fn(args.results_dir)
        except Exception as e:
            _skip(fn.__name__, f"error: {type(e).__name__}: {str(e)[:90]}")

    print(f"\n{len(MADE)} figure(s) written, {len(SKIPPED)} skipped.")
    if SKIPPED:
        print("\nskipped:")
        for n, why in SKIPPED:
            print(f"  - {n}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
