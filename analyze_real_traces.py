#!/usr/bin/env python3
"""
analyze_real_traces.py
======================
Turns REAL captured browser traces into the four-panel human-imitation figure and a diagnostic
table. Nothing here is synthetic: every number and every line comes from HUMAN_TRACE output
recorded while NetGent was actually driving a browser.

WHY THIS EXISTS: Exp 3 produced `mouse_paths.png` and `fitts.png` from a synthetic grid sweep --
no browser, no NetGent, and every move horizontal (y fixed at 540). Those figures characterize a
calibration harness, not the system. This script replaces them with the real thing.

HOW TO GET TRACES: set HUMAN_TRACE to a file path before any browser run --
    HUMAN_TRACE=/out/traces/human_run1.jsonl python3 exp2_end_to_end.py ...
Exp 8 (exp8_controller_ab.py) already does this per condition and writes to <outdir>/traces/.

THE FOUR PANELS (one per imitation technique):
    A. Bezier path geometry   -- real cursor paths vs the straight line an unmodified bot draws
    B. Fitts's Law timing     -- movement time vs index of difficulty, with the fitted regression
    C. Keystroke dynamics     -- inter-key interval distribution and its coefficient of variation
    D. Scroll behaviour       -- tick quantization, deceleration, and overshoot corrections

USAGE:
    # single condition
    python3 analyze_real_traces.py traces/human_run1.jsonl -o figures

    # stock vs human overlay (panels show both where meaningful)
    python3 analyze_real_traces.py --human traces/human_*.jsonl --stock traces/stock_*.jsonl -o figures

OUTPUT: fig_imitation_4panel.{png,svg}, plus fig_idle_tremor.{png,svg} when idle data exists,
        and real_trace_stats.csv
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, TEAL_LT, GREY, FAIL, INK = "#0E6E78", "#5FA8B0", "#B9C2C4", "#C0563B", "#20302F"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#5b6a6a", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": "#eef2f2", "grid.linewidth": 0.8, "axes.axisbelow": True,
})


# --------------------------------------------------------------------------- #
def load(paths):
    evs = []
    for pat in paths:
        for p in sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []):
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return evs


def moves_from(evs):
    """Group move_start + following move_samples into complete movements."""
    out, cur = [], None
    for e in evs:
        ev = e.get("ev")
        if ev == "move_start":
            if cur and cur["samples"]:
                out.append(cur)
            cur = {"start": e, "samples": []}
        elif ev == "move_sample" and cur is not None:
            cur["samples"].append(e)
        elif ev in ("click", "key", "scroll_start", "idle_start") and cur is not None:
            if cur["samples"]:
                out.append(cur)
            cur = None
    if cur and cur["samples"]:
        out.append(cur)
    return out


def move_metrics(m):
    s, sm = m["start"], m["samples"]
    x0, y0 = s["from"]
    x1, y1 = s["to"]
    D = s.get("dist") or math.hypot(x1 - x0, y1 - y0)
    W = s.get("target_w") or 40.0
    if D <= 0:
        return None
    ID = math.log2(D / W + 1.0)

    ts = [e.get("t") for e in sm if e.get("t") is not None]
    MT = (ts[-1] - s["t"]) if (ts and s.get("t") is not None) else None

    # max perpendicular deviation (sagitta)
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    dx, dy = x1 - x0, y1 - y0
    sag = max((abs(dy * (e["x"] - x0) - dx * (e["y"] - y0)) / L) for e in sm) if sm else 0.0

    # path length vs straight-line -> straightness
    plen, px, py = 0.0, x0, y0
    for e in sm:
        plen += math.hypot(e["x"] - px, e["y"] - py)
        px, py = e["x"], e["y"]
    straightness = (L / plen) if plen else 1.0

    # velocity profile from consecutive samples
    peak_mean = None
    t_peak_frac = None
    if len(sm) >= 3 and all(e.get("t") is not None for e in sm):
        vs, tt = [], []
        prev = (x0, y0, s["t"])
        for e in sm:
            dt = e["t"] - prev[2]
            if dt > 0:
                vs.append(math.hypot(e["x"] - prev[0], e["y"] - prev[1]) / dt)
                tt.append(e["t"] - s["t"])
            prev = (e["x"], e["y"], e["t"])
        if vs and st.mean(vs) > 0:
            peak_mean = max(vs) / st.mean(vs)
            if MT:
                t_peak_frac = tt[vs.index(max(vs))] / MT

    return {"D": D, "W": W, "ID": ID, "MT": MT, "sagitta": sag, "sag_ratio": sag / D,
            "straightness": straightness, "peak_mean_v": peak_mean,
            "t_peak_frac": t_peak_frac, "intended": s.get("fitts_duration"),
            "steps": s.get("steps"), "path": [(x0, y0)] + [(e["x"], e["y"]) for e in sm],
            "endpoints": ((x0, y0), (x1, y1))}


def linreg(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else float("nan"))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*", help="trace files (treated as the human condition)")
    ap.add_argument("--human", nargs="*", default=[])
    ap.add_argument("--stock", nargs="*", default=[])
    ap.add_argument("-o", "--outdir", default="figures")
    ap.add_argument("--max-paths", type=int, default=40, help="paths drawn in panel A")
    args = ap.parse_args()

    human_paths = list(args.human) + list(args.traces)
    groups = {}
    if human_paths:
        groups["human"] = load(human_paths)
    if args.stock:
        groups["stock"] = load(args.stock)
    if not groups:
        print("no trace files given"); return 1

    os.makedirs(args.outdir, exist_ok=True)

    stats_rows = []
    parsed = {}
    for g, evs in groups.items():
        mv = [m for m in (move_metrics(x) for x in moves_from(evs)) if m]
        keys = [e for e in evs if e.get("ev") == "key" and e.get("delay_ms") is not None]
        typos = [e for e in evs if e.get("ev") == "typo"]
        sticks = [e for e in evs if e.get("ev") == "scroll_tick"]
        sstart = [e for e in evs if e.get("ev") == "scroll_start"]
        sover = [e for e in evs if e.get("ev") == "scroll_overshoot_correct"]
        send = [e for e in evs if e.get("ev") == "scroll_end"]
        tremor = [e for e in evs if e.get("ev") == "tremor"]
        idles = [e for e in evs if e.get("ev") == "idle_start"]
        parsed[g] = dict(moves=mv, keys=keys, typos=typos, sticks=sticks, sstart=sstart,
                         sover=sover, send=send, tremor=tremor, idles=idles, evs=evs)

        def cv(v):
            return (st.pstdev(v) / st.mean(v) * 100) if v and st.mean(v) else float("nan")

        sr = [m["sag_ratio"] for m in mv]
        pm = [m["peak_mean_v"] for m in mv if m["peak_mean_v"]]
        stt = [m["straightness"] for m in mv]
        ik = [e["delay_ms"] for e in keys]
        row = {
            "group": g, "n_moves": len(mv), "n_keys": len(keys), "n_typos": len(typos),
            "n_scroll_ticks": len(sticks), "n_scrolls": len(sstart),
            "n_overshoot_corrections": len(sover), "n_tremors": len(tremor),
            "sag_ratio_mean": round(st.mean(sr), 4) if sr else None,
            "sag_ratio_cv_pct": round(cv(sr), 1) if sr else None,
            "straightness_mean": round(st.mean(stt), 4) if stt else None,
            "straightness_cv_pct": round(cv(stt), 2) if stt else None,
            "peak_mean_v": round(st.mean(pm), 3) if pm else None,
            "interkey_ms_mean": round(st.mean(ik), 1) if ik else None,
            "interkey_cv_pct": round(cv(ik), 1) if ik else None,
            "typo_rate_pct": round(len(typos) / len(keys) * 100, 2) if keys else None,
        }
        mt = [(m["ID"], m["MT"]) for m in mv if m["MT"]]
        if len(mt) >= 3:
            fit = linreg([a for a, _ in mt], [b for _, b in mt])
            if fit:
                row["fitts_a"], row["fitts_b"], row["fitts_r2"] = (round(fit[0], 4),
                                                                    round(fit[1], 4),
                                                                    round(fit[2], 4))
        stats_rows.append(row)

    # ---------------- printed diagnostics ---------------- #
    print("\n" + "=" * 78)
    print("  REAL TRACE DIAGNOSTICS  (all values from live browser runs)")
    print("=" * 78)
    for r in stats_rows:
        print(f"\n  [{r['group']}]  moves={r['n_moves']}  keys={r['n_keys']}  "
              f"scroll_ticks={r['n_scroll_ticks']}  tremors={r['n_tremors']}")
        print(f"    Fitts       : a={r.get('fitts_a')}  b={r.get('fitts_b')}  "
              f"R2={r.get('fitts_r2')}          [target R2 > 0.8]")
        print(f"    curvature   : sagitta/D={r['sag_ratio_mean']}  CV={r['sag_ratio_cv_pct']}%"
              f"          [target CV > 25%]")
        print(f"    straightness: {r['straightness_mean']}  CV={r['straightness_cv_pct']}%"
              f"        [target CV > 5%]")
        print(f"    velocity    : peak/mean={r['peak_mean_v']}"
              f"                     [target ~1.875 = min-jerk]")
        print(f"    keystrokes  : {r['interkey_ms_mean']}ms  CV={r['interkey_cv_pct']}%  "
              f"typos={r['typo_rate_pct']}%   [human CV ~30-50%]")

    with open(os.path.join(args.outdir, "real_trace_stats.csv"), "w", newline="") as f:
        cols = sorted({k for r in stats_rows for k in r})
        w = csv.DictWriter(f, fieldnames=["group"] + [c for c in cols if c != "group"])
        w.writeheader(); w.writerows(stats_rows)

    # ---------------- the four-panel figure ---------------- #
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    prim = "human" if "human" in parsed else list(parsed)[0]
    P = parsed[prim]

    # --- A: Bezier path geometry --- #
    ax = axes[0][0]
    mv = P["moves"]
    if mv:
        shown = mv[:args.max_paths]
        for m in shown:
            (x0, y0), (x1, y1) = m["endpoints"]
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            # rotate each path onto a common horizontal axis so real (varied-direction)
            # movements can be compared on one set of axes
            xs, ys = [], []
            for (px, py) in m["path"]:
                rx, ry = px - x0, py - y0
                xs.append((rx * ux + ry * uy) / L)          # 0..1 along travel
                ys.append((-rx * uy + ry * ux))             # perpendicular deviation, px
            ax.plot(xs, ys, color=TEAL, alpha=0.35, lw=1.0)
        ax.axhline(0, color=GREY, ls=":", lw=1.5)
        ax.text(0.02, 0.96, f"n={len(shown)} real movements", transform=ax.transAxes,
                fontsize=8, va="top", color="#6b7a7a")
        ax.text(0.98, 0.04, "dotted = straight line (stock controller)",
                transform=ax.transAxes, fontsize=8, ha="right", color="#6b7a7a")
    else:
        ax.text(0.5, 0.5, "no movement data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("fraction of travel"); ax.set_ylabel("perpendicular deviation (px)")
    ax.set_title("A · Bézier path geometry", fontweight="bold", color=INK, loc="left")

    # --- B: Fitts timing --- #
    ax = axes[0][1]
    pts = [(m["ID"], m["MT"]) for m in mv if m["MT"]]
    if len(pts) >= 3:
        xs = [a for a, _ in pts]; ys = [b for _, b in pts]
        ax.scatter(xs, ys, s=18, color=TEAL, alpha=0.55, edgecolors="none")
        fit = linreg(xs, ys)
        if fit:
            a, b, r2 = fit
            lo, hi = min(xs), max(xs)
            ax.plot([lo, hi], [a + b * lo, a + b * hi], color=FAIL, lw=2)
            ax.text(0.03, 0.95, f"MT = {a:.3f} + {b:.3f}·ID\nR² = {r2:.3f}  (n={len(xs)})",
                    transform=ax.transAxes, va="top", fontsize=9, color=INK,
                    bbox=dict(fc="white", ec="#dde5e5", boxstyle="round,pad=0.35"))
    else:
        ax.text(0.5, 0.5, "no timed movements", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("index of difficulty  log₂(D/W + 1)  [bits]")
    ax.set_ylabel("movement time (s)")
    ax.set_title("B · Fitts's Law timing", fontweight="bold", color=INK, loc="left")

    # --- C: Keystroke dynamics --- #
    ax = axes[1][0]
    drew = False
    for g, col, lab in (("stock", GREY, "stock"), ("human", TEAL, "HumanController")):
        if g not in parsed:
            continue
        ik = [e["delay_ms"] for e in parsed[g]["keys"]]
        if len(ik) < 5:
            continue
        ax.hist(ik, bins=min(30, max(8, len(ik) // 4)), color=col, alpha=0.75,
                edgecolor="white", label=f"{lab} (n={len(ik)})")
        drew = True
    if drew:
        ik = [e["delay_ms"] for e in P["keys"]]
        if ik and st.mean(ik):
            cvv = st.pstdev(ik) / st.mean(ik) * 100
            ax.axvline(st.mean(ik), color=FAIL, ls="--", lw=1.5)
            ax.text(0.97, 0.95, f"mean {st.mean(ik):.0f} ms\nCV {cvv:.0f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
                    bbox=dict(fc="white", ec="#dde5e5", boxstyle="round,pad=0.35"))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no keystroke data\n(run a task that types)",
                ha="center", va="center", transform=ax.transAxes, color="#6b7a7a")
    ax.set_xlabel("inter-key interval (ms)"); ax.set_ylabel("count")
    ax.set_title("C · Keystroke dynamics", fontweight="bold", color=INK, loc="left")

    # --- D: Scroll behaviour --- #
    ax = axes[1][1]
    ticks = P["sticks"]
    if ticks:
        # reconstruct cumulative scroll position per burst using 'remaining'
        series, cur = [], []
        for e in P["evs"]:
            ev = e.get("ev")
            if ev == "scroll_start":
                if cur:
                    series.append(cur)
                cur = []
            elif ev == "scroll_tick" and cur is not None:
                cur.append(e)
            elif ev == "scroll_end":
                if cur:
                    series.append(cur); cur = []
        if cur:
            series.append(cur)
        for burst in series[:12]:
            pos, xs, ys = 0.0, [], []
            for i, e in enumerate(burst):
                pos += abs(e.get("amount", 0))
                xs.append(i); ys.append(pos)
            ax.plot(xs, ys, "-o", ms=2.5, lw=1.1, color=TEAL, alpha=0.55)
        amts = [abs(e.get("amount", 0)) for e in ticks]
        ax.text(0.03, 0.95,
                f"{len(series)} scroll bursts · {len(ticks)} ticks\n"
                f"tick size {min(amts):.0f}–{max(amts):.0f}px\n"
                f"{len(P['sover'])} overshoot corrections",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK,
                bbox=dict(fc="white", ec="#dde5e5", boxstyle="round,pad=0.35"))
    else:
        ax.text(0.5, 0.5, "no scroll data\n(run a task that scrolls)",
                ha="center", va="center", transform=ax.transAxes, color="#6b7a7a")
    ax.set_xlabel("tick index within a scroll burst"); ax.set_ylabel("cumulative pixels scrolled")
    ax.set_title("D · Scroll discretization & overshoot", fontweight="bold", color=INK, loc="left")

    for row in axes:
        for a in row:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(length=0)

    fig.suptitle("Human-imitation layer, measured on real NetGent browser runs",
                 fontweight="bold", color=INK, fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(args.outdir, f"fig_imitation_4panel.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {args.outdir}/fig_imitation_4panel.png")

    # ---------------- optional 5th: idle tremor ---------------- #
    if P["tremor"]:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        xs = [e["x"] for e in P["tremor"]]
        ys = [e["y"] for e in P["tremor"]]
        cx, cy = st.mean(xs), st.mean(ys)
        ax.scatter([x - cx for x in xs], [y - cy for y in ys], s=14, color=TEAL, alpha=0.6,
                   edgecolors="none")
        ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("Δx from rest (px)"); ax.set_ylabel("Δy from rest (px)")
        ax.set_title(f"Idle tremor · {len(xs)} samples over {len(P['idles'])} idle periods",
                     fontweight="bold", color=INK, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(args.outdir, f"fig_idle_tremor.{ext}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {args.outdir}/fig_idle_tremor.png")
    else:
        print("[skip] idle tremor figure -- no tremor events "
              "(the run never sat idle long enough)")

    print(f"[csv] {args.outdir}/real_trace_stats.csv")

    missing = [k for k, v in (("keystrokes", P["keys"]), ("scroll", P["sticks"]),
                              ("idle", P["tremor"])) if not v]
    if missing:
        print(f"\nNOTE: no {', '.join(missing)} data in these traces. Panels render empty. "
              f"Use a battery whose tasks type, scroll, and pause -- the hard battery "
              f"(saucedemo checkout) exercises all three.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
