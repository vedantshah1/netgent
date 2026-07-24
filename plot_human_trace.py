#!/usr/bin/env python3
"""
plot_human_trace.py
-------------------
Verify the human-imitation layer is ACTUALLY working, using the trace written by
human_controller.py (instead of trying to eyeball a cursor in noVNC).

Usage:
    # 1. record a trace during a run
    export HUMAN_TRACE=/out/human_trace.jsonl
    python3 run_planner.py --api-keys /keys.json ...

    # 2. analyse it
    python3 plot_human_trace.py /out/human_trace.jsonl -o /out/human_figs

It answers four questions with numbers, not vibes:
  1. Are mouse paths CURVED (Bezier) rather than straight lines?
     -> curvature ratio = path_length / straight_line_distance.  1.00 == straight (bot).
  2. Is each path a DIFFERENT curve (randomized control points)?
     -> spread of curvature ratios across moves. A fixed curve => ~0 variance.
  3. Does movement time scale with Fitts's Law (distance/target size)?
     -> regression of duration against the Fitts index of difficulty, report R^2.
  4. Are keystrokes non-uniform (real inter-key variance, digraph speedups)?
     -> mean/std/CV of inter-key delays. A bot has std ~= 0.

Also emits figures you can drop straight into the paper:
  mouse_paths.png    - the actual recorded cursor arcs
  fitts.png          - duration vs index of difficulty + fit
  keystrokes.png     - histogram of inter-key intervals
"""

import json
import math
import sys
import os
import argparse
from collections import defaultdict


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def group_moves(recs):
    """Split the trace into individual movements: each move_start + following samples."""
    moves, cur = [], None
    for r in recs:
        if r.get("ev") == "move_start":
            if cur:
                moves.append(cur)
            cur = {"meta": r, "pts": []}
        elif r.get("ev") == "move_sample" and cur is not None:
            cur["pts"].append((r["x"], r["y"], r["t"]))
    if cur:
        moves.append(cur)
    return [m for m in moves if len(m["pts"]) >= 3]


def path_length(pts):
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
               for i in range(1, len(pts)))


def max_deviation(pts, p0, p1):
    """Max perpendicular distance of the sampled path from the straight line p0->p1.

    This is the honest measure of "is it an arc". Path-length ratio is a poor proxy: a
    shallow arc is only ~2-3% longer than its chord, which is lost in sampling noise.
    Deviation in pixels is directly interpretable (and is what a detector would see).
    """
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return 0.0
    best = 0.0
    for (x, y, _t) in pts:
        # perpendicular distance from point to the infinite line through p0,p1
        d = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / L
        best = max(best, d)
    return best


def analyse(recs):
    moves = group_moves(recs)
    keys = [r for r in recs if r.get("ev") == "key"]
    typos = [r for r in recs if r.get("ev") == "typo"]

    print("=" * 72)
    print(f"TRACE SUMMARY: {len(moves)} mouse moves, {len(keys)} keystrokes, {len(typos)} typos")
    print("=" * 72)

    if moves:
        ratios, ids, durs, devs = [], [], [], []
        for m in moves:
            meta, pts = m["meta"], m["pts"]
            straight = math.hypot(meta["to"][0] - meta["from"][0],
                                  meta["to"][1] - meta["from"][1])
            if straight < 5:
                continue
            # include the true start point: samples begin at i=1, so path length alone
            # under-measures. Prepend origin before measuring.
            full = [(meta["from"][0], meta["from"][1], pts[0][2])] + pts
            ratio = path_length(full) / straight
            ratios.append(ratio)
            devs.append(max_deviation(full, meta["from"], meta["to"]) / straight)
            w = max(8.0, meta.get("target_w", 40))
            ids.append(math.log2(straight / w + 1.0))   # Shannon formulation (MacKenzie)
            durs.append(pts[-1][2] - pts[0][2])

        if ratios:
            mean_r = sum(ratios) / len(ratios)
            var_r = sum((x - mean_r) ** 2 for x in ratios) / len(ratios)
            std_r = math.sqrt(var_r)
            md = sum(devs) / len(devs)
            sd = math.sqrt(sum((x - md) ** 2 for x in devs) / len(devs))
            print("\n[1] CURVATURE")
            print(f"    max perpendicular deviation from straight line, as fraction of distance:")
            print(f"      mean={md*100:.2f}%  min={min(devs)*100:.2f}%  max={max(devs)*100:.2f}%")
            print(f"    path_length/chord ratio: mean={mean_r:.4f} (a shallow arc is only ~2-3% longer)")
            print(f"    VERDICT: {'PASS - paths bow away from the straight line' if md > 0.01 else 'FAIL - paths are straight (Bezier not applied)'}")
            print("\n[2] RANDOMIZATION (does each move get its OWN arc?)")
            print(f"    deviation std across moves = {sd*100:.2f}% of distance")
            print(f"    VERDICT: {'PASS - arcs differ per move' if sd > 0.005 else 'FAIL - every move uses the SAME fixed curve'}")

            # Fitts regression: duration = a + b*ID
            n = len(ids)
            if n >= 3:
                mx = sum(ids) / n
                my = sum(durs) / n
                sxy = sum((ids[i] - mx) * (durs[i] - my) for i in range(n))
                sxx = sum((ids[i] - mx) ** 2 for i in range(n))
                b = sxy / sxx if sxx else 0.0
                a = my - b * mx
                ss_res = sum((durs[i] - (a + b * ids[i])) ** 2 for i in range(n))
                ss_tot = sum((durs[i] - my) ** 2 for i in range(n))
                r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
                print("\n[3] FITTS'S LAW, Shannon formulation  (MT = a + b*log2(D/W + 1))")
                print(f"    a={a:.3f}s  b={b:.3f}s/bit  R^2={r2:.3f}  (n={n})")
                print(f"    VERDICT: {'PASS - duration scales with difficulty' if b > 0 and r2 > 0.3 else 'WEAK - check sample size / spread'}")

    if keys:
        ds = [k["delay_ms"] for k in keys]
        mean_d = sum(ds) / len(ds)
        std_d = math.sqrt(sum((x - mean_d) ** 2 for x in ds) / len(ds))
        cv = std_d / mean_d if mean_d else 0
        cps = 1000.0 / mean_d if mean_d else 0
        print("\n[4] KEYSTROKE TIMING")
        print(f"    mean={mean_d:.1f}ms  std={std_d:.1f}ms  CV={cv:.3f}  ~{cps/5*60:.0f} WPM")
        print(f"    VERDICT: {'PASS - non-uniform, human-like rhythm' if cv > 0.15 else 'FAIL - near-constant interval (robotic)'}")
        # digraph check
        fast = [k["delay_ms"] for k in keys if k.get("prev") and (str(k["prev"]) + str(k["ch"])).lower() in
                {"th","he","in","er","an","re","on","at","en","nd","ti","es","or","te"}]
        if fast:
            mf = sum(fast) / len(fast)
            print(f"    common digraphs: mean={mf:.1f}ms vs overall {mean_d:.1f}ms "
                  f"({'PASS - rolls are faster' if mf < mean_d else 'no speedup seen'})")
    print()
    return moves, keys


def plot(moves, keys, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figures. "
              "pip install matplotlib --break-system-packages")
        return
    os.makedirs(outdir, exist_ok=True)

    if moves:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in moves[:40]:
            xs = [p[0] for p in m["pts"]]
            ys = [p[1] for p in m["pts"]]
            ax.plot(xs, ys, lw=1.1, alpha=0.85)
            ax.plot([m["meta"]["from"][0], m["meta"]["to"][0]],
                    [m["meta"]["from"][1], m["meta"]["to"][1]],
                    ls=":", lw=0.7, color="grey", alpha=0.5)
        ax.set_title("Recorded cursor paths (solid) vs straight line (dotted)")
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.invert_yaxis()
        fig.tight_layout(); fig.savefig(f"{outdir}/mouse_paths.png", dpi=150)
        print(f"[plot] wrote {outdir}/mouse_paths.png")

        ids, durs = [], []
        for m in moves:
            meta, pts = m["meta"], m["pts"]
            d = math.hypot(meta["to"][0] - meta["from"][0], meta["to"][1] - meta["from"][1])
            if d < 5:
                continue
            w = max(8.0, meta.get("target_w", 40))
            ids.append(math.log2(d / w + 1.0))   # Shannon formulation (MacKenzie)
            durs.append(pts[-1][2] - pts[0][2])
        if ids:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(ids, durs, s=18, alpha=0.7)
            ax.set_xlabel("Index of difficulty  log2(D/W + 1)  [bits]")
            ax.set_ylabel("Movement time (s)")
            ax.set_title("Fitts's Law: movement time vs difficulty")
            fig.tight_layout(); fig.savefig(f"{outdir}/fitts.png", dpi=150)
            print(f"[plot] wrote {outdir}/fitts.png")

    if keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([k["delay_ms"] for k in keys], bins=30)
        ax.set_xlabel("Inter-key interval (ms)"); ax.set_ylabel("count")
        ax.set_title("Keystroke timing distribution")
        fig.tight_layout(); fig.savefig(f"{outdir}/keystrokes.png", dpi=150)
        print(f"[plot] wrote {outdir}/keystrokes.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("-o", "--outdir", default="human_figs")
    a = ap.parse_args()
    recs = load(a.trace)
    if not recs:
        print(f"No records in {a.trace}. Did you export HUMAN_TRACE before the run?")
        sys.exit(1)
    m, k = analyse(recs)
    plot(m, k, a.outdir)
