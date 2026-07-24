#!/usr/bin/env python3
"""
exp3_fitts_calibration.py
=========================
EXPERIMENT 3 — A proper Fitts's Law regression for the human-imitation mouse layer.

MOTIVATION (RESULTS_MATRIX.md Table 6): the real-run Fitts plot had n=5 points clustered in
ID 0.06-0.82, R^2=0.605 -- too weak to publish; a reviewer will flag it. The fix is to drive
the cursor through a SYSTEMATIC sweep of movements across a wide range of distances and target
sizes, so the index-of-difficulty axis is well-covered and the regression is meaningful.

WHAT THIS TESTS:
  Whether HumanController's movement time obeys the Shannon formulation of Fitts's Law,
      MT = a + b * ID,   ID = log2(D/W + 1)
  over a wide, evenly-sampled ID range -- and reports a,b,R^2 with enough n to be defensible.

METHOD:
  - Generates a grid of (distance, target_width) pairs spanning ID ~0.5 to ~7 bits.
  - For each, issues N repeats of a HumanController move of that distance to a target of that
    width, recording the actual movement time from the trace.
  - No web page or NetGent needed -- it drives the controller's mouse directly on the Xvfb
    display. So it needs a DISPLAY (run in the Docker/Xvfb wrapper) but NOT a browser or LLM.

This isolates the motor model from everything else: it's a pure measurement of the mouse
timing law, which is exactly what you want to cite MacKenzie for.

USAGE (in the Docker/Xvfb wrapper, DISPLAY set):
    python3 exp3_fitts_calibration.py -o /out/fitts_cal
    python3 exp3_fitts_calibration.py -o /out/fitts_cal --repeats 4 --profile default

OUTPUT: /out/fitts_cal/fitts_calibration.jsonl (raw), then run plot_human_trace.py on it, and
a printed regression summary (a, b, R^2, n).
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="results/exp3_fitts")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per (distance,width) cell")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--screen-w", type=int, default=1920)
    ap.add_argument("--screen-h", type=int, default=1080)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    trace_path = os.path.join(args.outdir, "fitts_calibration.jsonl")
    os.environ["HUMAN_TRACE"] = trace_path   # HumanController writes its trace here

    # import AFTER setting HUMAN_TRACE so the module picks it up
    import human_controller as HC
    HC.TRACE_PATH = trace_path
    HC._trace_fh = None

    # We drive the controller's mouse geometry directly. HumanController needs a driver for
    # element lookups, but _human_move_to only uses pyautogui -- so we can call it with a
    # dummy driver as long as we only use coordinate moves.
    class _DummyDriver:
        pass

    controller = HC.HumanController(_DummyDriver(), profile=args.profile)

    import pyautogui
    sw, sh = args.screen_w, args.screen_h
    cx, cy = sw // 2, sh // 2

    # sweep: distances and target widths chosen to spread ID = log2(D/W + 1) widely
    distances = [60, 120, 250, 450, 700, 1000, 1400]
    widths = [12, 24, 48, 96, 160]

    print(f"Fitts calibration sweep: {len(distances)}x{len(widths)} cells x {args.repeats} "
          f"= {len(distances)*len(widths)*args.repeats} moves")
    print(f"Writing trace to {trace_path}")

    cells = []
    for D in distances:
        for W in widths:
            ID = math.log2(D / max(8.0, W) + 1.0)
            cells.append((D, W, ID))
    cells.sort(key=lambda c: c[2])

    n_done = 0
    for D, W, ID in cells:
        for _ in range(args.repeats):
            # start at a random-ish anchor, move distance D horizontally (kept on-screen)
            start_x = cx - D // 2
            start_y = cy
            target_x = start_x + D
            # clamp
            start_x = max(50, min(sw - 50, start_x))
            target_x = max(50, min(sw - 50, target_x))
            try:
                pyautogui.moveTo(start_x, start_y, duration=0.05)
                controller._human_move_to(target_x, cy, target_w=W)
                n_done += 1
            except Exception as e:
                print(f"  move failed (D={D},W={W}): {e}")
        print(f"  ID={ID:.2f}  D={D:>4} W={W:>3}  done")

    print(f"\n{n_done} moves recorded. Now analysing...")

    # analyse the trace ourselves for the regression (plot_human_trace also works on this file)
    moves = _group_moves(trace_path)
    ids, mts = [], []
    for m in moves:
        meta, pts = m["meta"], m["pts"]
        D = math.hypot(meta["to"][0] - meta["from"][0], meta["to"][1] - meta["from"][1])
        if D < 5:
            continue
        W = max(8.0, meta.get("target_w", 40))
        ids.append(math.log2(D / W + 1.0))
        mts.append(pts[-1][2] - pts[0][2])

    if len(ids) >= 5:
        a, b, r2 = _linfit(ids, mts)
        print("\n" + "=" * 60)
        print("FITTS REGRESSION (Shannon formulation, MT = a + b*ID)")
        print("=" * 60)
        print(f"  n           = {len(ids)}")
        print(f"  ID range    = {min(ids):.2f} .. {max(ids):.2f} bits")
        print(f"  a intercept = {a:.4f} s")
        print(f"  b slope     = {b:.4f} s/bit")
        print(f"  R^2         = {r2:.4f}")
        print(f"\n  index of performance IP = 1/b = {1/b:.2f} bits/s" if b > 0 else "")
        if r2 > 0.8 and len(ids) >= 30:
            print("  -> Strong fit over a wide range. Publishable as a Fitts-conformance result.")
        elif r2 > 0.6:
            print("  -> Moderate fit. Consider more repeats to tighten it.")
        else:
            print("  -> Weak fit; check the trace / increase repeats.")
        with open(os.path.join(args.outdir, "fitts_regression.json"), "w") as f:
            json.dump({"n": len(ids), "a": a, "b": b, "r2": r2,
                       "id_min": min(ids), "id_max": max(ids)}, f, indent=2)
    else:
        print("Not enough moves captured to fit.")

    print(f"\nPlot with:  python3 plot_human_trace.py {trace_path} -o {args.outdir}/figs")


def _group_moves(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
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


def _linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = sum((xs[i] - mx) ** 2 for i in range(n))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    ss_tot = sum((ys[i] - my) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return a, b, r2


if __name__ == "__main__":
    main()
