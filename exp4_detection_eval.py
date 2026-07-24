#!/usr/bin/env python3
"""
exp4_detection_eval.py
======================
EXPERIMENT 4 — The honest "does the imitation actually help against detection?" test.

MOTIVATION (RESULTS_MATRIX.md MISSING #5): your BeCAPTCHA-Mouse citation reports 93% bot
detection from a SINGLE trajectory. So it is not enough to show the movements "look human" --
you have to show they're HARDER TO SEPARATE from human input than the stock controller's.

WHAT THIS TESTS:
  Given traces from (a) the stock PyAutoGUIController and (b) HumanController, how separable are
  they from each other and (ideally) from real human input? We quantify separability with simple,
  transparent features + a classifier. If a trivial classifier tells stock vs human apart at
  ~100% but struggles more with HumanController-vs-real-human, that's the honest, defensible
  claim: HumanController removes the OBVIOUS tells the stock controller has.

  This is deliberately a WHITE-BOX, reproducible detector (logistic regression on
  interpretable motion features), NOT a claim to beat production detectors. The paper framing is
  "reduces trivial separability", with BeCAPTCHA cited as the ceiling we do NOT clear.

FEATURES (per movement, all cheap + interpretable):
  - straightness (chord/path length)      stock ~1.0 (dead straight) is a giant tell
  - max perpendicular deviation / distance
  - velocity profile: peak/mean ratio, time-to-peak fraction (humans accelerate then decelerate)
  - number of direction reversals (submovements)
  - jerk proxy (mean abs 3rd difference)
FEATURES (per keystroke stream):
  - inter-key interval mean, std, CV      stock: CV ~0 (uniform) is a giant tell
  - fraction of intervals within 1ms of the mode

MODES:
  A) --from-traces stock.jsonl human.jsonl [real_human.jsonl]
       classify between provided trace files (real_human optional).
  B) --synth
       generate stock-style and human-style traces from the controllers' own math (no browser)
       for a quick, reproducible demonstration of separability. Good for a first result.

USAGE:
    python3 exp4_detection_eval.py --synth -o /out/exp4
    python3 exp4_detection_eval.py --from-traces stock.jsonl human_trace.jsonl -o /out/exp4

OUTPUT: printed separability report (accuracy, per-feature discriminability) + exp4_features.csv.
"""

import argparse
import json
import math
import os
import random
import sys
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
#  Feature extraction                                                         #
# --------------------------------------------------------------------------- #

def move_features(pts, start, end):
    """Interpretable motion features for one movement path (list of (x,y,t))."""
    if len(pts) < 3:
        return None
    full = [(start[0], start[1], pts[0][2])] + pts
    chord = math.hypot(end[0] - start[0], end[1] - start[1])
    if chord < 5:
        return None
    path_len = sum(math.hypot(full[i][0]-full[i-1][0], full[i][1]-full[i-1][1])
                   for i in range(1, len(full)))
    straightness = chord / path_len if path_len else 1.0

    # perpendicular deviation
    x0, y0 = start; x1, y1 = end
    L = math.hypot(x1-x0, y1-y0) or 1.0
    max_dev = max(abs((y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0)/L for (x, y, _t) in full)

    # velocity profile
    vs, ts = [], []
    for i in range(1, len(full)):
        dt = full[i][2] - full[i-1][2]
        d = math.hypot(full[i][0]-full[i-1][0], full[i][1]-full[i-1][1])
        if dt > 0:
            vs.append(d/dt); ts.append(full[i][2])
    if not vs:
        return None
    peak_v = max(vs); mean_v = statistics.mean(vs)
    peak_idx = vs.index(peak_v)
    time_to_peak_frac = peak_idx / len(vs)

    # direction reversals in x and y (submovement proxy)
    reversals = 0
    for i in range(2, len(full)):
        dx1 = full[i-1][0]-full[i-2][0]; dx2 = full[i][0]-full[i-1][0]
        if dx1 * dx2 < 0:
            reversals += 1

    return {
        "straightness": straightness,
        "dev_ratio": max_dev / chord,
        "peak_mean_v_ratio": peak_v / mean_v if mean_v else 1.0,
        "time_to_peak_frac": time_to_peak_frac,
        "reversals": reversals,
    }


def keystroke_features(intervals_ms):
    if len(intervals_ms) < 3:
        return None
    m = statistics.mean(intervals_ms)
    sd = statistics.pstdev(intervals_ms)
    cv = sd / m if m else 0.0
    # fraction near the mode (uniform typing clusters hard)
    rounded = [round(x) for x in intervals_ms]
    mode = max(set(rounded), key=rounded.count)
    near_mode = sum(1 for x in intervals_ms if abs(x - mode) <= 1) / len(intervals_ms)
    return {"ik_mean": m, "ik_cv": cv, "near_mode_frac": near_mode}


# --------------------------------------------------------------------------- #
#  Trace loading / synthesis                                                  #
# --------------------------------------------------------------------------- #

def moves_from_trace(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    moves, cur, keys = [], None, []
    for r in recs:
        if r.get("ev") == "move_start":
            if cur: moves.append(cur)
            cur = {"start": r["from"], "end": r["to"], "pts": []}
        elif r.get("ev") == "move_sample" and cur is not None:
            cur["pts"].append((r["x"], r["y"], r["t"]))
        elif r.get("ev") == "key" and r.get("delay_ms") is not None:
            keys.append(r["delay_ms"])
    if cur: moves.append(cur)
    return moves, keys


def synth_traces(n=60, seed=0):
    """Generate stock-style and human-style move sets from the real controller math, so this
    runs with no browser. Stock = straight linear moves + uniform keystroke interval."""
    import human_controller as HC
    rng = random.Random(seed)
    stock_moves, human_moves = [], []

    for _ in range(n):
        sx, sy = rng.uniform(100, 1800), rng.uniform(100, 1000)
        tx, ty = rng.uniform(100, 1800), rng.uniform(100, 1000)
        D = math.hypot(tx-sx, ty-sy)
        W = rng.choice([16, 32, 64, 120])

        # STOCK: straight line, constant velocity, fixed duration 0.5s (matches stock controller)
        steps = 25
        dur = 0.5
        stock_pts = []
        for i in range(1, steps+1):
            t = i/steps
            stock_pts.append((sx + (tx-sx)*t, sy + (ty-sy)*t, t*dur))
        stock_moves.append({"start": [sx, sy], "end": [tx, ty], "pts": stock_pts})

        # HUMAN: real Bezier + Fitts timing from the controller's own functions
        dur_h = HC.fitts_duration(D, target_w=W, rng=rng)
        c1, c2 = HC.random_bezier_controls(sx, sy, tx, ty, rng=rng)
        hsteps = max(12, int(D/22))
        hpts = []
        for i in range(1, hsteps+1):
            t = i/hsteps; u = 1-t
            bx = u**3*sx + 3*u**2*t*c1[0] + 3*u*t**2*c2[0] + t**3*tx
            by = u**3*sy + 3*u**2*t*c1[1] + 3*u*t**2*c2[1] + t**3*ty
            hpts.append((bx, by, t*dur_h))
        human_moves.append({"start": [sx, sy], "end": [tx, ty], "pts": hpts})

    # keystrokes
    stock_keys = [20.0] * 40   # stock controller: single fixed interval reused for every char
    human_keys = []
    prev = None
    for ch in ("the quick brown fox jumps over the lazy dog " * 2):
        human_keys.append(HC.keystroke_delay(prev, ch, rng=rng) * 1000); prev = ch
    return stock_moves, stock_keys, human_moves, human_keys


# --------------------------------------------------------------------------- #
#  Tiny logistic-regression classifier (no sklearn dependency)                #
# --------------------------------------------------------------------------- #

def logreg_cv(X, y, folds=5, iters=400, lr=0.1):
    """Standardize, then k-fold CV logistic regression. Returns mean accuracy + weights."""
    n, d = len(X), len(X[0])
    means = [statistics.mean(col) for col in zip(*X)]
    stds = [statistics.pstdev(col) or 1.0 for col in zip(*X)]
    Xs = [[(row[j]-means[j])/stds[j] for j in range(d)] for row in X]

    idx = list(range(n)); random.Random(0).shuffle(idx)
    fold_size = max(1, n // folds)
    accs = []
    for f in range(folds):
        test = set(idx[f*fold_size:(f+1)*fold_size])
        tr = [i for i in range(n) if i not in test]
        te = [i for i in range(n) if i in test]
        if not te: continue
        w = [0.0]*d; b = 0.0
        for _ in range(iters):
            for i in tr:
                z = b + sum(w[j]*Xs[i][j] for j in range(d))
                p = 1/(1+math.exp(-max(-30, min(30, z))))
                g = p - y[i]
                b -= lr*g
                for j in range(d):
                    w[j] -= lr*g*Xs[i][j]
        correct = 0
        for i in te:
            z = b + sum(w[j]*Xs[i][j] for j in range(d))
            correct += int((1 if z > 0 else 0) == y[i])
        accs.append(correct/len(te))
    return statistics.mean(accs) if accs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-traces", nargs="+", default=None,
                    help="stock.jsonl human.jsonl [real_human.jsonl]")
    ap.add_argument("--synth", action="store_true", help="synthesize traces (no browser)")
    ap.add_argument("-o", "--outdir", default="results/exp4_detection")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.synth:
        sm, sk, hm, hk = synth_traces(n=args.n)
        groups = {"stock": (sm, sk), "human": (hm, hk)}
    elif args.from_traces:
        labels = ["stock", "human", "real_human"]
        groups = {}
        for lab, path in zip(labels, args.from_traces):
            mv, ky = moves_from_trace(path)
            groups[lab] = (mv, ky)
    else:
        print("Provide --synth or --from-traces. See --help."); sys.exit(1)

    # feature matrices
    print("=" * 66)
    print("EXPERIMENT 4 — separability of controller output from human motion")
    print("=" * 66)

    # per-feature means, group by group
    feat_rows = {}
    for lab, (moves, keys) in groups.items():
        mf = [move_features(m["pts"], m["start"], m["end"]) for m in moves]
        mf = [f for f in mf if f]
        feat_rows[lab] = mf
        if mf:
            print(f"\n[{lab}]  ({len(mf)} moves)")
            for k in ("straightness", "dev_ratio", "peak_mean_v_ratio", "reversals"):
                vals = [f[k] for f in mf]
                print(f"    {k:<20} mean={statistics.mean(vals):.3f}")
        kf = keystroke_features(keys)
        if kf:
            print(f"    keystroke CV={kf['ik_cv']:.3f}  near-mode-frac={kf['near_mode_frac']:.3f}")

    # classify stock vs human (the core result)
    if "stock" in feat_rows and "human" in feat_rows:
        keys_order = ["straightness", "dev_ratio", "peak_mean_v_ratio", "time_to_peak_frac", "reversals"]
        X, y = [], []
        for lab, target in (("stock", 0), ("human", 1)):
            for f in feat_rows[lab]:
                X.append([f[k] for k in keys_order]); y.append(target)
        acc = logreg_cv(X, y)
        print("\n" + "-" * 66)
        print(f"CLASSIFIER: stock vs HumanController  ->  {acc*100:.1f}% CV accuracy")
        print("-" * 66)
        print("  Interpretation: HIGH accuracy here means the two are easily told apart -- which")
        print("  is EXPECTED and fine. The point of the paper's comparison is that the STOCK")
        print("  controller is trivially separable from HUMAN (straightness ~1.0, uniform keys),")
        print("  whereas HumanController's features overlap the human range. If you have a")
        print("  real_human trace, the meaningful number is human-vs-real_human accuracy:")
        print("  the LOWER that is, the better HumanController mimics real input.")
        if "real_human" in feat_rows and feat_rows["real_human"]:
            X2, y2 = [], []
            for lab, target in (("human", 0), ("real_human", 1)):
                for f in feat_rows[lab]:
                    X2.append([f[k] for k in keys_order]); y2.append(target)
            acc2 = logreg_cv(X2, y2)
            print(f"\n  CLASSIFIER: HumanController vs REAL human -> {acc2*100:.1f}% "
                  f"(closer to 50% = better mimicry)")

        # honest ceiling reminder
        print("\n  NOTE FOR PAPER: BeCAPTCHA-Mouse reports 93% detection of high-realism synthetic")
        print("  trajectories from a single sample. Frame this as 'removes trivial tells', NOT")
        print("  'defeats detection'.")

    # dump features
    import csv
    with open(os.path.join(args.outdir, "exp4_features.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "straightness", "dev_ratio", "peak_mean_v_ratio",
                    "time_to_peak_frac", "reversals"])
        for lab, mf in feat_rows.items():
            for f in mf:
                w.writerow([lab, f["straightness"], f["dev_ratio"], f["peak_mean_v_ratio"],
                            f["time_to_peak_frac"], f["reversals"]])
    print(f"\n[csv] {args.outdir}/exp4_features.csv")


if __name__ == "__main__":
    main()
