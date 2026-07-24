#!/usr/bin/env python3
"""
patch_imitation_v2.py — the remaining human-imitation fixes.

Apply AFTER patch_fixes.py (which handles PAUSE=0 and the min-jerk velocity warp).

WHAT THIS FIXES — the "too consistent" signature:
  Exp 3 and Exp 4 independently measured the same defect from different code paths:
  path curvature scales PERFECTLY linearly with distance, so sagitta/D is a near-constant 0.10.

      D (px)     60     120    250    450    700   1000   1400
      sagitta/D  .1053  .1040  .1035  .1042  .1032  .1014  .0990

  Flat to three decimals across a 23x range, and Exp 4 put straightness CV at 0.77% over 60
  movements. Real human curvature does not do this. A detector does not need to know anything
  about Bezier curves to catch it -- "this ratio is suspiciously stable" is exactly the
  software-level attribute-consistency analysis FP-Scanner uses. The imitation currently defeats
  naive checks while introducing a new, cleaner fingerprint.

THE FIX (three parts, all in random_bezier_controls):
  1. curviness becomes DISTANCE-DEPENDENT -- short corrective movements are relatively straighter
     than long sweeping ones, which is what human pointing actually looks like.
  2. the per-move multiplier widens from uniform(0.5, 1.0) to a lognormal-ish spread, so the
     ratio has a heavy tail instead of a tight box.
  3. the two control points get INDEPENDENT offsets (still same side, to avoid an S-wiggle),
     so the arc is no longer perfectly symmetric about its midpoint.

Target: sagitta/D coefficient of variation above ~25% and a visible slope across distance, versus
the current ~4%. Verify with analyze_real_traces.py after re-running Exp 3.

USAGE:  python3 patch_imitation_v2.py [--dir .] [--dry-run] [--revert]
"""
import argparse
import os
import sys

TARGET = "human_controller.py"

OLD = """def random_bezier_controls(x0, y0, x1, y1, curviness=0.18, rng=random):"""

NEW = """def _curviness_for_distance(dist, base=0.18, rng=random):
    \"\"\"Distance-dependent curvature.

    A constant `curviness` makes sagitta/D identical at every distance, which Exp 3 measured at
    0.099-0.105 across a 23x distance range (CV ~4%). That flatness is itself a fingerprint.
    Human pointing is relatively straighter over short corrective moves and bows more over long
    sweeps, so scale the base by distance and give it a heavy-tailed multiplier.
    \"\"\"
    # 0.55x at ~50px rising to ~1.6x at long range -- gentle, monotone, bounded
    scale = 0.55 + 0.90 * math.log10(max(dist, 20.0) / 20.0) / math.log10(70.0)
    scale = max(0.45, min(1.6, scale))
    # heavy-tailed per-move multiplier (was a tight uniform(0.5, 1.0) box)
    mult = math.exp(rng.gauss(0.0, 0.45))
    mult = max(0.25, min(2.6, mult))
    return base * scale * mult


def random_bezier_controls(sx, sy, tx, ty, curviness=0.18, rng=random):"""

OLD2 = """    off1 = side * dist * curviness * rng.uniform(0.5, 1.0)
    off2 = side * dist * curviness * rng.uniform(0.5, 1.0)"""

NEW2 = """    # Independent draws per control point (same side, so the path stays a single smooth arc
    # rather than an S) -- this breaks the near-perfect symmetry the old shared draw produced.
    off1 = side * dist * _curviness_for_distance(dist, curviness, rng)
    off2 = side * dist * _curviness_for_distance(dist, curviness, rng)"""

PATCHES = [
    {"name": "add _curviness_for_distance()", "old": OLD, "new": NEW,
     "sentinel": "def _curviness_for_distance("},
    {"name": "use distance-dependent curvature for both control points",
     "old": OLD2, "new": NEW2,
     "sentinel": "_curviness_for_distance(dist, curviness, rng)"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path = os.path.join(args.dir, TARGET)
    if not os.path.exists(path):
        print(f"[MISS] {TARGET} not found in {args.dir}")
        return 1

    src = open(path, encoding="utf-8").read()
    applied = skipped = failed = 0

    for p in (list(reversed(PATCHES)) if args.revert else PATCHES):
        is_applied = p["sentinel"] in src
        if is_applied == (not args.revert):
            print(f"[SKIP] {p['name']} (already {'applied' if is_applied else 'reverted'})")
            skipped += 1
            continue
        old, new = (p["new"], p["old"]) if args.revert else (p["old"], p["new"])
        if old not in src:
            print(f"[FAIL] {p['name']} -- anchor not found")
            failed += 1
            continue
        if src.count(old) > 1:
            print(f"[FAIL] {p['name']} -- anchor ambiguous ({src.count(old)}x)")
            failed += 1
            continue
        src = src.replace(old, new, 1)
        print(f"[ OK ] {p['name']} {'reverted' if args.revert else 'applied'}")
        applied += 1

    if not args.dry_run and applied:
        open(path, "w", encoding="utf-8").write(src)

    print(f"\napplied={applied} skipped={skipped} failed={failed}"
          f"{'  (dry run -- nothing written)' if args.dry_run else ''}")

    if not args.revert and failed == 0:
        print("""
WHAT TO CHECK AFTER RE-RUNNING EXP 3:
  sagitta/D coefficient of variation   was ~4%    -> target >25%
  sagitta/D vs distance                was flat   -> target a visible upward slope
  straightness CV (Exp 4)              was 0.77%  -> target >5%
  peak/mean velocity (Exp 4)           was 1.117  -> target ~1.8 (from patch_fixes.py)

  analyze_real_traces.py prints all four. If curvature CV is still under 10%, raise the gauss
  sigma in _curviness_for_distance from 0.45 toward 0.7 -- but change it ONCE and re-freeze.
""")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
