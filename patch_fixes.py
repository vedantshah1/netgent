#!/usr/bin/env python3
"""
patch_fixes.py — apply the four fixes the experiment round surfaced.

Idempotent: safe to run twice. Each patch verifies the target text exists before
replacing, and reports SKIP if already applied.

  1. human_controller.py : pyautogui.PAUSE = 0    (Exp 3 -- 6x timing inflation)
  2. human_controller.py : min-jerk velocity warp (Exp 4 -- peak/mean 1.12 -> ~1.875)
  3. netgent_planner.py  : NO_TERMINATE_ACTION WARN -> ERROR (Exp 5 -- fatal, not cosmetic)
  4. exp5_fewshot_ablation.py : task_success_rate bug (0.0 was a parse-failure artifact)

USAGE:  python3 patch_fixes.py [--dir .] [--dry-run]
"""
import argparse
import os
import sys

# --------------------------------------------------------------------------- #
PATCHES = []


def patch(fname, name, old, new):
    PATCHES.append({"file": fname, "name": name, "old": old, "new": new})


# --- 1 + 2: human_controller.py -------------------------------------------- #

patch(
    "human_controller.py",
    "PAUSE=0 (kill pyautogui's 0.1s per-call dead time)",
    old="""    def _human_move_to(self, tx, ty, target_w=40.0):""",
    new="""    # ------------------------------------------------------------------ #
    # pyautogui inserts a global PAUSE (default 0.1s) after EVERY call. With
    # 12-63 moveTo calls per movement that added 1.2-6.3s of dead time and made
    # measured movement time = steps * 0.1021s, completely overriding the
    # Fitts-derived duration (Exp 3: measured/intended ratio 6.0x mean, 14.3x max).
    # All human timing is modelled explicitly below, so PAUSE must be zero.
    # ------------------------------------------------------------------ #
    _PAUSE_DISABLED = False

    def _ensure_pause_disabled(self):
        if not HumanController._PAUSE_DISABLED:
            pyautogui.PAUSE = 0.0
            HumanController._PAUSE_DISABLED = True

    def _human_move_to(self, tx, ty, target_w=40.0):""",
)

patch(
    "human_controller.py",
    "min-jerk velocity profile + arc-length reparameterization",
    old="""        # sample the cubic Bezier and step the OS cursor along it; short per-segment
        # durations make the tween's velocity profile (slow-fast-slow) emerge naturally.
        steps = max(12, int(dist / 22))
        _trace({"ev": "move_start", "from": [sx, sy], "to": [tx, ty], "dist": dist,
                "target_w": target_w, "fitts_duration": duration,
                "ctrl1": list(c1), "ctrl2": list(c2), "steps": steps})
        for i in range(1, steps + 1):
            t = i / steps
            u = 1 - t
            bx = (u**3) * sx + 3 * (u**2) * t * c1[0] + 3 * u * (t**2) * c2[0] + (t**3) * tx
            by = (u**3) * sy + 3 * (u**2) * t * c1[1] + 3 * u * (t**2) * c2[1] + (t**3) * ty
            # micro-jitter perpendicular to travel, tapering to 0 near the target
            bx += self.rng.uniform(-1, 1) * (1 - t) * 1.2
            by += self.rng.uniform(-1, 1) * (1 - t) * 1.2
            pyautogui.moveTo(bx, by, duration=duration / steps)
            _trace({"ev": "move_sample", "x": bx, "y": by, "i": i, "steps": steps})""",
    new="""        self._ensure_pause_disabled()

        # Walking the Bezier at uniform parameter spacing with equal time per step
        # produces NEARLY CONSTANT SPEED (Exp 4 measured peak/mean = 1.117, versus
        # 1.875 for the minimum-jerk model that describes human pointing). The curve
        # bent in space but not in time. Two corrections:
        #   (a) arc-length reparameterize, so Bezier parameter != distance travelled
        #   (b) minimum-jerk time warp  s(tau) = 10t^3 - 15t^4 + 6t^5
        #       -> ds/dtau = 30 tau^2 (1-tau)^2, peaking at 1.875x mean at tau=0.5
        steps = max(12, int(dist / 22))

        def _bez(t):
            u = 1 - t
            return ((u**3) * sx + 3 * (u**2) * t * c1[0] + 3 * u * (t**2) * c2[0] + (t**3) * tx,
                    (u**3) * sy + 3 * (u**2) * t * c1[1] + 3 * u * (t**2) * c2[1] + (t**3) * ty)

        # (a) build a cumulative arc-length lookup table over the curve
        LUT_N = 256
        lut_t, lut_s, acc = [0.0], [0.0], 0.0
        px, py = _bez(0.0)
        for k in range(1, LUT_N + 1):
            tk = k / LUT_N
            qx, qy = _bez(tk)
            acc += math.hypot(qx - px, qy - py)
            lut_t.append(tk); lut_s.append(acc)
            px, py = qx, qy
        total_len = acc or 1.0

        def _t_at_arc_fraction(frac):
            \"\"\"Invert the arc-length LUT: what Bezier t gives this fraction of the path?\"\"\"
            target = frac * total_len
            lo, hi = 0, len(lut_s) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lut_s[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == 0:
                return 0.0
            s0, s1 = lut_s[lo - 1], lut_s[lo]
            w = 0.0 if s1 == s0 else (target - s0) / (s1 - s0)
            return lut_t[lo - 1] + w * (lut_t[lo] - lut_t[lo - 1])

        step_dt = duration / steps
        _trace({"ev": "move_start", "from": [sx, sy], "to": [tx, ty], "dist": dist,
                "target_w": target_w, "fitts_duration": duration,
                "ctrl1": list(c1), "ctrl2": list(c2), "steps": steps,
                "profile": "min_jerk_arclen", "step_dt": step_dt,
                "path_len": total_len})
        for i in range(1, steps + 1):
            tau = i / steps                                   # normalized TIME
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5        # (b) min-jerk arc fraction
            t = _t_at_arc_fraction(s)                          # (a) -> Bezier parameter
            bx, by = _bez(t)
            # micro-jitter perpendicular to travel, tapering to 0 near the target
            bx += self.rng.uniform(-1, 1) * (1 - tau) * 1.2
            by += self.rng.uniform(-1, 1) * (1 - tau) * 1.2
            pyautogui.moveTo(bx, by)          # instant; timing is modelled explicitly
            _trace({"ev": "move_sample", "x": bx, "y": by, "i": i, "steps": steps,
                    "tau": tau, "arc_frac": s})
            time.sleep(step_dt)""",
)

# --- 3: netgent_planner.py -------------------------------------------------- #

patch(
    "netgent_planner.py",
    "NO_TERMINATE_ACTION: WARN -> ERROR",
    old="""    if not any("terminat" in a.lower() for a in (last.get("actions") or [])):
        warn("Last state's actions don't mention 'Terminate'. NetGent's web agent "
             "needs an explicit terminate instruction to stop cleanly.", state=last_name)""",
    new="""    if not any("terminat" in a.lower() for a in (last.get("actions") or [])):
        # Promoted WARN -> ERROR on Exp 5 evidence: zero-shot generation omitted the
        # Terminate action in 35/35 workflows (few-shot 4/36, chi2=56.6) while
        # struct_valid_rate still read 0.97. A workflow that cannot signal completion
        # is functionally broken, so structural validity must not pass it.
        err("Last state's actions don't mention 'Terminate'. NetGent's web agent "
            "needs an explicit terminate instruction to stop cleanly.", state=last_name)""",
)

# --- 4: exp5_fewshot_ablation.py -------------------------------------------- #

patch(
    "exp5_fewshot_ablation.py",
    "task_success_rate: don't count parse failures when not in browser mode",
    old="""                per.append({"parse_ok": False, "n_err": 0, "state_count": None, "task_success": False})""",
    new="""                # task_success must stay None when we are NOT measuring the browser --
                # otherwise a parse failure becomes the ONLY record in the denominator and
                # the summary reports a bogus 0.0 task-success rate (it did exactly that).
                per.append({"parse_ok": False, "n_err": 0, "state_count": None,
                            "task_success": (False if browser else None)})""",
)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory containing the source files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    applied = skipped = failed = 0
    for p in PATCHES:
        path = os.path.join(args.dir, p["file"])
        if not os.path.exists(path):
            print(f"[MISS] {p['file']:28} -- file not found"); failed += 1; continue
        src = open(path, encoding="utf-8").read()
        if p["new"] in src:
            print(f"[SKIP] {p['file']:28} {p['name']} (already applied)"); skipped += 1; continue
        if p["old"] not in src:
            print(f"[FAIL] {p['file']:28} {p['name']}")
            print(f"       anchor text not found -- file may have drifted; patch by hand")
            failed += 1; continue
        if src.count(p["old"]) > 1:
            print(f"[FAIL] {p['file']:28} {p['name']} -- anchor is ambiguous ({src.count(p['old'])}x)")
            failed += 1; continue
        if not args.dry_run:
            open(path, "w", encoding="utf-8").write(src.replace(p["old"], p["new"], 1))
        print(f"[ OK ] {p['file']:28} {p['name']}"); applied += 1

    print(f"\napplied={applied} skipped={skipped} failed={failed}"
          f"{'  (dry run -- nothing written)' if args.dry_run else ''}")

    print("\nPOST-PATCH CHECKS:")
    print("  1. human_controller.py must import `time` and `math` at module level.")
    print("     grep -n '^import time' human_controller.py")
    print("     grep -n '^import math' human_controller.py")
    print("  2. netgent_planner.py: confirm err() is in scope at that point in lint_workflow.")
    print("     grep -n 'def err' netgent_planner.py")
    print("  3. Re-run Exp 3 -- R2 should jump from 0.43 toward >0.8.")
    print("  4. Re-run Exp 4 --synth -- peak_mean_v_ratio should move 1.117 -> ~1.8.")
    print("  5. WARNING: promoting NO_TERMINATE to ERROR will DROP zero-shot struct")
    print("     validity from 0.97 to ~0.00 in Exp 5. That is the correct, intended")
    print("     result -- report it as such, and re-run Exp 5 so the table is consistent.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
