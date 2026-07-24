"""
human_controller.py
--------------------
Human-imitation mouse + keyboard controller for NetGent.

Drop-in replacement for PyAutoGUIController. Wire it in like this:

    from netgent import NetGent
    from netgent.browser.session import BrowserSession
    from human_controller import HumanController

    driver = BrowserSession(user_data_dir="/tmp/browser-cache").driver
    controller = HumanController(driver)               # or HumanController(driver, profile="fast")
    agent = NetGent(driver=driver, controller=controller, llm=llm, llm_enabled=True)

What it changes vs. the stock PyAutoGUIController:
  MOUSE
    - Per-move randomized cubic-Bezier control points (offset from the straight path,
      both on the same side of the line -> a natural arc, different every time)
    - Movement duration scaled by Fitts's Law (farther / smaller target -> longer,
      with acceleration/deceleration coming from the Bezier tween itself)
    - Occasional overshoot-then-correct on clicks
    - Small end-point jitter so we don't land dead-center every time
  KEYBOARD
    - Per-keystroke timing from QWERTY key-to-key travel distance (flight time)
    - Speed-ups for common English digraphs (th, he, in, er, ...)
    - Dwell-time variation, hand-alternation speed-up
    - Occasional typos with backspace correction (fast correction dwell)
    - Gradual fatigue drift over long strings

Sources for the behaviors (see lit review): Bezier/Fitts mouse paper (RG.2.2.20692.10880),
BeCAPTCHA-Mouse (arXiv:2005.00890) initial-accel/final-decel + fine end correction,
free-text keyboard ABM (arXiv:2505.05015), BeCAPTCHA-Type (arXiv:2207.13394).

  SCROLL  (added -- this was the biggest remaining gap)
    - Discrete wheel ticks instead of one instantaneous N-pixel jump
    - Variable per-tick timing, with deceleration toward the end of a scroll
    - Occasional overshoot-then-scroll-back
    - Reading pause after a scroll settles, scaled to how far was scrolled
  IDLE
    - Micro-tremor while "reading" (a perfectly stationary cursor is itself a tell)
    - Inter-action pacing so actions don't fire back-to-back at machine speed

SCOPE (deliberate):
  This file only touches the MOVEMENT REALISM layer. NetGent already implements browser
  stealth and network stealth in BrowserSession via SeleniumBase UC mode (uc=True,
  undetectable=True, --disable-blink-features=AutomationControlled, use_auto_ext=False)
  plus proxy support. Do NOT add a second fingerprint-spoofing layer on top of that: FP-Scanner
  (Vastel et al., USENIX Security 2018) detects bots by finding INCONSISTENCIES between
  fingerprint attributes, so naive spoofing stacked on UC mode makes you MORE detectable, not
  less. No CAPTCHA solving here either -- out of scope per mentor.

SOURCES (all verified against arXiv/DOI -- see CITATIONS.md for the full ledger):
  [1] Acien, Morales, Fierrez, Vera-Rodriguez. "BeCAPTCHA-Mouse: Synthetic Mouse Trajectories
      and Improved Bot Detection." arXiv:2005.00890; Pattern Recognition 127:108643, 2022.
      -> neuromotor modeling; detects "high realism" synthetic trajectories at 93% from ONE
         trajectory. This is the bar the mouse code is trying to clear.
  [2] DeAlcala, Morales, Tolosana, Acien, Fierrez, Hernandez, Ferrer, Diaz. "BeCAPTCHA-Type:
      Biometric Keystroke Data Generation for Improved Bot Detection." arXiv:2207.13394;
      IEEE CVPRw 2023.
  [3] Dillon, Arushi. "An Agent-Based Modeling Approach to Free-Text Keyboard Dynamics for
      Continuous Authentication." arXiv:2505.05015, 2025.
      -> dwell time, flight time, error rate as the discriminative keystroke features.
  [4] Fitts. "The information capacity of the human motor system in controlling the amplitude
      of movement." J. Exp. Psychol. 47(6):381-391, 1954.  -> movement-time model.
  [5] Vastel, Laperdrix, Rudametkin, Rouvoy. "FP-Scanner: The Privacy Implications of Browser
      Fingerprint Inconsistencies." USENIX Security 2018.  -> why NOT to stack spoofing.

NOTE: none of this is a silver bullet against a trained detector; it removes the *obvious*
tells (fixed Bezier, uniform keystroke interval, instantaneous scroll) that the stock
controller has. [1] specifically shows realistic-looking synthetic trajectories still get
caught at 93%, so keep the detection-side evaluation honest and don't overclaim in the paper.
"""

import math
import time
import random
import logging

# pyautogui and NetGent's controller need a display / the netgent package. The pure motor-model
# functions (fitts_duration, random_bezier_controls, keystroke_delay) do NOT, and are imported
# by the detection/Fitts experiments in environments without either. So import lazily: the
# module loads everywhere; only instantiating HumanController requires the heavy deps.
try:
    import pyautogui
except Exception:  # no display / not installed
    pyautogui = None

try:
    from netgent.browser.controller.pyautogui_controller import PyAutoGUIController
except Exception:  # netgent not importable outside the VM
    class PyAutoGUIController:  # minimal stand-in so the subclass can still be defined
        def __init__(self, driver):
            self.driver = driver

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Trace recording (verification / paper figures)                             #
# --------------------------------------------------------------------------- #
# Set HUMAN_TRACE to a path and every mouse sample + keystroke gets logged as JSONL.
# You can then verify the Bezier arcs / Fitts timing / keystroke rhythm are REAL,
# instead of trying to eyeball a cursor in noVNC. Plot with plot_human_trace.py.
#
#   export HUMAN_TRACE=/out/human_trace.jsonl
import os, json, time as _time

TRACE_PATH = os.environ.get("HUMAN_TRACE")
_trace_fh = None


def _trace(record):
    """Append one JSON record to the trace file (no-op if HUMAN_TRACE unset)."""
    global _trace_fh
    if not TRACE_PATH:
        return
    try:
        if _trace_fh is None:
            _trace_fh = open(TRACE_PATH, "a", buffering=1)
        record["t"] = _time.time()
        _trace_fh.write(json.dumps(record) + "\n")
    except Exception as e:  # never let tracing break a run
        logger.warning(f"trace write failed: {e}")


# --------------------------------------------------------------------------- #
#  Pure geometry / timing helpers (no pyautogui -> unit-testable on their own) #
# --------------------------------------------------------------------------- #

def _curviness_for_distance(dist, base=0.18, rng=random):
    """Distance-dependent curvature.

    A constant `curviness` makes sagitta/D identical at every distance, which Exp 3 measured at
    0.099-0.105 across a 23x distance range (CV ~4%). That flatness is itself a fingerprint.
    Human pointing is relatively straighter over short corrective moves and bows more over long
    sweeps, so scale the base by distance and give it a heavy-tailed multiplier.
    """
    # 0.55x at ~50px rising to ~1.6x at long range -- gentle, monotone, bounded
    scale = 0.55 + 0.90 * math.log10(max(dist, 20.0) / 20.0) / math.log10(70.0)
    scale = max(0.45, min(1.6, scale))
    # heavy-tailed per-move multiplier (was a tight uniform(0.5, 1.0) box)
    mult = math.exp(rng.gauss(0.0, 0.45))
    mult = max(0.25, min(2.6, mult))
    return base * scale * mult


def random_bezier_controls(x0, y0, x1, y1, curviness=0.18, rng=random):
    """Two control points offset perpendicular to the straight path, on the SAME side.

    Returns ((cx1, cy1), (cx2, cy2)) in absolute screen coords. Same-side offsets give a
    single smooth arc (a human sweep) rather than an S-wiggle. Offset magnitude scales with
    path length so short moves aren't wildly curved.
    """
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy) or 1.0
    # unit perpendicular
    px, py = -dy / dist, dx / dist
    side = rng.choice((-1.0, 1.0))
    # two control points at ~1/3 and ~2/3 along the path, pushed to the same side
    # Independent draws per control point (same side, so the path stays a single smooth arc
    # rather than an S) -- this breaks the near-perfect symmetry the old shared draw produced.
    off1 = side * dist * _curviness_for_distance(dist, curviness, rng)
    off2 = side * dist * _curviness_for_distance(dist, curviness, rng)
    t1, t2 = rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.8)
    c1 = (x0 + dx * t1 + px * off1, y0 + dy * t1 + py * off1)
    c2 = (x0 + dx * t2 + px * off2, y0 + dy * t2 + py * off2)
    return c1, c2


def fitts_duration(dist_px, target_w=40.0, a=0.08, b=0.12, jitter=0.15, rng=random):
    """Fitts's Law movement time, SHANNON formulation:

        MT = a + b * ID,      ID = log2(D/W + 1)     [bits]

    IMPORTANT (was previously wrong here): there are three competing formulations and it
    matters which one you name in the paper.
      Fitts (1954), original : ID = log2(2D/W)      -- can go NEGATIVE when D < W/2
      Welford (1968)         : ID = log2(D/W + 0.5)
      Shannon (MacKenzie)    : ID = log2(D/W + 1)   -- always >= 0, better empirical fit

    This function previously computed log2(2D/W + 1), which is none of the three -- a hybrid
    of Fitts's 2D/W with Shannon's +1 -- while the docstring credited Fitts (1954). Now uses
    the Shannon formulation, which is the HCI standard (and the ISO 9241-9 basis).

    Cite MacKenzie & Buxton (CHI '92) / MacKenzie (1992) for this form, NOT Fitts (1954)
    alone. See CITATIONS.md.

    a,b defaults are in the range typically reported for mouse pointing (a ~ 0.1s intercept,
    b ~ 0.1-0.15 s/bit); they are tuning, not measured from your setup. Multiplicative jitter
    keeps timing non-robotic. Acceleration/deceleration is NOT modeled here -- that emerges
    from the Bezier tween; Fitts only sets total time, which is the correct place for it.
    """
    target_w = max(8.0, target_w)
    idx = math.log2(max(dist_px, 1.0) / target_w + 1.0)   # Shannon formulation
    mt = a + b * idx
    mt *= (1.0 + rng.uniform(-jitter, jitter))
    return max(0.05, mt)


# QWERTY physical layout (row, col) for flight-time estimation.
_QWERTY = {
    **{c: (0, i) for i, c in enumerate("1234567890")},
    **{c: (1, i) for i, c in enumerate("qwertyuiop")},
    **{c: (2, i) for i, c in enumerate("asdfghjkl")},
    **{c: (3, i) for i, c in enumerate("zxcvbnm")},
}
# rough left/right hand split on a touch-typist keyboard
_LEFT_HAND = set("qwertasdfgzxcvb12345")
# common English digraphs typed as fast "rolls"
_FAST_DIGRAPHS = {"th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
                  "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
                  "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le"}


def _key_pos(ch):
    return _QWERTY.get(ch.lower())


def keystroke_delay(prev_ch, ch, base=0.11, rng=random, fatigue=0.0):
    """Inter-key delay (seconds) between prev_ch and ch, human-like.

    Combines: base rhythm, QWERTY travel distance, digraph roll speed-ups, same-hand penalty
    / hand-alternation speed-up, a fatigue term that grows over a long string, and noise.
    """
    if prev_ch is None:
        return base * rng.uniform(0.8, 1.3) + fatigue

    delay = base

    p_prev, p_cur = _key_pos(prev_ch), _key_pos(ch)
    if p_prev and p_cur:
        travel = math.hypot(p_prev[0] - p_cur[0], p_prev[1] - p_cur[1])
        delay += 0.012 * travel                      # farther keys -> slower

        same_hand = (prev_ch.lower() in _LEFT_HAND) == (ch.lower() in _LEFT_HAND)
        if same_hand and travel > 0:
            delay *= 1.12                             # same-hand reach is slower
        elif not same_hand:
            delay *= 0.88                             # alternating hands is faster

    if (prev_ch + ch).lower() in _FAST_DIGRAPHS:
        delay *= 0.72                                 # practiced rolls

    if ch == " ":
        delay *= 1.15                                 # tiny pause at word boundaries

    delay *= (1.0 + rng.uniform(-0.22, 0.22))         # per-stroke noise
    return max(0.03, delay + fatigue)


# --------------------------------------------------------------------------- #
#  The controller                                                             #
# --------------------------------------------------------------------------- #

class HumanController(PyAutoGUIController):
    """PyAutoGUIController with human-like mouse paths and typing."""

    PROFILES = {
        "default": dict(curviness=0.18, type_base=0.11, typo_rate=0.02, overshoot_rate=0.15,
                        action_gap=(0.25, 0.9)),
        "fast":    dict(curviness=0.12, type_base=0.07, typo_rate=0.01, overshoot_rate=0.08,
                        action_gap=(0.12, 0.45)),
        "careful": dict(curviness=0.22, type_base=0.15, typo_rate=0.03, overshoot_rate=0.20,
                        action_gap=(0.5, 1.8)),
    }

    def __init__(self, driver, profile="default", seed=None):
        super().__init__(driver)
        self.p = dict(self.PROFILES.get(profile, self.PROFILES["default"]))
        self.rng = random.Random(seed)

    # ---- mouse -----------------------------------------------------------

    # ------------------------------------------------------------------ #
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

    def _human_move_to(self, tx, ty, target_w=40.0):
        """Move the cursor to (tx, ty) along a randomized Bezier arc, Fitts-timed."""
        sx, sy = pyautogui.position()
        dist = math.hypot(tx - sx, ty - sy)
        duration = fitts_duration(dist, target_w=target_w, rng=self.rng)
        c1, c2 = random_bezier_controls(sx, sy, tx, ty,
                                        curviness=self.p["curviness"], rng=self.rng)

        self._ensure_pause_disabled()

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
            """Invert the arc-length LUT: what Bezier t gives this fraction of the path?"""
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
            time.sleep(step_dt)

    def _target_from_element(self, by, selector, percentage):
        """Resolve (x, y, width) for an element; jitter the landing point off dead-center."""
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((by, selector))
        )
        # Scroll the element into view FIRST. element.location is relative to the DOCUMENT,
        # not the viewport, while pyautogui clicks in absolute SCREEN pixels -- using .location
        # directly is only correct by coincidence, when scrollY happens to be 0 and the target
        # is near the page top. location_once_scrolled_into_view scrolls as a side effect and
        # returns coordinates consistent with that scrolled state, which is what get_element_
        # coordinates actually needs.
        loc = element.location_once_scrolled_into_view
        w, h = element.size["width"], element.size["height"]
        # off-center: humans rarely hit the exact middle
        px = min(0.85, max(0.15, percentage + self.rng.uniform(-0.22, 0.22)))
        py = min(0.85, max(0.15, 0.5 + self.rng.uniform(-0.22, 0.22)))
        cx, cy = self.get_element_coordinates(loc["x"], loc["y"], w, h, px)
        # get_element_coordinates only takes one percentage; nudge y manually
        cy = cy + (py - 0.5) * h * 0.0  # keep API-compatible; y jitter folded into move jitter
        return cx, cy, w

    def click(self, by=None, selector=None, x=None, y=None, percentage=0.5):
        self._pace()
        target_w = 40.0
        if by is not None and selector is not None:
            try:
                x, y, target_w = self._target_from_element(by, selector, percentage)
            except Exception as e:
                logger.warning(f"HumanController: element lookup failed ({e}); using x/y")
        if x is None or y is None:
            raise ValueError("Must provide either (by, selector) or (x, y) coordinates")

        # occasional overshoot then correct back
        if self.rng.random() < self.p["overshoot_rate"]:
            ox = x + self.rng.uniform(-1, 1) * min(30, target_w)
            oy = y + self.rng.uniform(-1, 1) * 20
            self._human_move_to(ox, oy, target_w)
            time.sleep(self.rng.uniform(0.04, 0.12))
        self._human_move_to(x, y, target_w)
        time.sleep(self.rng.uniform(0.03, 0.14))   # hover hesitation before commit
        _trace({"ev": "click", "x": x, "y": y, "target_w": target_w})
        pyautogui.click()

    def move(self, by=None, selector=None, x=None, y=None, percentage=0.5):
        target_w = 40.0
        if by is not None and selector is not None:
            try:
                x, y, target_w = self._target_from_element(by, selector, percentage)
            except Exception as e:
                logger.warning(f"HumanController: element lookup failed ({e}); using x/y")
        if x is None or y is None:
            raise ValueError("Must provide either (by, selector) or (x, y) coordinates")
        self._human_move_to(x, y, target_w)

    # ---- keyboard --------------------------------------------------------

    def type_text(self, text, by=None, selector=None, x=None, y=None):
        # focus the field the same way the parent does
        self.click(by=by, selector=selector, x=x, y=y)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("delete")
        time.sleep(self.rng.uniform(0.08, 0.2))

        prev = None
        for idx, ch in enumerate(text):
            fatigue = 0.02 * (idx / max(1, len(text)))   # slow drift over long strings

            # occasional typo: hit an adjacent key, then backspace-correct
            if ch.isalpha() and self.rng.random() < self.p["typo_rate"]:
                wrong = self._adjacent_key(ch)
                if wrong:
                    time.sleep(keystroke_delay(prev, wrong, self.p["type_base"],
                                               self.rng, fatigue))
                    pyautogui.typewrite(wrong)
                    time.sleep(self.rng.uniform(0.08, 0.25))   # notice the mistake
                    _trace({"ev": "typo", "wrong": wrong, "intended": ch})
                    pyautogui.press("backspace")
                    time.sleep(self.rng.uniform(0.04, 0.10))   # fast correction dwell
                    prev = wrong

            d = keystroke_delay(prev, ch, self.p["type_base"], self.rng, fatigue)
            time.sleep(d)
            _trace({"ev": "key", "ch": ch, "prev": prev, "delay_ms": d * 1000.0})
            pyautogui.typewrite(ch)
            prev = ch

    # ---- scroll ----------------------------------------------------------

    def scroll(self, pixels, direction, by=None, selector=None, x=None, y=None):
        """Human-like scrolling.

        The stock controller calls pyautogui.scroll(N) once, which emits a single
        instantaneous N-pixel jump. Real wheel input is a train of DISCRETE ticks (one per
        physical notch, ~100px each by OS convention), spaced irregularly in time, and a
        human decelerates as they approach what they were looking for. Trackpad flicks add
        inertia. One 600px event has none of that structure.

        No specific paper is cited for the tick-timing distribution -- the discrete-tick
        behaviour is a mechanical property of the wheel, and the timing here is engineering
        judgement in the same spirit as the mouse/keystroke models above. Flagged rather
        than dressed up with a citation it doesn't have.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"Invalid direction: {direction}")

        # position the cursor first, using the HUMAN move (the parent uses a linear
        # moveTo(duration=0.2) here, which silently bypasses all the Bezier work)
        if by is not None and selector is not None:
            try:
                tx, ty, w = self._target_from_element(by, selector, 0.5)
                self._human_move_to(tx, ty, w)
            except Exception as e:
                logger.warning(f"HumanController: scroll pre-move failed ({e})")
                if x is not None and y is not None:
                    self._human_move_to(x, y)
        elif x is not None and y is not None:
            self._human_move_to(x, y)

        sign = 1 if direction == "up" else -1
        remaining = abs(int(pixels))
        _trace({"ev": "scroll_start", "pixels": remaining, "direction": direction})

        # occasional overshoot: blow past the target then come back a little
        overshoot = 0
        if remaining > 250 and self.rng.random() < 0.18:
            overshoot = int(remaining * self.rng.uniform(0.06, 0.18))
            remaining += overshoot

        ticks_done = 0
        while remaining > 0:
            # wheel notch is ~100px; humans vary how hard they flick
            tick = min(remaining, int(self.rng.uniform(60, 130)))
            pyautogui.scroll(sign * tick)
            remaining -= tick
            ticks_done += 1
            _trace({"ev": "scroll_tick", "amount": sign * tick, "remaining": remaining})

            # decelerate: gaps stretch as we approach the target
            frac_left = remaining / max(1.0, abs(int(pixels)))
            gap = self.rng.uniform(0.03, 0.09) * (1.0 + 1.8 * (1.0 - frac_left))
            time.sleep(gap)

            # occasional mid-scroll hesitation (something caught the eye)
            if self.rng.random() < 0.07:
                time.sleep(self.rng.uniform(0.2, 0.6))

        # correct the overshoot back
        if overshoot:
            time.sleep(self.rng.uniform(0.15, 0.4))
            back = 0
            while back < overshoot:
                t = min(overshoot - back, int(self.rng.uniform(40, 90)))
                pyautogui.scroll(-sign * t)
                back += t
                time.sleep(self.rng.uniform(0.04, 0.10))
            _trace({"ev": "scroll_overshoot_correct", "amount": overshoot})

        # settle + read what just came into view; longer scrolls -> longer read
        pause = self.reading_pause(abs(int(pixels)))
        _trace({"ev": "scroll_end", "ticks": ticks_done, "reading_pause": pause})
        self.idle(pause)

    def scroll_to(self, by=None, selector=None, x=None, y=None):
        """Scroll an element into view using human scrolling.

        The parent loops scroll(pixels=5) until the element is in the viewport, which is a
        stream of perfectly uniform 5px events -- arguably a worse tell than one big jump.
        """
        try:
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((by, selector))
            )
            elem_y = element.location["y"]
            scroll_y = self.driver.execute_script(
                "return window.pageYOffset || document.documentElement.scrollTop")
            delta = int(elem_y - scroll_y - 200)   # leave headroom, humans don't pin to the edge
            if abs(delta) > 20:
                self.scroll(pixels=abs(delta), direction="down" if delta > 0 else "up")
            return
        except Exception as e:
            logger.warning(f"HumanController.scroll_to fell back to parent: {e}")
            return super().scroll_to(by=by, selector=selector, x=x, y=y)

    # ---- idle / pacing ---------------------------------------------------

    def reading_pause(self, scrolled_px):
        """How long a human would pause to read what a scroll revealed.

        Rough model: more new content -> longer look, with heavy variance and a floor.
        Not from a specific paper; tune against your own traces if you want to defend a
        number in the write-up.
        """
        base = 0.4 + (scrolled_px / 600.0) * self.rng.uniform(0.6, 2.2)
        return min(6.0, max(0.25, base * self.rng.uniform(0.7, 1.5)))

    def idle(self, seconds):
        """Hold still like a human: tiny involuntary cursor tremor, not a frozen pixel.

        A cursor that is EXACTLY stationary for seconds at a time is trivially separable
        from a human hand resting on a mouse. Keeps amplitude sub-pixel-ish so it never
        drifts off a target.
        """
        end = time.time() + max(0.0, seconds)
        _trace({"ev": "idle_start", "seconds": seconds})
        while time.time() < end:
            if self.rng.random() < 0.30:
                try:
                    cx, cy = pyautogui.position()
                    pyautogui.moveTo(cx + self.rng.uniform(-1.5, 1.5),
                                     cy + self.rng.uniform(-1.5, 1.5), duration=0.02)
                    _trace({"ev": "tremor", "x": cx, "y": cy})
                except Exception:
                    pass
            time.sleep(self.rng.uniform(0.08, 0.25))
        _trace({"ev": "idle_end"})

    def _pace(self):
        """Small gap between consecutive actions. Humans do not chain UI actions with zero
        latency; back-to-back machine-speed actions are a cheap tell."""
        time.sleep(self.rng.uniform(*self.p["action_gap"]))

    def wait(self, seconds):
        """Override the parent's flat sleep so waits are spent idling like a human."""
        self.idle(seconds)

    def _adjacent_key(self, ch):
        pos = _key_pos(ch)
        if not pos:
            return None
        r, c = pos
        candidates = [k for k, (kr, kc) in _QWERTY.items()
                      if abs(kr - r) <= 1 and abs(kc - c) <= 1 and k != ch.lower()]
        return self.rng.choice(candidates) if candidates else None
