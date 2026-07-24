#!/usr/bin/env python3
"""
patch_scroll_fix.py -- fix HumanController clicking the wrong element on any page where the
target isn't near the top.

ROOT CAUSE: _target_from_element() reads element.location["x"/"y"], which Selenium returns
relative to the DOCUMENT origin, and feeds it straight into pyautogui, which operates in
absolute SCREEN pixels. There is no scrollIntoView call and no scroll-offset compensation
anywhere in the function. The two coordinate systems only happen to agree when the page hasn't
scrolled and the target sits near the top -- true for every saucedemo checkout button, false for
the-internet.herokuapp.com/dynamic_controls's "Enable" button, which sits below the "Remove"
section. Without scroll compensation, the click undershoots vertically and lands on whatever IS
near the top of the (scrolled or unscrolled) viewport -- in this case, Remove.

FIX: use Selenium's `location_once_scrolled_into_view`, which scrolls the element into view as a
side effect and returns its location in that resulting, now-consistent state -- the standard
idiom for this exact problem, rather than hand-computing scroll offsets.

USAGE: python3 patch_scroll_fix.py [--dir .] [--dry-run] [--revert]
"""
import argparse
import os
import sys

TARGET = "human_controller.py"

OLD = '''        w, h = element.size["width"], element.size["height"]
        # off-center: humans rarely hit the exact middle
        px = min(0.85, max(0.15, percentage + self.rng.uniform(-0.22, 0.22)))
        py = min(0.85, max(0.15, 0.5 + self.rng.uniform(-0.22, 0.22)))
        cx, cy = self.get_element_coordinates(element.location["x"], element.location["y"],
                                              w, h, px)'''

NEW = '''        # Scroll the element into view FIRST. element.location is relative to the DOCUMENT,
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
        cx, cy = self.get_element_coordinates(loc["x"], loc["y"], w, h, px)'''

SENTINEL = "location_once_scrolled_into_view"


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
    is_applied = SENTINEL in src
    want_applied = not args.revert

    if is_applied == want_applied:
        print(f"[SKIP] already {'applied' if is_applied else 'reverted'}")
        return 0

    old, new = (NEW, OLD) if args.revert else (OLD, NEW)
    if old not in src:
        print("[FAIL] anchor not found -- file may have drifted; patch by hand")
        return 1
    if src.count(old) > 1:
        print(f"[FAIL] anchor ambiguous ({src.count(old)}x)")
        return 1

    src = src.replace(old, new, 1)
    if not args.dry_run:
        open(path, "w", encoding="utf-8").write(src)
    print(f"[ OK ] {'reverted' if args.revert else 'applied'}"
          f"{'  (dry run -- nothing written)' if args.dry_run else ''}")

    if not args.revert:
        print("""
VERIFY: re-run task 2's Enable-button state. It should now click Enable, not Remove.

IMPORTANT CAVEAT: I have not been able to run this against a real browser myself -- I don't
have one in this environment. This is a well-reasoned, code-grounded fix (confirmed the bug
mechanism structurally and the page layout matches the symptom exactly), but please treat the
FIRST re-run as a test of the fix, not an assumed success. If Enable still isn't clicked
correctly, report back what happens -- that would mean the bug is elsewhere (e.g. inside the
upstream get_element_coordinates() itself, which isn't in our project files to inspect).

Since this changes click behavior for EVERY task, not just task 2, it's worth a quick re-check
that task 1 (saucedemo) still passes after this patch, even though its buttons were unaffected
by the bug -- scrolling into view should be a no-op for elements already in view, but confirm.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
