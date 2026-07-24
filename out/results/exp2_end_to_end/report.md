# Experiment 2 — End-to-end task accuracy

5 prompts, max_repairs=1, human=yes


## Task: go to example.com

*Expected difficulty:* trivial single navigation -- sanity check the whole pipeline works.

**1. Generate:** 2 states in 2.4s — On Browser Home Page -> On Example.com

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**True** — final_url='https://example.com/' contains 'example.com'? True

Task VERIFIED SUCCESSFUL.


## Task: go to en.wikipedia.org, search for Bezier curve, and open the article

*Expected difficulty:* search + result navigation; the wikipedia consistency fork lives here.

**1. Generate:** 4 states in 11.8s — On Browser Home Page -> On Wikipedia Homepage -> On Search Results Or Article Page -> On Bezier Curve Article Page

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — final_url='https://en.wikipedia.org/wiki/Main_Page' contains '/wiki/B'? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 4 states.*

**3-4. Run attempt 2:** task_success=**True** — final_url='https://en.wikipedia.org/wiki/B%C3%A9zier_curve' contains '/wiki/B'? True

Task VERIFIED SUCCESSFUL.


## Task: go to the-internet.herokuapp.com/login and log in with username tomsmith and password SuperSecretPassword!

*Expected difficulty:* a real login flow with a stable success page (/secure area).

**1. Generate:** 3 states in 4.4s — On Browser Home Page -> On Login Page -> On Secure Area Page

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — final_url='https://the-internet.herokuapp.com/login' contains '/secure'? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 3 states.*

**3-4. Run attempt 2:** task_success=**False** — final_url='https://the-internet.herokuapp.com/login' contains '/secure'? False

Failed after 2 attempt(s); self-heal budget exhausted.


## Task: go to the-internet.herokuapp.com/dynamic_loading/2, click Start, and wait for the hidden text to appear

*Expected difficulty:* async wait -- needs a wait/observe state, tests the empty-actions pattern.

**1. Generate:** 4 states in 5.9s — On Browser Home Page -> On Dynamic Loading Page -> Waiting for Hidden Text -> Hidden Text Appeared

**2. Lint:** 0 errors / 1 issues

```
  [LINT][INFO] (Waiting for Hidden Text) Empty actions with a wait/verification-sounding name -- looks like an intentional 'sit here until something external changes' state. Confirm that's what you meant.
```

**3-4. Run attempt 1:** task_success=**True** — page text contains 'Hello World'? True  (url='https://the-internet.herokuapp.com/dynamic_loading/2')

Task VERIFIED SUCCESSFUL.


## Task: go to the-internet.herokuapp.com/checkboxes and make sure both checkboxes are checked

*Expected difficulty:* post-condition task (both checked) -- hardest; structural consistency was low here.

**1. Generate:** 3 states in 3.6s — On Browser Home Page -> On Checkboxes Page - Ensure Checked -> Both Checkboxes Confirmed Checked

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**True** — page text contains 'checkboxes'? True  (url='https://the-internet.herokuapp.com/checkboxes')

Task VERIFIED SUCCESSFUL.


---
# SUMMARY
- Task success (final): **4/5**
- Success on first try (no self-heal): 3/5
- Recovered by self-healing: 1/5
- Structural validity (lint clean at generation): 5/5


**Note:** self-heal here is driven by the STOPGAP verifier, not Oliver's real one — label accordingly in the paper.
