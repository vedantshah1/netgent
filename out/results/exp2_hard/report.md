# Experiment 2 — End-to-end task accuracy

5 prompts, max_repairs=2, human=yes


## Task: go to the-internet.herokuapp.com/login, log in with username tomsmith and password SuperSecretPassword!, then click Logout and confirm you are returned to the login page

*Expected difficulty:* extends the already-working login flow with a second phase (logout) on a NEW page -- tests whether the planner can chain two dependent multi-page transitions instead of stopping at the first goal.

**1. Generate:** 4 states in 3.8s — On Browser Home Page -> On Login Page -> On Secure Area Page -> Returned to Login Page After Logout

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — page text contains 'You logged out of the secure area'? False  (url='https://the-internet.herokuapp.com/login')

*Self-heal (EMPTY_TRIGGERS): regenerated to 4 states.*

**3-4. Run attempt 2:** task_success=**True** — page text contains 'You logged out of the secure area'? True  (url='https://the-internet.herokuapp.com/login')

Task VERIFIED SUCCESSFUL.


## Task: go to the-internet.herokuapp.com/dynamic_controls, click the Remove button to remove the checkbox, and wait for confirmation that it is gone

*Expected difficulty:* AJAX-delayed DOM mutation (~5s) with no URL change -- needs a genuine wait/observe state (empty-actions pattern) keyed on a text change, not a navigation.

**1. Generate:** 3 states in 10.9s — On Browser Home Page -> On Dynamic Controls Page -> Checkbox Removed Confirmation

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**True** — page text contains "It's gone"? True  (url='https://the-internet.herokuapp.com/dynamic_controls')

Task VERIFIED SUCCESSFUL.


## Task: go to the-internet.herokuapp.com/add_remove_elements/, click 'Add Element', and confirm a Delete button appears

*Expected difficulty:* success is a DOM-only change (no URL change at all) -- the trigger has to be keyed on a visible element appearing, which is a harder recognition problem than a URL substring.

**1. Generate:** 3 states in 3.5s — On Browser Home Page -> On Add/Remove Elements Page -> Delete Button Appeared

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — page text contains 'Delete'? False  (url='https://the-internet.herokuapp.com/add_remove_elements/')

*Self-heal (EMPTY_TRIGGERS): regenerated to 3 states.*

**3-4. Run attempt 2:** task_success=**True** — page text contains 'Delete'? True  (url='https://the-internet.herokuapp.com/add_remove_elements/')

Task VERIFIED SUCCESSFUL.


## Task: go to the-internet.herokuapp.com/hovers, hover over the first user's image to reveal their profile link, then click it and confirm you land on their profile page

*Expected difficulty:* requires a hover-to-reveal interaction before the click target even exists -- hover isn't demonstrated in the planner's few-shot example, so this tests whether the action vocabulary generalizes to an unseen interaction type.

**1. Generate:** 3 states in 6.8s — On Browser Home Page -> On Hovers Page -> On Profile Page

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — final_url='https://the-internet.herokuapp.com/hovers' matches //users/\d/? False

*Self-heal (TIMEOUT): regenerated to 3 states.*

**3-4. Run attempt 2:** task_success=**False** — final_url='https://the-internet.herokuapp.com/hovers' matches //users/\d/? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 3 states.*

**3-4. Run attempt 3:** task_success=**False** — final_url='https://the-internet.herokuapp.com/hovers' matches //users/\d/? False

Failed after 3 attempt(s); self-heal budget exhausted.


## Task: go to automationexercise.com, search for 'dress', add the first result to your cart, then view the cart

*Expected difficulty:* closest real analog to the Amazon search-and-cart idea -- multi-page (home -> search results -> cart popup -> cart page), a real third-party site rather than a controlled fixture, hardest in the set.

**1. Generate:** 5 states in 9.4s — On Browser Home Page -> On Automation Exercise Homepage -> On Search Results Page -> Product Added Confirmation -> On Cart Page

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — final_url='https://automationexercise.com/' contains '/view_cart'? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 5 states.*

**3-4. Run attempt 2:** task_success=**False** — final_url='https://automationexercise.com/' contains '/view_cart'? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 5 states.*

**3-4. Run attempt 3:** task_success=**False** — final_url='https://automationexercise.com/' contains '/view_cart'? False

Failed after 3 attempt(s); self-heal budget exhausted.


---
# SUMMARY
- Task success (final): **3/5**
- Success on first try (no self-heal): 1/5
- Recovered by self-healing: 2/5
- Structural validity (lint clean at generation): 5/5


**Note:** self-heal here is driven by the STOPGAP verifier, not Oliver's real one — label accordingly in the paper.
