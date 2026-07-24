# Experiment 2 — End-to-end task accuracy

5 prompts, max_repairs=2, human=yes


## Task: go to saucedemo.com, log in with username standard_user and password secret_sauce, add the Sauce Labs Backpack and the Sauce Labs Bike Light to the cart, open the shopping cart, click checkout, enter first name Test, last name User, postal code 12345, continue, and then finish the order

*Expected difficulty:* Canonical 8-phase e-commerce funnel: fresh tab -> login -> inventory (two adds) -> cart -> checkout step one -> form fill -> overview -> complete. Every phase gates the next, and the success marker only appears on the final page, so partial completion cannot be mistaken for success.

**1. Generate:** 7 states in 4.7s — On Browser Home Page -> On Login Page -> On Products Page -> On Cart Page -> On Checkout Step One Page -> On Checkout Step Two Page -> On Checkout Complete Page

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — page text contains 'Thank you for your order'? False  (url='https://www.saucedemo.com/')

*Self-heal (EMPTY_TRIGGERS): regenerated to 7 states.*

**3-4. Run attempt 2:** task_success=**False** — page text contains 'Thank you for your order'? False  (url='https://www.saucedemo.com/inventory.html')

*Self-heal (TIMEOUT): regenerated to 7 states.*

**3-4. Run attempt 3:** task_success=**False** — page text contains 'Thank you for your order'? False  (url='https://www.saucedemo.com/inventory.html')

Failed after 3 attempt(s); self-heal budget exhausted.


## Task: go to saucedemo.com, log in with username standard_user and password secret_sauce, sort the products by price from low to high, add the two cheapest products to the cart, open the cart, remove one of them, and confirm exactly one item remains

*Expected difficulty:* Adds a SORT (changes DOM order before selection) and a REMOVE (undoes prior state). Tests whether the planner can reason about post-conditions rather than a fixed click sequence. WARNING: the success condition here is weak -- 'text_contains 1' will match almost anything. Before running, replace it with a check on the cart badge specifically, or use url_contains '/cart.html' plus a manual read of the report. Flagged rather than silently shipped.

**1. Generate:** 7 states in 7.4s — On Browser Home Page -> On Login Page -> On Products Page - Sort -> On Products Page - Add to Cart -> On Products Page - Open Cart -> On Cart Page - Remove Item -> On Cart Page - Confirm One Item

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — page text contains '1'? False  (url='https://www.saucedemo.com/')

*Self-heal (EMPTY_TRIGGERS): regenerated to 7 states.*

**3-4. Run attempt 2:** task_success=**False** — page text contains '1'? False  (url='https://www.saucedemo.com/')

*Self-heal (EMPTY_TRIGGERS): regenerated to 7 states.*

**3-4. Run attempt 3:** task_success=**True** — page text contains '1'? True  (url='https://www.saucedemo.com/inventory.html')

Task VERIFIED SUCCESSFUL.


## Task: go to the-internet.herokuapp.com/login, log in with username tomsmith and password SuperSecretPassword!, click Logout, then go to the-internet.herokuapp.com/dynamic_controls, click the Enable button, wait for the input field to become enabled, and type hello into it

*Expected difficulty:* Three dependent phases on a controlled fixture: auth in, auth out, then a fresh async interaction on a different page. Requires a genuine wait/observe state between Enable and typing. Crisp single success marker that only appears after the async enable completes.

**1. Generate:** 8 states in 9.0s — Start and Navigate to Login -> On Login Page -> On Secure Area After Login -> After Logout, Navigate to Dynamic Controls -> On Dynamic Controls Page, Click Enable -> Waiting for Input Field to Enable -> Input Field Enabled, Type 'hello' -> Task Complete

**2. Lint:** 0 errors / 1 issues

```
  [LINT][INFO] (Waiting for Input Field to Enable) Empty actions with a wait/verification-sounding name -- looks like an intentional 'sit here until something external changes' state. Confirm that's what you meant.
```

**3-4. Run attempt 1:** task_success=**False** — page text contains "It's enabled"? False  (url='https://the-internet.herokuapp.com/login')

*Self-heal (EMPTY_TRIGGERS): regenerated to 8 states.*

**3-4. Run attempt 2:** task_success=**False** — page text contains "It's enabled"? False  (url='https://the-internet.herokuapp.com/login')

*Self-heal (EMPTY_TRIGGERS): regenerated to 8 states.*

**3-4. Run attempt 3:** task_success=**False** — page text contains "It's enabled"? False  (url='https://the-internet.herokuapp.com/login')

Failed after 3 attempt(s); self-heal budget exhausted.


## Task: go to automationexercise.com, search for 'dress', add the first two products in the results to your cart, view the cart, and confirm both products are listed

*Expected difficulty:* Live third-party site, so it doubles as the bot-detection stressor. Two sequential adds each raise a modal that must be dismissed before the next add -- that modal cycle is what pushes this past 6 states. NOTE: the 5-state version of this task failed 3/3 in the medium battery, always stuck on the homepage with EMPTY_TRIGGERS. If it fails again here, check for a cookie-consent or ad overlay before blaming the planner.

**1. Generate:** 4 states in 9.8s — On Browser Home Page -> On Automation Exercise Homepage -> On Search Results Page -> On Cart Page

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — final_url='https://automationexercise.com/' contains '/view_cart'? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 5 states.*

**3-4. Run attempt 2:** task_success=**False** — final_url='https://automationexercise.com/' contains '/view_cart'? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 5 states.*

**3-4. Run attempt 3:** task_success=**False** — success-check raised: HTTPConnectionPool(host='localhost', port=58641): Max retries exceeded with url: /session/54c251e4786700c528dc6c7f94284344/url (Caused by ReadTimeoutError("HTTPConnectionPool(host='localhost', port=58641): Read timed out. (read timeout=120)"))

Failed after 3 attempt(s); self-heal budget exhausted.


## Task: go to saucedemo.com, log in with username standard_user and password secret_sauce, open the Sauce Labs Fleece Jacket product page, add it to the cart from that page, go back to the products list, open the shopping cart, click checkout, and fill in first name Test, last name User, postal code 12345

*Expected difficulty:* Includes a drill-down into a product detail page and a BACK navigation, which the medium battery never exercised. Back-navigation is a distinct trigger problem: the URL returns to a previously-visited state, so a naive state machine can loop. Good stress test for trigger disambiguation.

**1. Generate:** 9 states in 9.5s — On Browser Home Page -> On Login Page -> On Products Page -> On Fleece Jacket Product Page -> On Fleece Jacket Product Page (After Add to Cart) -> On Products Page (After Returning) -> On Shopping Cart Page -> On Checkout Your Information Page -> On Checkout Overview Page

**2. Lint:** clean

**3-4. Run attempt 1:** task_success=**False** — final_url='https://www.saucedemo.com/' contains 'checkout-step-two'? False

*Self-heal (EMPTY_TRIGGERS): regenerated to 9 states.*

**3-4. Run attempt 2:** task_success=**False** — final_url='https://www.saucedemo.com/inventory.html' contains 'checkout-step-two'? False

*Self-heal (UNKNOWN): regenerated to 9 states.*

**3-4. Run attempt 3:** task_success=**False** — final_url='https://www.saucedemo.com/inventory.html' contains 'checkout-step-two'? False

Failed after 3 attempt(s); self-heal budget exhausted.


---
# SUMMARY
- Task success (final): **1/5**
- Success on first try (no self-heal): 0/5
- Recovered by self-healing: 1/5
- Structural validity (lint clean at generation): 5/5


**Note:** self-heal here is driven by the STOPGAP verifier, not Oliver's real one — label accordingly in the paper.
