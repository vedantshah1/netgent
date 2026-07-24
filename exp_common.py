"""
exp_common.py
-------------
Shared helpers for the experiment suite, so the browser-run + success-check logic lives in
ONE place instead of being copy-pasted (and drifting) across experiments.

The single most important idea in here: `run_and_verify()` returns BOTH
  - structural_valid : did the workflow parse + pass static lint?  (cheap proxy)
  - task_success     : did it actually complete the task in the browser?  (ground truth)
side by side, because those two are different questions and the paper must report both.
A workflow can be structurally perfect and still fail the task.
"""

import re
import time

from netgent_planner import lint_workflow, stopgap_verify


# --------------------------------------------------------------------------- #
#  Success conditions -- grounded in the REAL browser end state               #
# --------------------------------------------------------------------------- #

def check_success(condition, driver):
    """Evaluate a prompt's success condition against the live browser.

    Reads current_url / body text straight off the driver, so the verdict reflects what
    actually happened on the page, not what the workflow intended. Returns (ok, detail).
    Success condition types:
        {"type": "url_contains",  "value": "/secure"}
        {"type": "url_regex",     "value": "wiki/.+"}
        {"type": "text_contains", "value": "Hello World"}
        {"type": "text_absent",   "value": "error"}          # success = text NOT present
        {"type": "element_exists","value": "css=selector"}   # css= or xpath=
    """
    ctype = condition.get("type")
    val = condition.get("value", "")
    try:
        cur_url = driver.current_url or ""
    except Exception:
        cur_url = ""
    try:
        page_text = driver.find_element("tag name", "body").text or ""
    except Exception:
        page_text = ""

    if ctype == "url_contains":
        ok = val.lower() in cur_url.lower()
        return ok, f"url={cur_url!r} contains {val!r}? {ok}"
    if ctype == "url_regex":
        ok = bool(re.search(val, cur_url))
        return ok, f"url={cur_url!r} =~ /{val}/? {ok}"
    if ctype == "text_contains":
        ok = val.lower() in page_text.lower()
        return ok, f"page contains {val!r}? {ok} (url={cur_url!r})"
    if ctype == "text_absent":
        ok = val.lower() not in page_text.lower()
        return ok, f"page does NOT contain {val!r}? {ok} (url={cur_url!r})"
    if ctype == "element_exists":
        by, sel = _split_selector(val)
        try:
            driver.find_element(by, sel)
            return True, f"element {val!r} present"
        except Exception:
            return False, f"element {val!r} NOT found (url={cur_url!r})"
    return False, f"unknown success condition type: {ctype!r}"


def _split_selector(val):
    if val.startswith("css="):
        return "css selector", val[4:]
    if val.startswith("xpath="):
        return "xpath", val[6:]
    return "css selector", val


# --------------------------------------------------------------------------- #
#  One browser run + dual verdict                                             #
# --------------------------------------------------------------------------- #

def run_and_verify(workflow, condition, llm, use_human=True, user_data_dir="/tmp/browser-cache",
                   controller_factory=None):
    """Run one workflow in a real browser and return a dual-verdict dict.

    Returns:
      {
        "structural_valid": bool,      # parsed + 0 lint ERRORs (from lint_workflow)
        "lint_errors": int,
        "task_success": bool,          # the success condition held on the real page
        "success_detail": str,
        "final_url": str,
        "run_result": dict,            # raw NetGent result (or {"error":...})
        "stopgap": {...},              # stopgap_verify report (structural failure class)
        "latency_s": float,
      }

    controller_factory(driver) -> controller lets an experiment inject a specific controller
    (e.g. the stock one vs HumanController) for A/B tests. Default: HumanController if
    use_human else NetGent's stock controller.
    """
    from netgent import NetGent
    from netgent.utils.message import StatePrompt
    from netgent.browser.session import BrowserSession

    # structural verdict first -- doesn't need the browser
    lint = lint_workflow(workflow)
    lint_errors = sum(1 for i in lint if i["level"] == "ERROR")

    prompts = [StatePrompt(**s) for s in workflow]
    driver = BrowserSession(user_data_dir=user_data_dir).driver

    if controller_factory is not None:
        controller = controller_factory(driver)
    elif use_human:
        from human_controller import HumanController
        controller = HumanController(driver)
    else:
        controller = None

    agent = NetGent(driver=driver, controller=controller, llm=llm, llm_enabled=True)
    t0 = time.time()
    run_result, ok, detail, final_url = {}, False, "", ""
    try:
        run_result = agent.run(state_prompts=prompts, state_repository=[])
    except Exception as e:
        run_result = {"error": str(e)}
    finally:
        latency = time.time() - t0
        try:
            ok, detail = check_success(condition, driver)
            final_url = driver.current_url
        except Exception as e:
            detail = f"success-check raised: {e}"
        try:
            agent.controller.quit()
        except Exception:
            pass

    return {
        "structural_valid": lint_errors == 0,
        "lint_errors": lint_errors,
        "lint_issues": lint,
        "task_success": ok,
        "success_detail": detail,
        "final_url": final_url,
        "run_result": run_result,
        "stopgap": stopgap_verify(run_result, workflow),
        "latency_s": latency,
    }
