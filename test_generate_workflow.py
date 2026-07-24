#!/usr/bin/env python3
"""
test_generate_workflow.py
--------------------------
Isolated test harness for Task 2 (generate_workflow) ONLY. No Docker, no browser, no
NetGent, no self-healing loop -- just: sentence -> Gemini -> parsed/validated/linted
workflow. This is the thing your mentor asked you to nail down first.

Runs a battery of prompts, and for each one:
  1. Calls generate_workflow() (parsing + schema validation happens inside it)
  2. Runs the static lint_workflow() checks (empty triggers, missing end_state, etc.)
  3. Prints a PASS/FAIL verdict and saves the generated JSON for manual inspection

A prompt "PASSES" if generation didn't crash AND lint found zero ERRORs. WARN/INFO do
not fail the run but are shown, since they're often real signal (e.g. "only 1 state").

Usage:
    python3 test_generate_workflow.py --api-keys api_keys.json
    python3 test_generate_workflow.py --api-keys api_keys.json --repeat 3   # check consistency
    python3 test_generate_workflow.py --api-keys api_keys.json --prompt "go to X and do Y"
    python3 test_generate_workflow.py --api-keys api_keys.json --prompts-file mine.txt

--repeat N regenerates each prompt N times with the SAME sentence, so you can see how
much the output varies run-to-run (a planner that's wildly inconsistent on identical
input is a real problem worth knowing about before it's buried under self-healing).

Runs fine outside Docker as long as `pip install langchain-google-genai` is available
and you have a Gemini API key -- this is the whole point of isolating Task 2.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netgent_planner import generate_workflow, lint_workflow, format_lint_report
from planner_telemetry import Telemetry, set_telemetry

# A battery covering different site shapes: simple nav, search, forms, e-commerce,
# multi-step. Deliberately AVOIDS Gmail/anything needing real credentials or 2FA --
# see the earlier note about not testing self-healing and workflow-gen on the same
# adversarial target at once.
DEFAULT_PROMPTS = [
    "go to wikipedia, search for bezier curves, and scroll all the way down to the end of the article",
    "go to the-internet.herokuapp.com/login, log in with username 'tomsmith' and password 'SuperSecretPassword!'",
    "go to duckduckgo.com and search for netgent browser automation",
    "go to news.ycombinator.com and click on the first story link",
    "go to the-internet.herokuapp.com/dropdown and select option 2 from the dropdown",
    "go to example.com and confirm the page title is visible",
]


def build_llm(api_keys_path, model="gemini-2.5-flash", temperature=0.2, thinking_budget=None):
    """Build the Gemini chat model.

    thinking_budget: gemini-2.5-flash reasoning control (an int token cap).
        None -> library default (dynamic thinking)
        0    -> attempt to DISABLE thinking (some langchain-google-genai versions ignore 0 and
                still spend reasoning tokens -- issue #928 -- so ALWAYS verify against measured
                reasoning tokens in telemetry, never assume the setting took effect)
        -1   -> explicit dynamic thinking
    """
    with open(api_keys_path) as f:
        keys = json.load(f)
    from langchain_google_genai import ChatGoogleGenerativeAI
    kwargs = dict(model=model, temperature=temperature, api_key=keys["google_api_key"])
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    return ChatGoogleGenerativeAI(**kwargs)


def run_one(sentence, llm, quiet_debug=True):
    """Generate once, lint it, return (workflow_or_None, issues, error_or_None, elapsed)."""
    import netgent_planner as P
    old_debug = P.DEBUG
    P.DEBUG = not quiet_debug  # keep the battery output readable; rerun with --verbose to see raw LLM I/O
    t0 = time.time()
    try:
        workflow = generate_workflow(sentence, llm)
        elapsed = time.time() - t0
        issues = lint_workflow(workflow, sentence=sentence)
        return workflow, issues, None, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return None, [], str(e), elapsed
    finally:
        P.DEBUG = old_debug


def summarize(issues):
    n_err = sum(1 for i in issues if i["level"] == "ERROR")
    n_warn = sum(1 for i in issues if i["level"] == "WARN")
    n_info = sum(1 for i in issues if i["level"] == "INFO")
    return n_err, n_warn, n_info


def main():
    ap = argparse.ArgumentParser(description="Standalone test for generate_workflow (Task 2 only)")
    ap.add_argument("--api-keys", required=True)
    ap.add_argument("--prompt", default=None, help="Test a single sentence instead of the battery")
    ap.add_argument("--prompts-file", default=None, help="One sentence per line")
    ap.add_argument("--repeat", type=int, default=1, help="Regenerate each prompt N times (consistency check)")
    ap.add_argument("--outdir", default="workflow_tests", help="Where to save generated JSON")
    ap.add_argument("--verbose", action="store_true", help="Show raw LLM prompts/output ([PLANNER DEBUG] lines)")
    ap.add_argument("--telemetry-dir", default=None,
                    help="Record full prompts/outputs/tokens/latency as JSONL here "
                         "(default: <outdir>/telemetry). Analyse with analyze_planner_results.py")
    args = ap.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    llm = build_llm(args.api_keys)
    os.makedirs(args.outdir, exist_ok=True)

    # Turn on full telemetry for the battery: every prompt, raw output, token count and
    # latency gets written to JSONL so nothing about this run is lost.
    tel_dir = args.telemetry_dir or os.path.join(args.outdir, "telemetry")
    tel = set_telemetry(Telemetry(outdir=tel_dir, meta={
        "harness": "test_generate_workflow.py",
        "n_prompts": len(prompts),
        "repeat": args.repeat,
        "model": getattr(llm, "model", None) or getattr(llm, "model_name", None),
        "temperature": getattr(llm, "temperature", None),
    }))
    print(f"[telemetry] recording to {tel_dir}/ (llm_calls.jsonl, attempts.jsonl, runs.jsonl)")

    total, passed = 0, 0
    all_state_counts = []

    for p_idx, sentence in enumerate(prompts):
        print("\n" + "=" * 78)
        print(f"PROMPT {p_idx + 1}/{len(prompts)}: {sentence}")
        print("=" * 78)

        for rep in range(args.repeat):
            total += 1
            tag = f"p{p_idx+1}" + (f"_r{rep+1}" if args.repeat > 1 else "")
            workflow, issues, error, elapsed = run_one(sentence, llm, quiet_debug=not args.verbose)

            if error:
                print(f"  [{tag}] EXCEPTION after {elapsed:.1f}s: {error}")
                continue

            n_err, n_warn, n_info = summarize(issues)
            state_names = [s["name"] for s in workflow]
            all_state_counts.append(len(workflow))
            verdict = "PASS" if n_err == 0 else "FAIL"
            if verdict == "PASS":
                passed += 1

            print(f"  [{tag}] {verdict}  ({elapsed:.1f}s, {len(workflow)} states, "
                  f"{n_err} errors / {n_warn} warnings / {n_info} info)")
            print(f"    states: {' -> '.join(state_names)}")
            if issues:
                print(format_lint_report(issues))

            outpath = os.path.join(args.outdir, f"{tag}.json")
            with open(outpath, "w") as f:
                json.dump({"sentence": sentence, "workflow": workflow, "issues": issues}, f, indent=2)

    print("\n" + "=" * 78)
    print(f"SUMMARY: {passed}/{total} passed (zero lint ERRORs)")
    if all_state_counts:
        avg = sum(all_state_counts) / len(all_state_counts)
        print(f"State count: min={min(all_state_counts)} max={max(all_state_counts)} avg={avg:.1f}")
    print(f"Generated workflows saved to {args.outdir}/")

    tel.close(summary={"total": total, "passed": passed,
                       "state_counts": all_state_counts})
    print(f"Telemetry written to {tel_dir}/")
    print(f"\nNext: python3 analyze_planner_results.py {tel_dir} "
          f"--csv {args.outdir}/results.csv --figs {args.outdir}/figs")
    print("=" * 78)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
