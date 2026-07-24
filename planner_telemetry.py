"""
planner_telemetry.py
--------------------
Structured data collection for the NetGent planner, so every run produces a permanent,
analysable record instead of scrollback you lose when the container exits.

Two record types, written as JSONL (one JSON object per line -- append-safe, streamable,
and trivially loadable into pandas):

  llm_calls.jsonl     one record per LLM invocation (prompt, raw output, tokens, latency)
  attempts.jsonl      one record per generate_workflow() call (workflow, lint, verdict)

Why JSONL and not one big JSON: a crashed/killed run still leaves every completed record
intact and parseable, which matters because several of your runs have died mid-way.

Usage (normally you don't call this directly -- netgent_planner does):

    from planner_telemetry import Telemetry
    tel = Telemetry(outdir="/out/telemetry", run_id="gmail_test_3")
    tel.log_llm_call(...)
    tel.log_attempt(...)
    tel.close()

Enable from the environment (what run_planner.py / the test harness do):
    export PLANNER_TELEMETRY_DIR=/out/telemetry
"""

import json
import os
import time
import uuid
import platform
import datetime


# --------------------------------------------------------------------------- #
#  Token / metadata extraction                                                #
# --------------------------------------------------------------------------- #

def extract_usage(resp):
    """Pull token counts out of a LangChain response, defensively.

    LangChain standardises on `.usage_metadata` = {input_tokens, output_tokens,
    total_tokens} (verified against langchain_core.messages.ai.UsageMetadata). Older
    versions / some providers only populate `.response_metadata`, and Gemini nests its
    own counts under different key names, so we try several shapes and return None
    rather than guessing if nothing matches.

    Returns dict with input_tokens / output_tokens / total_tokens (values may be None).
    """
    out = {"input_tokens": None, "output_tokens": None, "total_tokens": None,
           "reasoning_tokens": None}

    um = getattr(resp, "usage_metadata", None)
    if isinstance(um, dict) and um:
        out["input_tokens"] = um.get("input_tokens")
        out["output_tokens"] = um.get("output_tokens")
        out["total_tokens"] = um.get("total_tokens")
        # Gemini 2.5 thinking tokens live here; this is how you MEASURE whether
        # thinking_budget=0 actually took effect (issue #928: sometimes it doesn't).
        otd = um.get("output_token_details") or {}
        if isinstance(otd, dict):
            out["reasoning_tokens"] = otd.get("reasoning") or otd.get("reasoning_tokens")
        if out["total_tokens"] is None and None not in (out["input_tokens"], out["output_tokens"]):
            out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        if any(v is not None for v in out.values()):
            return out

    # Fallback: provider-specific nesting inside response_metadata
    rm = getattr(resp, "response_metadata", None) or {}
    if isinstance(rm, dict):
        nested = rm.get("usage_metadata") or rm.get("token_usage") or rm.get("usage") or {}
        if isinstance(nested, dict) and nested:
            out["input_tokens"] = (nested.get("prompt_token_count")
                                   or nested.get("input_tokens")
                                   or nested.get("prompt_tokens"))
            out["output_tokens"] = (nested.get("candidates_token_count")
                                    or nested.get("output_tokens")
                                    or nested.get("completion_tokens"))
            out["total_tokens"] = (nested.get("total_token_count")
                                   or nested.get("total_tokens"))
            if out["total_tokens"] is None and None not in (out["input_tokens"], out["output_tokens"]):
                out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


def extract_finish_reason(resp):
    """Best-effort finish reason ('STOP', 'MAX_TOKENS', 'SAFETY', ...).

    Worth capturing: a truncated response (MAX_TOKENS) and a safety block look identical
    downstream -- both just produce unparseable JSON -- but need completely different fixes.
    """
    rm = getattr(resp, "response_metadata", None) or {}
    if isinstance(rm, dict):
        for k in ("finish_reason", "finishReason", "stop_reason"):
            if rm.get(k):
                return str(rm[k])
        cands = rm.get("candidates")
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            fr = cands[0].get("finish_reason") or cands[0].get("finishReason")
            if fr:
                return str(fr)
    return None


def extract_safety(resp):
    """Gemini safety ratings, if present. A SAFETY finish is a real failure mode when
    prompts contain credentials or scraping-adjacent language."""
    rm = getattr(resp, "response_metadata", None) or {}
    if isinstance(rm, dict):
        for k in ("safety_ratings", "safetyRatings"):
            if rm.get(k):
                return rm[k]
    return None


# --------------------------------------------------------------------------- #
#  Recorder                                                                   #
# --------------------------------------------------------------------------- #

class Telemetry:
    """Append-only JSONL recorder. Cheap enough to leave on permanently."""

    def __init__(self, outdir=None, run_id=None, meta=None):
        self.outdir = outdir or os.environ.get("PLANNER_TELEMETRY_DIR")
        self.enabled = bool(self.outdir)
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._calls = []
        self._attempts = []
        if not self.enabled:
            return
        os.makedirs(self.outdir, exist_ok=True)
        self.calls_path = os.path.join(self.outdir, "llm_calls.jsonl")
        self.attempts_path = os.path.join(self.outdir, "attempts.jsonl")
        self.run_path = os.path.join(self.outdir, "runs.jsonl")
        self._write(self.run_path, {
            "type": "run_start",
            "run_id": self.run_id,
            "ts": time.time(),
            "iso": datetime.datetime.now().isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "meta": meta or {},
        })

    def _write(self, path, record):
        if not self.enabled:
            return
        try:
            with open(path, "a", buffering=1) as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            print(f"[TELEMETRY] write failed ({e})", flush=True)

    # ---- per-LLM-call ----------------------------------------------------

    def log_llm_call(self, stage, system_prompt, user_prompt, raw_output,
                     latency_s, resp=None, model=None, temperature=None, error=None):
        usage = extract_usage(resp) if resp is not None else {
            "input_tokens": None, "output_tokens": None, "total_tokens": None}
        rec = {
            "type": "llm_call",
            "run_id": self.run_id,
            "call_id": uuid.uuid4().hex[:10],
            "ts": time.time(),
            "iso": datetime.datetime.now().isoformat(),
            "stage": stage,
            "model": model,
            "temperature": temperature,
            "latency_s": round(latency_s, 4),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "system_prompt_chars": len(system_prompt or ""),
            "user_prompt_chars": len(user_prompt or ""),
            "raw_output": raw_output,
            "raw_output_chars": len(raw_output or ""),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "finish_reason": extract_finish_reason(resp) if resp is not None else None,
            "safety_ratings": extract_safety(resp) if resp is not None else None,
            "error": error,
        }
        self._calls.append(rec)
        if self.enabled:
            self._write(self.calls_path, rec)
        return rec

    # ---- per-generation-attempt ------------------------------------------

    def log_attempt(self, sentence, stage, workflow, issues, latency_s,
                    parse_ok, needed_fixup, parse_error=None, n_llm_calls=1,
                    tokens=None, extra=None):
        n_err = sum(1 for i in (issues or []) if i.get("level") == "ERROR")
        n_warn = sum(1 for i in (issues or []) if i.get("level") == "WARN")
        n_info = sum(1 for i in (issues or []) if i.get("level") == "INFO")
        wf = workflow or []
        rec = {
            "type": "attempt",
            "run_id": self.run_id,
            "attempt_id": uuid.uuid4().hex[:10],
            "ts": time.time(),
            "iso": datetime.datetime.now().isoformat(),
            "stage": stage,
            "sentence": sentence,
            "sentence_chars": len(sentence or ""),
            "latency_s": round(latency_s, 4),
            "n_llm_calls": n_llm_calls,
            "parse_ok": parse_ok,
            "needed_json_fixup": needed_fixup,
            "parse_error": parse_error,
            "verdict": "PASS" if (parse_ok and n_err == 0) else "FAIL",
            "workflow": wf,
            "state_count": len(wf),
            "state_names": [s.get("name") for s in wf],
            "total_triggers": sum(len(s.get("triggers") or []) for s in wf),
            "total_actions": sum(len(s.get("actions") or []) for s in wf),
            "states_with_no_triggers": [s.get("name") for s in wf if not (s.get("triggers") or [])],
            "states_with_no_actions": [s.get("name") for s in wf if not (s.get("actions") or [])],
            "lint_issues": issues or [],
            "lint_errors": n_err,
            "lint_warnings": n_warn,
            "lint_infos": n_info,
            "lint_codes": [_issue_code(i) for i in (issues or [])],
            "input_tokens": (tokens or {}).get("input_tokens"),
            "output_tokens": (tokens or {}).get("output_tokens"),
            "total_tokens": (tokens or {}).get("total_tokens"),
        }
        if extra:
            rec.update(extra)
        self._attempts.append(rec)
        if self.enabled:
            self._write(self.attempts_path, rec)
        return rec

    def close(self, summary=None):
        if not self.enabled:
            return
        self._write(self.run_path, {
            "type": "run_end",
            "run_id": self.run_id,
            "ts": time.time(),
            "iso": datetime.datetime.now().isoformat(),
            "n_llm_calls": len(self._calls),
            "n_attempts": len(self._attempts),
            "summary": summary or {},
        })


def _issue_code(issue):
    """Short stable code for a lint issue, so we can COUNT error types across runs
    instead of trying to group by full prose message."""
    msg = (issue.get("message") or "").lower()
    if "empty triggers" in msg:
        return "EMPTY_TRIGGERS"
    if "empty actions" in msg:
        return "EMPTY_ACTIONS"
    if "no 'end_state'" in msg or "no end_state" in msg:
        return "MISSING_END_STATE"
    if "used more than once" in msg:
        return "DUPLICATE_STATE_NAME"
    if "chrome://new-tab-page" in msg:
        return "NO_FRESH_TAB_TRIGGER"
    if "terminate" in msg:
        return "NO_TERMINATE_ACTION"
    if "only one state" in msg:
        return "SINGLE_STATE"
    if "is a lot for one task" in msg:
        return "TOO_MANY_STATES"
    if "unusually many" in msg:
        return "TOO_MANY_TRIGGERS"
    return "OTHER"


# Module-level default recorder, created lazily from the environment.
_default = None


def get_telemetry():
    global _default
    if _default is None:
        _default = Telemetry()
    return _default


def set_telemetry(tel):
    global _default
    _default = tel
    return _default
