#!/usr/bin/env python3
"""
condense_log.py
================
Strips embedded base64 blobs (screenshots) and other oversized fields out of raw .log files
(e.g. exp2_full_run.log, run.log from tee'd docker output) so they're actually readable and
shareable, while keeping every line of real debug output, prints, tracebacks, and results.

WHY THE .log FILES ARE HUGE: NetGent's own debug output embeds full-page base64 PNG
screenshots and full DOM element dumps inline in its metadata prints. A single state-transition
line can be tens of thousands of characters. This is separate from telemetry (llm_calls.jsonl)
-- it's raw stdout captured via `tee`.

RETROACTIVE -- works on logs you already have, no rerun needed. STREAMS the file line by line,
so it's safe on multi-GB logs.

USAGE:
    python3 condense_log.py exp2_full_run.log
    python3 condense_log.py exp2_full_run.log -o exp2_full_run.condensed.log
    python3 condense_log.py exp2_full_run.log --max-line 300           # more aggressive
    python3 condense_log.py results --recursive                        # every .log under a dir
    python3 condense_log.py exp2_full_run.log --headline               # keep ONLY the important lines
"""

import argparse
import os
import re
import sys

# base64 image/binary blobs: long runs of base64 alphabet, optionally padded with =
BASE64_RE = re.compile(r'[A-Za-z0-9+/]{80,}={0,2}')

# lines that actually matter for debugging the planner/self-healing loop
HEADLINE_RE = re.compile(
    r'PLANNER DEBUG|GENERATE|REPAIR|\[heal\]|\[run\]|\[lint\]|verify:stopgap|'
    r'task_success|Traceback|Error|Exception|SUMMARY|PASS|FAIL|'
    r'structural_valid|final_url|state_count|attempt',
    re.IGNORECASE)


def strip_base64(line):
    def _repl(m):
        s = m.group(0)
        return f"[...{len(s)} char base64/binary blob omitted...]"
    return BASE64_RE.sub(_repl, line)


def cap_line(line, max_len):
    if len(line) <= max_len:
        return line
    head = line[: max_len // 2]
    tail = line[-max_len // 2:]
    omitted = len(line) - max_len
    return f"{head}...[{omitted} more chars omitted]...{tail}"


def condense_file(path, outpath, max_line, headline_only):
    orig_bytes = os.path.getsize(path)
    n_lines = 0
    n_kept = 0
    n_shrunk = 0
    with open(path, "r", errors="replace") as fin, open(outpath, "w") as fout:
        for line in fin:
            n_lines += 1
            stripped_raw = line.rstrip("\n")
            if headline_only and not HEADLINE_RE.search(stripped_raw):
                continue
            new_line = strip_base64(stripped_raw)
            new_line = cap_line(new_line, max_line)
            if new_line != stripped_raw:
                n_shrunk += 1
            fout.write(new_line + "\n")
            n_kept += 1
    new_bytes = os.path.getsize(outpath)
    return orig_bytes, new_bytes, n_lines, n_kept, n_shrunk


def find_logs(root):
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".log"):
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="A .log file, OR a directory (use --recursive) to sweep")
    ap.add_argument("-o", "--output", default=None, help="Output path (single-file mode only)")
    ap.add_argument("--max-line", type=int, default=500,
                     help="Max chars per line after base64 stripping (default 500)")
    ap.add_argument("--recursive", action="store_true",
                     help="Treat 'path' as a directory and condense every .log under it")
    ap.add_argument("--headline", action="store_true",
                     help="Keep ONLY lines matching known important markers (most aggressive)")
    args = ap.parse_args()

    if args.recursive:
        logs = find_logs(args.path)
        if not logs:
            print(f"No .log files found under {args.path}")
            sys.exit(1)
        print(f"Found {len(logs)} log file(s) under {args.path}\n")
        total_orig = total_new = 0
        for p in logs:
            outp = p[:-4] + (".headline.log" if args.headline else ".condensed.log")
            ob, nb, nl, nk, ns = condense_file(p, outp, args.max_line, args.headline)
            total_orig += ob
            total_new += nb
            print(f"  {p}")
            print(f"    {ob/1024:.0f} KB -> {nb/1024:.0f} KB "
                  f"({100*(1-nb/ob):.0f}% smaller)" if ob else "    empty")
            print(f"    {nl} lines -> {nk} kept, {ns} shrunk")
            print(f"    wrote {outp}")
        print("\n" + "=" * 60)
        if total_orig:
            print(f"TOTAL: {total_orig/1024:.0f} KB -> {total_new/1024:.0f} KB "
                  f"({100*(1-total_new/total_orig):.0f}% smaller)")
    else:
        default_suffix = ".headline.log" if args.headline else ".condensed.log"
        outp = args.output or (args.path[:-4] + default_suffix if args.path.endswith(".log")
                                else args.path + default_suffix)
        ob, nb, nl, nk, ns = condense_file(args.path, outp, args.max_line, args.headline)
        print(f"{args.path}")
        if ob:
            print(f"  {ob/1024:.0f} KB -> {nb/1024:.0f} KB ({100*(1-nb/ob):.0f}% smaller)")
        else:
            print("  empty file")
        print(f"  {nl} lines total -> {nk} kept, {ns} shrunk")
        print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
