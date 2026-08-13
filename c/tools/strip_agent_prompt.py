"""Build a minimal-schema tool catalog + compact system prompt, and size it.

The orchestrator sends ~16k tokens of boilerplate on every turn, which at this
engine's prefill rate is ~28 minutes to first token. Almost none of it is the
user's actual question. This strips the catalog to what the model demonstrably
needs to emit a *correct* call -- names, parameter names, types, required, and
enums -- and drops the prose that costs tokens without changing behaviour.

Parameter names are kept deliberately: with no schema at all the model invented
`paths: [...]` where the real tool takes `path: "..."`, so dropping schemas
trades a long wait for silently-wrong arguments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from tokenizers import Tokenizer  # noqa: E402

MODEL = Path("/home/dinga/Projects/colib/c/ornith35")
TOK = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
SPEC = json.load(open("/tmp/lf_prompt.json"))

# The tools an orchestrator actually needs to localize, read, edit and finish.
CORE = ["set_phase", "knowledgebase", "glob_files", "grep_files", "list_dir", "read_file",
        "write_file", "search_replace", "run_command", "git_status", "git_diff",
        "task_create", "task_update", "task_list", "finish"]

COMPACT_SYSTEM = """You are a software engineering agent working in {cwd}.

Work in phases: call set_phase to move between triage, localize, edit, verify.
Localize with grep_files/glob_files/list_dir and read_file before editing.
Make edits with search_replace or write_file. Verify with run_command.
Call finish when the task is complete, with a short summary of what you did.

Rules:
- Read a file before editing it. Never invent file paths or line numbers.
- Prefer the smallest edit that solves the problem.
- Do not modify files unless the task asks you to.
- One tool call at a time. Wait for the result before deciding the next step.
- If the task is a question, answer it directly and call finish; do not edit.
"""


def first_sentence(text: str, limit: int = 90) -> str:
    text = " ".join((text or "").split())
    cut = text.find(". ")
    if 0 < cut < limit:
        return text[:cut]
    return text[:limit].rstrip()


def minify_params(params: dict) -> dict:
    props = {}
    for name, spec in (params.get("properties") or {}).items():
        keep = {}
        if spec.get("type"):
            keep["type"] = spec["type"]
        if spec.get("enum"):                       # semantically load-bearing, keep
            keep["enum"] = spec["enum"]
        if spec.get("items", {}).get("type"):
            keep["items"] = {"type": spec["items"]["type"]}
        props[name] = keep or {"type": "string"}
    out = {"type": "object", "properties": props}
    if params.get("required"):
        out["required"] = params["required"]
    return out


def minify(tool: dict) -> dict:
    fn = tool["function"]
    return {"type": "function", "function": {
        "name": fn["name"],
        "description": first_sentence(fn.get("description", "")),
        "parameters": minify_params(fn.get("parameters") or {}),
    }}


def ntok(s: str) -> int:
    return len(TOK.encode(s, add_special_tokens=False).ids)


full_tools = SPEC["tools"]
mini_all = [minify(t) for t in full_tools]
mini_core = [minify(t) for t in full_tools if t["function"]["name"] in CORE]
compact_sys = COMPACT_SYSTEM.format(cwd="/home/dinga/Projects/colib")

variants = [
    ("current (full system + 36 tools)", SPEC["system"], full_tools),
    ("compact system + 36 minimal tools", compact_sys, mini_all),
    ("compact system + 15 core tools", compact_sys, mini_core),
]

print(f"{'variant':38s} {'system':>8} {'tools':>8} {'total tok':>10} {'est TTFT':>10}")
print("-" * 78)
RATE = 12.0  # measured prefill tok/s for this engine
for label, sysmsg, tools in variants:
    st, tt = ntok(sysmsg), ntok(json.dumps(tools))
    total = st + tt + 60
    mins = total / RATE / 60.0
    print(f"{label:38s} {st:8d} {tt:8d} {total:10d} {mins:9.1f}m")

json.dump({"system": compact_sys, "tools": mini_all}, open("/tmp/lf_prompt_min.json", "w"))
json.dump({"system": compact_sys, "tools": mini_core}, open("/tmp/lf_prompt_core.json", "w"))
print("\nwrote /tmp/lf_prompt_min.json and /tmp/lf_prompt_core.json")
print("\nexample minified tool:")
print(json.dumps(next(t for t in mini_core if t["function"]["name"] == "read_file"), indent=1))
