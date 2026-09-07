#!/usr/bin/env python3
"""Score a deep-lane model on the review task LocalForge actually gives it.

The activation gate only proves the model emits a well-formed five-field JSON
verdict. That is necessary and nowhere near sufficient: a model can produce
perfectly-shaped JSON full of nothing. The open question for the Qwen swap has
never been speed -- it is whether it reviews as well as Ornith, and
Qwen3.8-Flash-Next has no published coding benchmark at all.

Each case is a small diff with one planted, unambiguous defect, plus the
keywords a reviewer must mention to have actually seen it. Scoring is
deliberately crude and mechanical:

  json      -- parsed as one object with exactly the five required fields
  verdict   -- flagged changes_required (every case here has a real defect)
  caught    -- required_actions mention the planted problem

A model that guesses "changes_required" every time scores well on `verdict` and
badly on `caught`, which is the distinction that matters.

    python3 review_quality_bench.py --model <container> --ram-gb 12 [--out f.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import pathlib
import subprocess
import tempfile
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "qwen"

PROTOCOL = (
    "Respond with exactly one JSON object and no prose, no markdown fence, and no "
    "commentary. It must have exactly these top-level fields: verdict, "
    "required_actions, non_blocking_notes, focused_test_evidence_reviewed, "
    "recommendation. Use verdict \"approved\" only when required_actions is empty. "
    "Use verdict \"changes_required\" when any actionable suggestion remains."
)

CASES = [
    {
        "name": "null-instead-of-throw",
        "diff": "-  if (!user) throw new NotFoundError(id);\n"
                "+  if (!user) return null;\n"
                "   return user.profile;",
        "keywords": ["null", "throw", "caller", "crash", "crashes", "undefined", "crashing"],
    },
    {
        "name": "off-by-one-slice",
        "diff": "-  return items.slice(0, limit);\n"
                "+  return items.slice(0, limit + 1);",
        "keywords": ["off-by-one", "off by one", "limit", "extra", "one more", "boundary"],
    },
    {
        "name": "swallowed-exception",
        "diff": "   try { await flush(); }\n"
                "-  catch (e) { logger.error(e); throw e; }\n"
                "+  catch (e) { }",
        "keywords": ["swallow", "silent", "catch", "error", "log", "rethrow", "ignored"],
    },
    {
        "name": "missing-await",
        "diff": "-  await db.commit();\n"
                "+  db.commit();",
        "keywords": ["await", "async", "promise", "race", "unhandled", "before"],
    },
    {
        "name": "auth-check-removed",
        "diff": "-  if (!ctx.user.isAdmin) return forbidden();\n"
                "   return deleteAccount(ctx.params.id);",
        "keywords": ["auth", "admin", "permission", "authorization", "access", "privilege"],
    },
]


def build_prompt(case: dict) -> str:
    return (
        f"{PROTOCOL}\n\n"
        "Work specification: the change must preserve existing behaviour and be covered by tests.\n"
        "Focused test evidence: none provided.\n"
        f"Diff under review:\n{case['diff']}\n"
    )


def extract_json(raw: str):
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:
                    return None
    return None


REQUIRED = {"verdict", "required_actions", "non_blocking_notes",
            "focused_test_evidence_reviewed", "recommendation"}


def run_case(model: str, case: dict, ram_gb: int, ctx: int, ngen: int, timeout: int,
             expert_q3: bool = False) -> dict:
    env = dict(os.environ)
    env.update({
        "SNAP": model, "MTP": "0", "CTX": str(ctx), "TEXT": "1",
        "NGEN": str(ngen), "RAM_GB": str(ram_gb), "RAM_HEADROOM_GB": "2",
        "PROMPT": build_prompt(case),
    })
    # CHAT=1 wraps the prompt in the model's chat template, which is what
    # LocalForge does -- it sends {role:system},{role:user} through the
    # OpenAI-compatible server. Sending a raw completion prompt instead makes an
    # instruct model ramble, and scored Ornith 0/1 on a run that had in fact
    # completed normally. The template opens <think>, so the budget has to cover
    # reasoning before the JSON.
    env["CHAT"] = "1"
    # The review contract is one JSON object. Reasoning first spends the output
    # budget before the first brace at this decode rate, so it is disabled
    # unless a caller asks for it.
    env.setdefault("NO_THINK", "1")
    # A q3 container needs EXPERT_Q3 or the engine exits immediately. The first
    # run of this bench scored Ornith 0/5 in half a second per case for exactly
    # that reason -- a harness bug reported as a model result.
    if expert_q3:
        env["EXPERT_Q3"] = "1"
    started = time.time()
    # Output goes to files, not pipes. capture_output=True waits for EOF on the
    # pipes rather than for the child to exit, so anything still holding the
    # write end keeps it waiting after the engine is gone -- observed hanging
    # 58 minutes on a run whose engine had already exited, with the timeout
    # never reached because it was set to two hours. Files cannot deadlock, and
    # the wait is on the process itself.
    with tempfile.TemporaryDirectory() as tmp:
        out_path = pathlib.Path(tmp) / "stdout"
        err_path = pathlib.Path(tmp) / "stderr"
        with out_path.open("w") as fo, err_path.open("w") as fe:
            proc = subprocess.Popen([str(ENGINE)], env=env, stdout=fo, stderr=fe,
                                    stdin=subprocess.DEVNULL, text=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=60)
                # Keep what it managed to produce. A timeout that discards the
                # partial output says only "too slow" -- it cannot distinguish a
                # model reasoning its way toward a good answer from one emitting
                # nonsense, and re-running to find out costs another 90 minutes.
                partial = ""
                try:
                    partial = out_path.read_text(errors="replace")[-1500:]
                except OSError:
                    pass
                return {"name": case["name"], "json": False, "verdict_ok": False,
                        "caught": False, "seconds": timeout, "note": "timeout",
                        "raw_tail": partial}
        raw = out_path.read_text(errors="replace")
        err = err_path.read_text(errors="replace")
    elapsed = time.time() - started
    # Surface engine failures instead of scoring them as a bad review. An early
    # run reported Ornith 0/5 at half a second per case; the engine had actually
    # died on its RAM-plan check because a previous model had not yet released
    # memory, and stderr was being thrown away.
    # "Did the engine generate?" is the question, and elapsed time is a poor
    # proxy for it: a small container loads in 0.2s and legitimately finishes
    # fast, which the old `elapsed < 5` guard reported as a failure. TEXT=1
    # makes the engine print a "text:" marker before its output, so its absence
    # means nothing was generated whatever the clock says.
    # The engine prints "text:" when it has a tokenizer and "tokens:" when it
    # does not (TEXT=1 only takes effect with one). Either marker means it
    # reached generation; neither means it did not.
    if proc.returncode != 0 or not ("text:" in raw or "tokens:" in raw):
        tail = (err or "").strip().splitlines()
        why = tail[-1] if tail else f"exit {proc.returncode}, no generation marker in stdout"
        return {"name": case["name"], "json": False, "verdict_ok": False,
                "caught": False, "seconds": round(elapsed, 1),
                "error": why[:200]}
    obj = extract_json(raw)
    ok = isinstance(obj, dict) and set(obj) == REQUIRED
    verdict_ok = ok and str(obj.get("verdict", "")).strip() == "changes_required"
    body = json.dumps(obj.get("required_actions", ""), ensure_ascii=False).lower() if ok else ""
    caught = any(k in body for k in case["keywords"]) if ok else False
    return {"name": case["name"], "json": bool(ok), "verdict_ok": bool(verdict_ok),
            "caught": bool(caught), "seconds": round(elapsed, 1),
            "actions": obj.get("required_actions") if ok else None,
            # A failed case costs ~40 minutes to reproduce; keep the text.
            "raw_tail": None if ok else raw[-600:]}


def run_case_http(endpoint: str, model_id: str, case: dict, timeout: int,
                  max_tokens: int) -> dict:
    """Score one case against an OpenAI-compatible endpoint.

    Some engines have no chat mode in their CLI at all: glm53 takes a raw
    prompt or token ids, and the chat template lives in openai_server.py. Going
    over HTTP is therefore the only way to score them, and it is also the path
    LocalForge actually uses -- the direct-engine path above approximates it
    with CHAT=1. Scoring is shared, so the two backends stay comparable.

    The prompt is split system/user the way the harness sends it rather than
    concatenated, because that is what the template renders.
    """
    body = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": PROTOCOL},
            {"role": "user", "content": build_prompt(case).split("\n\n", 1)[1]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        return {"name": case["name"], "json": False, "verdict_ok": False, "caught": False,
                "seconds": round(time.time() - started, 1),
                "error": f"HTTP {exc.code}: {detail}"}
    except Exception as exc:                                   # noqa: BLE001
        # A timeout here is the endpoint's, not the process's: keep it a result
        # rather than an exception so one slow case does not lose the others.
        return {"name": case["name"], "json": False, "verdict_ok": False, "caught": False,
                "seconds": round(time.time() - started, 1), "error": str(exc)[:200]}
    elapsed = time.time() - started

    raw = ""
    try:
        raw = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return {"name": case["name"], "json": False, "verdict_ok": False, "caught": False,
                "seconds": round(elapsed, 1),
                "error": f"no message content: {json.dumps(payload)[:200]}"}

    obj = extract_json(raw)
    ok = isinstance(obj, dict) and set(obj) == REQUIRED
    verdict_ok = ok and str(obj.get("verdict", "")).strip() == "changes_required"
    text = json.dumps(obj.get("required_actions", ""), ensure_ascii=False).lower() if ok else ""
    caught = any(k in text for k in case["keywords"]) if ok else False
    metrics = payload.get("shiftwing_metrics") or {}
    return {"name": case["name"], "json": bool(ok), "verdict_ok": bool(verdict_ok),
            "caught": bool(caught), "seconds": round(elapsed, 1),
            "actions": obj.get("required_actions") if ok else None,
            "ttft_ms": metrics.get("ttft_ms"),
            "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
            "raw_tail": None if ok else raw[-600:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ram-gb", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ngen", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--expert-q3", action="store_true",
                    help="required for q3 containers such as ornith397")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N cases (a full pass is hours on this hardware)")
    ap.add_argument("--endpoint",
                    help="score over an OpenAI-compatible API (e.g. "
                         "http://127.0.0.1:8000/v1) instead of running the "
                         "engine directly; required for engines whose CLI has "
                         "no chat mode, such as glm53")
    ap.add_argument("--model-id", default=None,
                    help="model id to request from --endpoint (default: --model)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output budget over --endpoint (default: --ngen)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = []
    cases = CASES[:a.limit] if a.limit else CASES
    for case in cases:
        if a.endpoint:
            r = run_case_http(a.endpoint, a.model_id or a.model, case, a.timeout,
                              a.max_tokens or a.ngen)
        else:
            r = run_case(a.model, case, a.ram_gb, a.ctx, a.ngen, a.timeout, a.expert_q3)
        rows.append(r)
        note = f"  ENGINE FAILED: {r['error']}" if r.get("error") else ""
        print(f"  {r['name']:<24} json={int(r['json'])} verdict={int(r['verdict_ok'])} "
              f"caught={int(r['caught'])}  {r['seconds']}s{note}", flush=True)

    n = len(rows)
    summary = {
        "model": a.model,
        "endpoint": a.endpoint,
        "cases": n,
        "json_ok": sum(r["json"] for r in rows),
        "verdict_ok": sum(r["verdict_ok"] for r in rows),
        "caught": sum(r["caught"] for r in rows),
        "seconds_total": round(sum(r["seconds"] for r in rows), 1),
        "rows": rows,
    }
    print(f"\n{a.model}: json {summary['json_ok']}/{n}  "
          f"verdict {summary['verdict_ok']}/{n}  caught {summary['caught']}/{n}  "
          f"({summary['seconds_total']}s)")
    if a.out:
        a.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
