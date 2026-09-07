#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible HTTP gateway for the Shiftwing Qwen engine.

Adapted from JustVugg/colibri at commit
81f08a09e5651ce52616dc720f68810f9021c0be (Apache-2.0).
"""

import argparse
import codecs
import collections
import contextlib
import json
import math
import mimetypes
import os
import select
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent


def default_engine():
    """Return the Qwen engine built next to this gateway."""
    for name in ("qwen", "qwen.exe"):
        candidate = HERE / name
        if candidate.exists():
            return candidate
    return HERE / "qwen"
END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
MAX_BODY = 4 << 20
PROFILE_TURNS = 120           # rolling window of per-turn PERF snapshots kept for /profile
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass

class FirstModelOutputTimeoutError(RuntimeError):
    pass



def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


def _engine_error(fields, message):
    """Turn an engine ERROR frame into the right exception type.

    CONTEXT_EXCEEDED is a client mistake, not a server fault: the prompt is longer than the
    engine's context. Report it the way every OpenAI-compatible server does, so clients that
    know how to compact a conversation actually get the chance to (previously the engine
    silently truncated the prompt instead, which is #401)."""
    if fields and fields[0] == "CONTEXT_EXCEEDED":
        limit = fields[2] if len(fields) > 2 else "the context"
        used = fields[1] if len(fields) > 1 else "?"
        return APIError(400,
                        f"This model's maximum context length is {limit} tokens, however your "
                        f"messages resulted in at least {used} tokens. Please shorten the "
                        f"conversation, or restart the server with a larger CTX.",
                        "messages", "context_length_exceeded")
    return RuntimeError(message)


class GenerationScheduler:
    """Bounded FIFO admission for the engine's independent KV contexts."""

    def __init__(self, max_queue=8, queue_timeout=300, capacity=1):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.capacity = capacity
        self.free_slots = set(range(capacity))
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = 0
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None, slot=None):
        ticket = object()
        entry = (ticket, slot)          # (#B2) remember each waiter's target slot for fair, per-slot admission
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active >= self.capacity or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(entry)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(entry)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                available = min(self.free_slots) if slot is None and self.free_slots else slot
                # (#B2) Admit as soon as our target slot is free AND no strictly-earlier
                # waiter also wants it (an earlier waiter "wants" it if it is any-slot or
                # pinned to the same slot). This replaces the old strict FIFO-head rule,
                # which let a head pinned to a busy slot block every request behind it —
                # even ones targeting a currently-free slot (head-of-line blocking).
                # ponytail: O(queue) scan per wakeup — negligible at the default max_queue;
                # switch to per-slot wait sets if max_queue is ever raised to thousands.
                can_admit = available in self.free_slots
                if can_admit:
                    for t2, s2 in self.queue:
                        if t2 is ticket:
                            break
                        if s2 is None or s2 == available:
                            can_admit = False
                            break
                if can_admit:
                    break
                if cancelled and cancelled():
                    self.queue.remove(entry)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(entry)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.remove(entry)
            self.free_slots.remove(available)
            self.active += 1
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        try:
            yield wait_seconds, available
        finally:
            with self.condition:
                self.active -= 1
                self.free_slots.add(available)
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "capacity": self.capacity,
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Shiftwing currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


# ---- GLM-5.2 tool calling -----------------------------------------------------------------
# The model expresses tool calls as ordinary text (from chat_template.jinja):
#   <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# and tool results come back as <|observation|><tool_response>{content}</tool_response>.
# We render those markers into the prompt and parse them back into OpenAI `tool_calls`.
import re

BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN,  TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_BOX_RE  = re.compile(re.escape(BOX_START) + r"(.*?)" + re.escape(BOX_END), re.DOTALL)
_ARG_RE  = re.compile(r"<arg_key>([^<]*)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"\s*([A-Za-z0-9_.\-]+)")
_TAG_RE  = re.compile(r"</?arg_key>|</?arg_value>")
# A closing tag the model started but never finished ("</tool_cal", "</tool"), at end of reply.
_PARTIAL_END_RE = re.compile(r"<(?:/(?:t(?:o(?:o(?:l(?:_(?:c(?:a(?:l)?)?)?)?)?)?)?)?)?\Z")

# De-mangler: opt-in recovery for heavily-quantized models that drop the
# <arg_key>K</arg_key><arg_value> structure. Default OFF (never rewrites well-formed output).
_SALVAGE = os.environ.get("COLI_TOOL_SALVAGE", "0") == "1"


def _tool_param_order(tools):
    """name -> ordered param names (required first) from the request schema, for de-mangling."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = ((fn.get("parameters") or {}).get("properties") or {})
        required = list((fn.get("parameters") or {}).get("required") or [])
        out[name] = required + [p for p in params if p not in required]
    return out


def _tool_param_types(tools):
    """name -> {param: declared JSON-schema type}. The model emits every argument as text;
    without the schema a string-typed value that happens to look numeric ("12345" for an
    order id, an SKU, a phone number) would be json.loads()'d into an int and the tool would
    receive the wrong type."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        types = {}
        for key, spec in props.items():
            if isinstance(spec, dict):
                t = spec.get("type")
                if isinstance(t, list):          # {"type": ["string", "null"]}
                    t = next((x for x in t if x != "null"), None)
                types[key] = t
        out[name] = types
    return out


def _coerce_arg(value, declared):
    """Decode a raw <arg_value> according to the declared schema type.

    A string-typed parameter is kept verbatim -- never parsed as JSON. Everything else keeps
    the previous permissive behaviour (parse if it parses, otherwise leave as text)."""
    if declared == "string":
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if declared in ("integer", "number") and isinstance(parsed, bool):
        return value                              # `true` is not a number
    if declared and declared not in ("integer", "number", "boolean", "object", "array"):
        return value
    return parsed


def _unclosed_tail(reply, tools):
    """Body of a trailing <tool_call> that was never closed, or None.

    Only returned when the recovery is unambiguous, so ordinary prose that merely mentions
    "<tool_call>" can never be turned into a call. Both conditions must hold:
      * the last BOX_START is not followed by a BOX_END (a closed box is the strict parser's job);
      * the tail carries a complete <arg_key>..</arg_value> pair, OR it is exactly the name of a
        tool the client declared (the zero-argument case).
    """
    start = reply.rfind(BOX_START)
    if start < 0 or BOX_END in reply[start:]:
        return None
    inner = _PARTIAL_END_RE.sub("", reply[start + len(BOX_START):])
    if _ARG_RE.search(inner):
        return inner
    declared = {(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                for t in (tools or []) if isinstance(t, dict)}
    return inner if inner.strip() in declared else None


def parse_glm_tool_calls(reply, tools=None):
    """Return (content, tool_calls). Strict GLM parse; optional de-mangler (COLI_TOOL_SALVAGE=1)
    rescues malformed int4 output by mapping a lone payload onto the tool's primary parameter."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    # #401: a box the model opened but never closed -- it ran out of budget, or the closing tag
    # came out mangled ("</tool_cal"). The call itself is often perfectly well-formed, but the
    # strict regex needs BOTH tags, so the client used to get *zero* tool_calls. Recover the tail,
    # but only when it is unambiguous (see _unclosed_tail) so prose can never fabricate a call.
    boxes = [m.group(1) for m in _BOX_RE.finditer(reply)]
    tail = _unclosed_tail(reply, tools)
    if tail is not None:
        boxes.append(tail)
    for inner in boxes:
        name_match = _NAME_RE.match(inner)
        name = name_match.group(1) if name_match else inner.strip()
        args = {}
        types = param_types.get(name, {})
        for arg in _ARG_RE.finditer(inner):
            key, value = arg.group(1), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            rest = inner[name_match.end():] if name_match else ""
            payload = _TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if tools and not calls and re.search(r"</?tool_call>|</?arg_key>|</?arg_value>", reply):
        # Diagnosi per la #401: il client ha dichiarato i tools e il modello ha PROVATO la
        # sintassi, ma il parse rigoroso non ha agganciato nulla (tipico output int4 storpiato).
        # EN: #401 field diagnosis: tools were declared and the model attempted the syntax,
        # EN: but the strict parse matched nothing (typically quantization-mangled output).
        sys.stderr.write("[api] tools declared and tool-call markers present, but no call "
                         "parsed -- output may be quantization-mangled; try COLI_TOOL_SALVAGE=1\n")
        sys.stderr.flush()
    text = _BOX_RE.sub("", reply)
    if tail is not None:                       # drop the recovered tail from the visible content
        text = text[:text.rindex(BOX_START)]
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        dm, rec = len(salvaged), (1 if tail is not None else 0)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d unclosed-recovered, "
                         "%d de-mangled [%s]%s\n"
                         % (len(calls), max(0, len(calls) - dm - rec), rec, dm,
                            "CLEAN" if dm == 0 and rec == 0 else "RECOVERED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


def render_glm_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    if tools:
        # AUTHORITATIVE GLM-5.2 tool-declaration block (byte-matches chat_template.jinja): the
        # `# Tools` + <tools></tools> XML structure is what the model was trained on. A made-up
        # preamble makes it hallucinate other frameworks' syntax (e.g. `end_action`).
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
        if forced:
            prompt.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the functions above. Do not answer directly.")
    prev_tool = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "user":
            prompt.append(f"<|user|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "assistant":
            # content may be null when the message is purely tool_calls
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            prompt.append(f"<|assistant|><think></think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:                       # one <|observation|> per consecutive tool run
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + content_text(message.get("content"), f"messages.{index}.content") + TR_CLOSE)
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


# ---- Qwen3.5 text chat and Qwen3-XML tools -----------------------------------------------
# The functions above are the vendored GLM protocol implementation. They were dead code
# while the deep lane ran on Ornith, shadowed by the Qwen definitions that follow. They
# are live again for the GLM-5.3-Flash lane and are now reached by name rather than by
# definition order: render_model_chat and parse_model_tool_calls dispatch on the family
# the converter stamped into config.json. GLM-5.3's chat_template.jinja is [gMASK]<sop>
# with <|system|>/<|user|>/<|assistant|> and contains no <|im_start|> at all, so serving
# GLM through the Qwen renderer sends control tokens the model was never trained on.

_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*"
    r"(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*"
    r"(.*?)\s*</parameter>",
    re.DOTALL,
)


def _qwen_argument(text):
    value = text.strip()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def parse_tool_calls(reply, tools=None):
    """Parse Qwen3-XML calls and hide the model's private reasoning channel."""
    _reasoning, body = split_assistant_reply(reply)
    calls = []
    matches = list(_QWEN_TOOL_CALL_RE.finditer(body))
    for match in matches:
        arguments = {}
        for parameter in _QWEN_PARAMETER_RE.finditer(match.group(2)):
            arguments[parameter.group(1)] = _qwen_argument(parameter.group(2))
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": match.group(1),
                "arguments": json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                ),
            },
        })
    if not matches:
        return body, []
    content = body[:matches[0].start()].strip()
    suffix = body[matches[-1].end():].strip()
    if suffix:
        # The official template forbids suffix text, but preserving it is safer
        # than silently deleting malformed model output.
        content = (content + "\n\n" if content else "") + suffix
    return content, calls


def split_assistant_reply(reply, assume_reasoning=False):
    """Return Qwen's private reasoning and public body.

    When the generation prompt already contains ``<think>`` the generated
    suffix starts inside that element, so only ``</think>`` appears in the
    decoded response. ``assume_reasoning`` distinguishes that case from an
    ordinary answer that happens to contain the same literal text.
    """
    body = reply.strip()
    reasoning = ""
    if body.startswith(THINK_OPEN) and THINK_CLOSE in body:
        reasoning, body = body[len(THINK_OPEN):].split(THINK_CLOSE, 1)
    elif assume_reasoning and THINK_CLOSE in body:
        reasoning, body = body.split(THINK_CLOSE, 1)
    return reasoning.strip(), body.strip()


def parse_assistant_reply(reply, tools=None, assume_reasoning=False, family=None):
    reasoning, body = split_assistant_reply(reply, assume_reasoning)
    content, calls = (parse_model_tool_calls(family, body, tools) if tools
                      else (body, []))
    return reasoning, content, calls


def _qwen_message_text(message, index):
    raw = message.get("content")
    if raw is None and message.get("role") == "assistant":
        return ""
    return content_text(raw, f"messages.{index}.content").strip()


def _qwen_tool_function(tool):
    if not isinstance(tool, dict):
        raise APIError(400, "Each tool must be an object.", "tools")
    function = tool.get("function", tool)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise APIError(400, "Each tool requires a function name.", "tools")
    return function


def _qwen_tool_xml(call):
    function = call.get("function", call) if isinstance(call, dict) else {}
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise APIError(400, "Assistant tool calls require a function name.",
                       "messages.tool_calls")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise APIError(400, "Assistant tool-call arguments must be valid JSON.",
                           "messages.tool_calls.function.arguments") from error
    if not isinstance(arguments, dict):
        raise APIError(400, "Assistant tool-call arguments must be an object.",
                       "messages.tool_calls.function.arguments")
    parts = [f"<tool_call>\n<function={name}>\n"]
    for key, value in arguments.items():
        rendered = (value if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        parts.append(f"<parameter={key}>\n{rendered}\n</parameter>\n")
    parts.append("</function>\n</tool_call>")
    return "".join(parts)


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None, cache_prefix_compatible=False):
    """Render the Qwen3.5 text-only ChatML/tool subset.

    The default is byte-compatible with the official template.  The serving
    path enables ``cache_prefix_compatible`` so a prior assistant turn retains
    the generation-time ``<think>`` envelope and the newly rendered transcript
    is an exact byte extension of the engine's cached token stream.
    """
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")

    selected_tools = list(tools or [])
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            selected_tools = [
                tool for tool in selected_tools
                if _qwen_tool_function(tool).get("name") == forced
            ]
    elif tool_choice == "none":
        selected_tools = []

    # OpenAI permits a leading run of system/developer messages. Qwen has one
    # system turn, so join that run without changing later conversational roles.
    system_parts = []
    first_body = 0
    for index, message in enumerate(messages):
        if message.get("role") not in ("system", "developer"):
            first_body = index
            break
        system_parts.append(_qwen_message_text(message, index))
    else:
        first_body = len(messages)

    prompt = []
    if selected_tools:
        prompt.append("<|im_start|>system\n# Tools\n\n"
                      "You have access to the following functions:\n\n<tools>")
        for tool in selected_tools:
            prompt.append("\n" + json.dumps(tool, ensure_ascii=False))
        prompt.append(
            "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
            "following format with NO suffix:\n\n<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\nthat can span\n"
            "multiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
            "<IMPORTANT>\nReminder:\n"
            "- Function calls MUST follow the specified format: an inner "
            "<function=...></function> block must be nested within "
            "<tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in "
            "natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like "
            "normal with your current knowledge and do not tell the user about "
            "function calls\n</IMPORTANT>"
        )
        if forced:
            prompt.append(f"\n\nYou must call `{forced}`.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the available functions.")
        if system_parts:
            prompt.append("\n\n" + "\n\n".join(part for part in system_parts if part))
        prompt.append("<|im_end|>\n")
    elif system_parts:
        prompt.append("<|im_start|>system\n" + "\n\n".join(system_parts)
                      + "<|im_end|>\n")

    last_query = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_query = index

    previous_tool = False
    for index in range(first_body, len(messages)):
        message = messages[index]
        role = message.get("role")
        if role in ("system", "developer"):
            raise APIError(400, "System/developer messages must precede the conversation.",
                           f"messages.{index}.role")
        if role == "user":
            prompt.append("<|im_start|>user\n" + _qwen_message_text(message, index)
                          + "<|im_end|>\n")
        elif role == "assistant":
            content = _qwen_message_text(message, index)
            reasoning = message.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = ""
                if content.startswith(THINK_OPEN) and THINK_CLOSE in content:
                    reasoning, content = content[len(THINK_OPEN):].split(THINK_CLOSE, 1)
                    reasoning, content = reasoning.strip(), content.strip()
            prompt.append("<|im_start|>assistant\n")
            if index > last_query or cache_prefix_compatible:
                prompt.append(f"<think>\n{reasoning.strip()}\n</think>\n\n")
            prompt.append(content)
            for call in message.get("tool_calls") or []:
                if content:
                    prompt.append("\n\n")
                    content = ""
                prompt.append(_qwen_tool_xml(call))
            prompt.append("<|im_end|>\n")
        elif role == "tool":
            if not previous_tool:
                prompt.append("<|im_start|>user")
            prompt.append("\n<tool_response>\n" + _qwen_message_text(message, index)
                          + "\n</tool_response>")
            next_is_tool = (index + 1 < len(messages)
                            and messages[index + 1].get("role") == "tool")
            if not next_is_tool:
                prompt.append("<|im_end|>\n")
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        previous_tool = role == "tool"

    prompt.append("<|im_start|>assistant\n")
    prompt.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(prompt)


def parse_model_tool_calls(family, reply, tools=None):
    """Parse tool calls in the dialect the family emits.

    GLM writes <arg_key>/<arg_value> boxes; Qwen writes Qwen3-XML. Reading one
    with the other's parser silently yields zero calls rather than an error,
    so the family decides.
    """
    if isinstance(family, str) and family.startswith("glm"):
        return parse_glm_tool_calls(reply, tools)
    return parse_tool_calls(reply, tools)


def snapshot_model_family(snapshot):
    """Return the converter's explicit family marker, defaulting to Qwen."""
    if snapshot is None:
        return "qwen3.5"
    path = Path(snapshot) / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "qwen3.5"
    family = config.get("shiftwing_model_family", config.get("colib_model_family"))
    return family if isinstance(family, str) and family else "qwen3.5"


def render_model_chat(snapshot, family, messages, enable_thinking=False,
                      reasoning_effort=None, tools=None, tool_choice=None,
                      cache_prefix_compatible=True):
    """Dispatch chat rendering by the converted snapshot's explicit family."""
    if isinstance(family, str) and family.startswith("glm"):
        return render_glm_chat(messages, enable_thinking, reasoning_effort,
                               tools, tool_choice)
    if family != "ornith-1.0":
        return render_chat(
            messages,
            enable_thinking,
            reasoning_effort,
            tools,
            tool_choice,
            cache_prefix_compatible=cache_prefix_compatible,
        )
    if snapshot is None:
        raise APIError(500, "Ornith rendering requires the model snapshot path.")

    selected_tools = list(tools or [])
    selected_messages = [dict(message) for message in messages]
    choice_instruction = None
    if tool_choice == "none":
        selected_tools = []
    elif tool_choice == "required":
        choice_instruction = (
            "You must call one of the provided functions. Do not answer directly."
        )
    elif isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            selected_tools = [
                tool for tool in selected_tools
                if _qwen_tool_function(tool).get("name") == forced
            ]
            choice_instruction = (
                f"You must call the function `{forced}`. Do not answer directly."
            )
    if choice_instruction:
        if (
            selected_messages
            and selected_messages[0].get("role") == "system"
            and isinstance(selected_messages[0].get("content"), str)
        ):
            selected_messages[0]["content"] = (
                selected_messages[0]["content"].rstrip()
                + "\n\n"
                + choice_instruction
            )
        else:
            selected_messages.insert(
                0, {"role": "system", "content": choice_instruction}
            )
    from tools.ornith_protocol import render_chat as render_ornith_chat

    try:
        return render_ornith_chat(
            Path(snapshot),
            selected_messages,
            tools=selected_tools or None,
            enable_thinking=enable_thinking,
        )
    except (FileNotFoundError, ValueError) as error:
        raise APIError(
            500,
            f"Ornith snapshot chat template could not be rendered: {error}",
            None,
            "model_template_error",
            "server_error",
        ) from error


# ---- Anthropic Messages API (#343) --------------------------------------------------------
# A translation layer, NOT a second engine path: /v1/messages rewrites an Anthropic-shaped
# request into the exact OpenAI-shaped body the existing path already validates, so prompt
# rendering, scheduling, generation and tool parsing stay single-sourced. Only the request
# translation and the response/SSE shapes are new. Claude Code is the reference client.

def _anthropic_block_text(blocks, param):
    """Text out of an Anthropic content array (tool_result content is the same shape)."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        raise APIError(400, "Content must be a string or an array of blocks.", param)
    parts = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise APIError(400, "Shiftwing currently supports text blocks only here.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(block.get("text"), str):
            raise APIError(400, "Text blocks require a string `text` field.", f"{param}.{index}.text")
        parts.append(block["text"])
    return "".join(parts)


def anthropic_to_openai(body):
    """Anthropic request -> (messages, tools, tool_choice) in OpenAI shape."""
    messages = []
    system = body.get("system")
    if isinstance(system, str):
        if system:
            messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = _anthropic_block_text(system, "system")
        if text:
            messages.append({"role": "system", "content": text})
    elif system is not None:
        raise APIError(400, "`system` must be a string or an array of text blocks.", "system")

    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise APIError(400, f"Unsupported message role: {role!r}. Anthropic messages are "
                           "`user` or `assistant`; a system prompt goes in the top-level `system`.",
                           f"messages.{index}.role", "unsupported_role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise APIError(400, "Message content must be a string or an array of blocks.",
                           f"messages.{index}.content")
        texts, calls, results = [], [], []
        for j, block in enumerate(content):
            where = f"messages.{index}.content.{j}"
            if not isinstance(block, dict):
                raise APIError(400, "Each content block must be an object.", where)
            kind = block.get("type")
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise APIError(400, "Text blocks require a string `text` field.", f"{where}.text")
                texts.append(block["text"])
            elif kind == "tool_use":
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise APIError(400, "`tool_use` blocks require a string `name`.", f"{where}.name")
                arguments = block.get("input")
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    raise APIError(400, "`tool_use.input` must be an object.", f"{where}.input")
                calls.append({"id": block.get("id") or ("toolu_" + uuid.uuid4().hex[:24]),
                              "type": "function",
                              "function": {"name": name,
                                           "arguments": json.dumps(arguments, ensure_ascii=False)}})
            elif kind == "tool_result":
                results.append({"role": "tool",
                                "tool_call_id": block.get("tool_use_id") or "",
                                "content": _anthropic_block_text(block.get("content", ""),
                                                                 f"{where}.content")})
            else:
                raise APIError(400, "Shiftwing supports `text`, `tool_use` and `tool_result` "
                               "content blocks only.", f"{where}.type", "unsupported_content_type")
        # tool results precede the user's own text: they answer the previous assistant turn
        messages.extend(results)
        text = "".join(texts)
        if role == "assistant":
            if text or calls:
                entry = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                messages.append(entry)
        elif text or not results:
            messages.append({"role": "user", "content": text})
    return messages


def anthropic_tools(body):
    """Anthropic tools/tool_choice -> OpenAI shape (validated downstream by generation_options)."""
    raw = body.get("tools")
    if raw is None:
        tools = None
    elif not isinstance(raw, list):
        raise APIError(400, "`tools` must be an array.", "tools")
    else:
        tools = []
        for index, tool in enumerate(raw):
            if not isinstance(tool, dict):
                raise APIError(400, "Each tool must be an object.", f"tools.{index}")
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise APIError(400, "Each tool requires a string `name`.", f"tools.{index}.name")
            schema = tool.get("input_schema")
            if schema is not None and not isinstance(schema, dict):
                raise APIError(400, "`input_schema` must be an object.", f"tools.{index}.input_schema")
            function = {"name": name, "parameters": schema or {"type": "object", "properties": {}}}
            if isinstance(tool.get("description"), str):
                function["description"] = tool["description"]
            tools.append({"type": "function", "function": function})
        tools = tools or None

    choice = body.get("tool_choice")
    if choice is None:
        return tools, None
    if not isinstance(choice, dict):
        raise APIError(400, "`tool_choice` must be an object.", "tool_choice")
    kind = choice.get("type")
    if kind == "auto":
        return tools, "auto"
    if kind == "any":
        return tools, "required"
    if kind == "none":
        return tools, "none"
    if kind == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise APIError(400, "`tool_choice.name` is required when type is `tool`.",
                           "tool_choice.name")
        return tools, {"type": "function", "function": {"name": name}}
    raise APIError(400, "`tool_choice.type` must be auto, any, none, or tool.", "tool_choice.type",
                   "unsupported_value")


# Generic whitespace-tolerant JSON grammar for response_format {"type": "json_object"}.
# Draft-source semantics: positions with one legal byte draft; jws points just keep
# the walker alive through the model's own spacing (see docs/grammar-draft.md).
GENERIC_JSON_GBNF = (
    'root ::= jws jval jws\n'
    'jval ::= jobj | jarr | jstr | jnum | "true" | "false" | "null"\n'
    'jobj ::= "{" jws ( jstr jws ":" jws jval jws ( "," jws jstr jws ":" jws jval jws )* )? "}"\n'
    'jarr ::= "[" jws ( jval jws ( "," jws jval jws )* )? "]"\n'
    'jstr ::= "\\"" jchar* "\\""\n'
    'jchar ::= [^"\\\\\\x00-\\x1f] | "\\\\" ( ["\\\\/bfnrt] | "u" jhex jhex jhex jhex )\n'
    'jhex ::= [0-9a-fA-F]\n'
    'jnum ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( ( "e" | "E" ) ( "+" | "-" )? [0-9]+ )?\n'
    'jws ::= ( " " | "\\t" | "\\n" | "\\r" )*\n'
)

def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "shiftwing currently supports `n=1` only.", "n", "unsupported_value")
    # `tools`/`functions` are handled by render_chat (declaration) + parse_tool_calls (output).
    # Validate tools/functions structure early so malformed input fails with a clear error.
    tools_raw = body.get("tools") or body.get("functions")
    if tools_raw is not None:
        if not isinstance(tools_raw, list):
            raise APIError(400, "`tools` must be a non-empty array.", "tools", "invalid_value")
        if not tools_raw:
            raise APIError(400, "`tools` must be a non-empty array.", "tools", "invalid_value")
        for idx, tool in enumerate(tools_raw):
            if not isinstance(tool, dict):
                raise APIError(400, f"Each tool must be an object, got {type(tool).__name__} at index {idx}.",
                               f"tools.{idx}", "invalid_value")
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            if not isinstance(fn, dict):
                raise APIError(400, f"Tool function must be an object at index {idx}.",
                               f"tools.{idx}.function", "invalid_value")
            if not fn.get("name"):
                raise APIError(400, f"Each tool must have a `name` at index {idx}.",
                               f"tools.{idx}.function.name", "invalid_value")
            if not isinstance(fn["name"], str):
                raise APIError(400, f"Tool `name` must be a string at index {idx}.",
                               f"tools.{idx}.function.name", "invalid_value")
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str):
            if choice not in ("auto", "none", "required"):
                raise APIError(400, "`tool_choice` must be one of \"auto\", \"none\", \"required\", "
                                    "or a function object.", "tool_choice", "unsupported_value")
        elif isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            if not name:
                raise APIError(400, "`tool_choice` function object must include a name.",
                               "tool_choice", "invalid_value")
            declared = [(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                        for t in (body.get("tools") or body.get("functions") or [])]
            if name not in declared:
                raise APIError(400, f"`tool_choice` names {name!r}, which is not in `tools`.",
                               "tool_choice", "invalid_value")
        else:
            raise APIError(400, "`tool_choice` must be a string or a function object.",
                           "tool_choice", "invalid_value")
        if choice != "none" and not (body.get("tools") or body.get("functions")):
            raise APIError(400, "`tool_choice` requires `tools`.", "tool_choice", "invalid_value")
    if body.get("stop") is not None:
        raise APIError(400, "Custom stop sequences are not supported yet.", "stop", "unsupported_parameter")
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    # The first Qwen mux implementation is target-only and does not yet carry a
    # per-request grammar. Refuse structured-output claims at the gateway instead
    # of sending a frame the engine cannot parse.
    grammar = None
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise APIError(400, "Structured response formats are not implemented by the "
                       "Qwen mux yet.", "response_format", "unsupported_parameter")

    maximum = body.get("max_completion_tokens")
    maximum_param = "max_completion_tokens"
    if maximum is None:
        maximum = body.get("max_tokens")
        maximum_param = "max_tokens"
    if maximum is None:
        # Client omitted max_tokens: honor the operator's configured budget (--max-tokens /
        # --ngen), not an arbitrary 256 — `coli serve --ngen 32768` must mean 32768 (#382).
        # Generation still ends at EOS, so this is a cap, not a target.
        maximum = limit
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    temperature = 0.0 if temperature is None else temperature
    top_p = 1.0 if top_p is None else top_p
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise APIError(400, f"`{maximum_param}` must be a positive integer.", maximum_param)
    if maximum > limit:
        maximum = limit   # clamp to the server's --max-tokens cap instead of 400 (#260): OpenAI
                          # clients (opencode/ai-sdk) default to large max_tokens; rejecting breaks them.
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
            not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise APIError(400, "`temperature` must be between 0 and 2.", "temperature")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or
            not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise APIError(400, "`top_p` must be greater than 0 and at most 1.", "top_p")
    if float(temperature) != 0.0 or float(top_p) != 1.0:
        raise APIError(400, "The Phase-9 Qwen mux currently supports greedy decoding "
                       "only (`temperature=0`, `top_p=1`).", None,
                       "unsupported_parameter")
    return maximum, float(temperature), float(top_p), grammar


def read_engine_turn(stream, sentinel, on_bytes):
    pending = b""
    while True:
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("shiftwing engine exited unexpectedly")
        pending += byte
        if pending.endswith(sentinel):
            data = pending[:-len(sentinel)]
            if data:
                on_bytes(data)
            break
        if len(pending) > len(sentinel):
            on_bytes(pending[:-len(sentinel)])
            pending = pending[-len(sentinel):]

    fields = stream.readline().decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
    }


def apply_engine_defaults(env):
    """Install engine defaults that must not be left to the engine's own fallbacks.

    CTX: the engine defaults max_seq to 512 and silently truncated anything longer,
    dropping the tail of the prompt -- including the trailing generation prompt --
    instead of reporting it (#401).  Agent transcripts carrying tool results run
    well past 512 tokens.

    Thread count is deliberately left to OpenMP.  Restricting it to physical cores
    looks right for AVX-512 FMA-bound code, but measured the other way on this
    workload: a full 3575-token prefill ran 374.7s on 16 hardware threads against
    403.0s on 8 physical cores, because the quantized matmul is latency-bound on
    its per-group reductions and SMT fills those bubbles.
    """
    env.setdefault("CTX", "16384")
    return env


def _sequence_split(fields, offset):
    """Read the optional trailing GDN/full-attention pair from a PERF row.

    `attention_s` reports the two summed. The pair is appended after the
    optional CUDA-event suffix, so its offset depends on whether that suffix is
    present, and an engine emitting neither is still valid.
    """
    if len(fields) < offset + 2:
        return None
    try:
        return float(fields[offset]), float(fields[offset + 1])
    except (TypeError, ValueError):
        return None


class Engine:
    def __init__(self, executable, model, cap=8, max_tokens=1024, env=None, kv_slots=1):
        child_env = dict(env or os.environ, SNAP=str(model), SERVE="1", SERVE_BATCH="1",
                         MTP="0", NGEN=str(max_tokens), KV_SLOTS=str(kv_slots))
        apply_engine_defaults(child_env)
        self.process = subprocess.Popen(
            [str(executable)], env=child_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0,
        )
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.prefill_cached = {}
        self.next_request_id = 1
        self.closed = False
        self.dispatcher_error = None
        self.trace = child_env.get("COLI_ENGINE_TRACE", "0") == "1"
        try:
            self.first_model_output_timeout_ms = int(
                child_env.get("FIRST_MODEL_OUTPUT_TIMEOUT_MS", "900000"))
        except ValueError as error:
            raise RuntimeError("FIRST_MODEL_OUTPUT_TIMEOUT_MS must be an integer") from error
        if self.first_model_output_timeout_ms <= 0:
            raise RuntimeError("FIRST_MODEL_OUTPUT_TIMEOUT_MS must be positive")
        try:
            self.prefill_first_progress_timeout_ms = int(
                child_env.get("PREFILL_FIRST_PROGRESS_TIMEOUT_MS", "180000"))
            self.prefill_progress_stall_timeout_ms = int(
                child_env.get("PREFILL_PROGRESS_STALL_TIMEOUT_MS", "120000"))
        except ValueError as error:
            raise RuntimeError("PREFILL_PROGRESS_STALL_TIMEOUT_MS must be an integer") from error
        if self.prefill_first_progress_timeout_ms <= 0:
            raise RuntimeError("PREFILL_FIRST_PROGRESS_TIMEOUT_MS must be positive")
        if self.prefill_progress_stall_timeout_ms <= 0:
            raise RuntimeError("PREFILL_PROGRESS_STALL_TIMEOUT_MS must be positive")
        self.kv_slots = kv_slots
        self.tiers = None
        self.q3_native = None
        self.q3_atlas = None
        self.hwinfo = None
        self.emap = None
        self.hits = None
        self.hits_seq = 0                      # latest "TIERS" snapshot from the engine
        self.profile = collections.deque(maxlen=PROFILE_TURNS)  # per-turn phase timings
        self.profile_seq = 0
        self.resident = None
        startup = self.process.stdout.read(len(READY))
        if startup != READY:
            raise RuntimeError(f"invalid engine handshake: {startup!r}")
        self.dispatcher = threading.Thread(target=self._dispatch_stdout,
                                           name="shiftwing-stdout", daemon=True)
        self.dispatcher.start()

    @staticmethod
    def _stats(fields):
        if len(fields) < 5 or fields[0] != "STAT":
            raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
        return {
            "completion_tokens": int(fields[1]),
            "tokens_per_second": float(fields[2]),
            "cache_hit_percent": float(fields[3]),
            "rss_gb": float(fields[4]),
            "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
            "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
        }

    def _fail_pending(self, error):
        with self.pending_lock:
            requests = list(self.pending.values())
            self.pending.clear()
            self.prefill_cached.clear()
        for events in requests:
            events.put(("error", error))

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if chunk == b"":
                raise RuntimeError("truncated engine DATA payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _dispatch_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline()
                if line == b"":
                    raise RuntimeError("shiftwing engine exited unexpectedly")
                fields = line.decode("utf-8", "replace").strip().split()
                if not fields:
                    continue
                kind = fields[0]
                if self.trace:
                    print(f"[engine<-] {' '.join(fields)}", file=sys.stderr, flush=True)
                if kind == "DATA" and len(fields) == 3:
                    request_id = fields[1]
                    size = int(fields[2])
                    if not 0 <= size <= 65536:
                        raise RuntimeError("invalid engine DATA size")
                    data = self._read_exact(size)
                    if self._read_exact(1) != b"\n":
                        raise RuntimeError("invalid engine DATA terminator")
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                    if events is not None:
                        events.put(("data", data))
                elif kind == "PREFILL_BEGIN" and len(fields) == 4:
                    request_id = fields[1]
                    total, cached = int(fields[2]), int(fields[3])
                    if total < 1 or cached < 0 or cached > total:
                        raise RuntimeError("invalid PREFILL_BEGIN")
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                        self.prefill_cached[request_id] = cached
                    if events is not None:
                        events.put(("progress", {
                            "phase": "prefill", "event": "begin",
                            "prompt_tokens_total": total,
                            "prompt_tokens_cached": cached,
                            "prompt_tokens_prefilled": cached,
                            "elapsed_ms": 0,
                        }))
                elif kind == "PREFILL_PROGRESS" and len(fields) == 5:
                    request_id = fields[1]
                    completed, total, elapsed = map(int, fields[2:])
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                        cached = self.prefill_cached.get(request_id, 0)
                    if total < 1 or completed < cached or completed > total or elapsed < 0:
                        raise RuntimeError("invalid PREFILL_PROGRESS")
                    if events is not None:
                        events.put(("progress", {
                            "phase": "prefill", "event": "progress",
                            "prompt_tokens_total": total,
                            "prompt_tokens_cached": cached,
                            "prompt_tokens_prefilled": completed,
                            "elapsed_ms": elapsed,
                        }))
                elif kind == "PREFILL_END" and len(fields) == 4:
                    request_id = fields[1]
                    total, elapsed = int(fields[2]), int(fields[3])
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                        cached = self.prefill_cached.pop(request_id, 0)
                    if events is not None:
                        events.put(("progress", {
                            "phase": "prefill", "event": "end",
                            "prompt_tokens_total": total,
                            "prompt_tokens_cached": cached,
                            "prompt_tokens_prefilled": total,
                            "elapsed_ms": elapsed,
                        }))
                elif kind == "DONE" and len(fields) >= 7:
                    request_id = fields[1]
                    stats = self._stats(fields[2:])
                    for turn in reversed(self.profile):
                        if turn.get("request_id") == request_id:
                            turn["prompt_tokens"] = stats["prompt_tokens"]
                            turn["completion_tokens"] = stats["completion_tokens"]
                            turn["forwards"] = stats["completion_tokens"]
                            stats["profile"] = dict(turn)
                            break
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                        self.prefill_cached.pop(request_id, None)
                    if events is not None:
                        events.put(("done", stats))
                elif kind == "HWINFO" and len(fields) >= 7:
                    parts = " ".join(fields[6:]).split("|")
                    self.hwinfo = {"cores": int(fields[1]), "ram_total_gb": float(fields[2]),
                                   "ram_avail_gb": float(fields[3]), "gpus": int(fields[4]),
                                   "vram_total_gb": float(fields[5]),
                                   "cpu": parts[0].strip() if len(parts)>0 else "",
                                   "gpu": parts[1].strip() if len(parts)>1 else ""}
                elif kind == "EMAP" and len(fields) == 4:
                    self.emap = {"rows": int(fields[1]), "cols": int(fields[2]), "map": fields[3]}
                elif kind == "HITS" and len(fields) == 4:
                    self.hits = fields[3]
                    self.hits_seq += 1
                elif kind == "PERF" and len(fields) >= 9:
                    # Inclusive per-request interval. Concurrent requests may share
                    # batched kernel time, so phase values are diagnostic, not additive.
                    turn = {
                        "request_id": fields[1],
                        "wall_s": float(fields[2]),
                        "expert_disk_s": float(fields[3]),
                        "expert_wait_s": float(fields[4]),
                        "expert_matmul_s": float(fields[5]),
                        "attention_s": float(fields[6]),
                        "kv_batch_s": float(fields[7]),
                        "lm_head_s": float(fields[8]),
                    }
                    if len(fields) >= 18:
                        turn.update({
                            "cuda_transactions": int(fields[9]),
                            "cuda_setup_s": float(fields[10]),
                            "cuda_routed_hidden_s": float(fields[11]),
                            "cuda_routed_down_s": float(fields[12]),
                            "cuda_route_reduce_s": float(fields[13]),
                            "cuda_shared_hidden_s": float(fields[14]),
                            "cuda_shared_scale_s": float(fields[15]),
                            "cuda_shared_down_s": float(fields[16]),
                            "cuda_download_s": float(fields[17]),
                        })
                    # attention_s sums the GDN and full-attention layers. The
                    # trailing pair splits it. It follows the optional CUDA
                    # suffix, so an engine that emits neither, or only CUDA,
                    # still parses: the split is read only when present.
                    split = _sequence_split(fields, 18 if len(fields) >= 18 else 9)
                    if split:
                        turn["gdn_s"], turn["attn_s"] = split
                    self.profile.append(turn)
                    self.profile_seq += 1
                elif kind == "DPERF" and len(fields) >= 9:
                    request_id = fields[1]
                    for turn in reversed(self.profile):
                        if turn.get("request_id") == request_id:
                            turn.update({
                                "decode_wall_s": float(fields[2]),
                                "decode_expert_disk_s": float(fields[3]),
                                "decode_expert_wait_s": float(fields[4]),
                                "decode_expert_matmul_s": float(fields[5]),
                                "decode_attention_s": float(fields[6]),
                                "decode_kv_batch_s": float(fields[7]),
                                "decode_lm_head_s": float(fields[8]),
                            })
                            if len(fields) >= 18:
                                turn.update({
                                    "decode_cuda_transactions": int(fields[9]),
                                    "decode_cuda_setup_s": float(fields[10]),
                                    "decode_cuda_routed_hidden_s": float(fields[11]),
                                    "decode_cuda_routed_down_s": float(fields[12]),
                                    "decode_cuda_route_reduce_s": float(fields[13]),
                                    "decode_cuda_shared_hidden_s": float(fields[14]),
                                    "decode_cuda_shared_scale_s": float(fields[15]),
                                    "decode_cuda_shared_down_s": float(fields[16]),
                                    "decode_cuda_download_s": float(fields[17]),
                                })
                            split = _sequence_split(
                                fields, 18 if len(fields) >= 18 else 9)
                            if split:
                                turn["decode_gdn_s"], turn["decode_attn_s"] = split
                            break
                elif kind == "CACHE" and len(fields) in (7, 9):
                    request_id = fields[1]
                    for turn in reversed(self.profile):
                        if turn.get("request_id") == request_id:
                            turn.update({
                                "expert_cpu_hits": int(fields[2]),
                                "expert_gpu_hits": int(fields[3]),
                                "expert_misses": int(fields[4]),
                                "expert_read_bytes": int(fields[5]),
                                "expert_direct_bytes": int(fields[6]),
                                "expert_uring_batches": int(fields[7]) if len(fields) == 9 else 0,
                                "expert_uring_reads": int(fields[8]) if len(fields) == 9 else 0,
                            })
                            break
                elif kind == "DCACHE" and len(fields) in (7, 9):
                    request_id = fields[1]
                    for turn in reversed(self.profile):
                        if turn.get("request_id") == request_id:
                            turn.update({
                                "decode_expert_cpu_hits": int(fields[2]),
                                "decode_expert_gpu_hits": int(fields[3]),
                                "decode_expert_misses": int(fields[4]),
                                "decode_expert_read_bytes": int(fields[5]),
                                "decode_expert_direct_bytes": int(fields[6]),
                                "decode_expert_uring_batches": int(fields[7]) if len(fields) == 9 else 0,
                                "decode_expert_uring_reads": int(fields[8]) if len(fields) == 9 else 0,
                            })
                            break
                elif kind == "PFPIPE" and len(fields) == 10:
                    request_id = fields[1]
                    for turn in reversed(self.profile):
                        if turn.get("request_id") == request_id:
                            turn.update({
                                "prefill_load_pipeline_active": bool(int(fields[2])),
                                "prefill_pipeline_batches": int(fields[3]),
                                "prefill_pipeline_experts": int(fields[4]),
                                "prefill_pipeline_bytes": int(fields[5]),
                                "prefill_pipeline_producer_load_s": float(fields[6]),
                                "prefill_pipeline_consumer_wait_s": float(fields[7]),
                                "prefill_pipeline_consumer_compute_s": float(fields[8]),
                                "prefill_pipeline_wall_s": float(fields[9]),
                            })
                            break
                elif kind == "PROF" and len(fields) >= 10:
                    # Legacy upstream spelling retained for drop-in compatibility.
                    self.profile.append({
                        "wall_s": float(fields[1]),
                        "prompt_tokens": int(fields[2]),
                        "completion_tokens": int(fields[3]),
                        "expert_disk_s": float(fields[4]),
                        "expert_wait_s": float(fields[5]),
                        "expert_matmul_s": float(fields[6]),
                        "attention_s": float(fields[7]),
                        "lm_head_s": float(fields[8]),
                        "forwards": int(fields[9]),
                    })
                    self.profile_seq += 1
                elif kind == "TIERS" and len(fields) >= 6:
                    self.tiers = {"vram": int(fields[1]), "ram": int(fields[2]),
                                  "disk": int(fields[3]), "vram_gb": float(fields[4]),
                                  "ram_gb": float(fields[5])}
                elif kind == "Q3NATIVE" and len(fields) == 5:
                    self.q3_native = {
                        "uploads": int(fields[1]),
                        "upload_bytes": int(fields[2]),
                        "gemv_calls": int(fields[3]),
                        "grouped_calls": int(fields[4]),
                    }
                elif kind == "Q3ATLAS" and len(fields) in (10, 12):
                    self.q3_atlas = {
                        "active": bool(int(fields[1])),
                        "refreshes": int(fields[2]),
                        "routes": int(fields[3]),
                        "host_entries": int(fields[4]),
                        "device_entries": int(fields[5]),
                        "device_capacity": int(fields[6]),
                        "loads": int(fields[7]),
                        "device_bytes": int(fields[8]),
                        "uncovered": int(fields[9]),
                        "prefill_batches": int(fields[10]) if len(fields) == 12 else 0,
                        "prefill_tokens": int(fields[11]) if len(fields) == 12 else 0,
                    }
                elif kind == "RESIDENT" and len(fields) == 9:
                    self.resident = {
                        "request_id": fields[1],
                        "layers": int(fields[2]),
                        "device_moe": int(fields[3]),
                        "host_moe": int(fields[4]),
                        "activation_h2d_bytes": int(fields[5]),
                        "activation_d2h_bytes": int(fields[6]),
                        "logits_d2h_bytes": int(fields[7]),
                        "router_d2h_bytes": int(fields[8]),
                    }
                elif kind == "ERROR" and len(fields) >= 2:
                    request_id = fields[1]
                    message = " ".join(fields[2:]) or "engine request failed"
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                        self.prefill_cached.pop(request_id, None)
                    if events is not None:
                        events.put(("error", _engine_error(fields[2:], message)))
                else:
                    # Telemetry is explicitly extensible. DATA/DONE/ERROR are the
                    # only request-control frames; unknown lines are advisory.
                    continue
        except Exception as error:
            if not self.closed:
                self.dispatcher_error = error
                self._fail_pending(error)

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, on_progress=None):
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        gpayload = grammar.encode("utf-8") if grammar else b""
        if b"\0" in gpayload:
            raise APIError(400, "NUL bytes are not supported in grammars.", "response_format")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        events = queue.Queue()
        with self.pending_lock:
            if self.closed:
                raise RuntimeError("shiftwing engine is shutting down")
            if self.dispatcher_error is not None:
                raise RuntimeError("shiftwing engine dispatcher stopped") from self.dispatcher_error
            if self.process.poll() is not None:
                raise RuntimeError("shiftwing engine is not running")
            request_id = str(self.next_request_id)
            self.next_request_id += 1
            self.pending[request_id] = events
        header = (f"SUBMIT {request_id} {cache_slot} {len(payload)} {max_tokens} "
                  f"{temperature:.8g} {top_p:.8g}"
                  + (f" {len(gpayload)}" if gpayload else "") + "\n").encode()
        request_started = time.monotonic()
        first_token_at = None
        prefill_elapsed_ms = None
        try:
            with self.write_lock:
                if self.process.poll() is not None:
                    raise RuntimeError("shiftwing engine is not running")
                self.process.stdin.write(header + payload + gpayload + b"\n")
                self.process.stdin.flush()
                if self.trace:
                    print(
                        f"[engine->] SUBMIT {request_id} slot={cache_slot} "
                        f"bytes={len(payload)} max={max_tokens}",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            raise

        cancel_sent = False

        def request_cancel():
            nonlocal cancel_sent
            if cancel_sent:
                return
            cancel_sent = True
            with self.write_lock:
                self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                self.process.stdin.flush()

        now = time.monotonic()
        first_output_deadline = now + self.first_model_output_timeout_ms / 1000.0
        progress_stall_deadline = now + self.prefill_first_progress_timeout_ms / 1000.0
        last_progress_marker = None
        progress_started = False
        while True:
            try:
                if first_output_deadline == float("inf"):
                    kind, value = events.get()
                else:
                    remaining = min(first_output_deadline, progress_stall_deadline) - time.monotonic()
                    kind, value = events.get(timeout=max(0.0, remaining))
            except queue.Empty:
                now = time.monotonic()
                stalled = progress_stall_deadline <= first_output_deadline and now >= progress_stall_deadline
                request_cancel()
                acknowledged = False;ack_deadline = time.monotonic() + 2.0
                while not acknowledged:
                    try:
                        ack_kind, _ = events.get(timeout=max(0.0, ack_deadline-time.monotonic()))
                        acknowledged = ack_kind == "error"
                    except queue.Empty:
                        break
                if not acknowledged and self.process.poll() is None:
                    self.process.terminate()
                raise FirstModelOutputTimeoutError(
                    (f"prefill made no progress for {(self.prefill_progress_stall_timeout_ms if progress_started else self.prefill_first_progress_timeout_ms)} ms"
                     if stalled else
                     f"no model output within {self.first_model_output_timeout_ms} ms"))
            if kind == "progress":
                if value.get("event") == "end":
                    prefill_elapsed_ms = value.get("elapsed_ms")
                if not cancel_sent:
                    marker = (value.get("prompt_tokens_cached"),
                              value.get("prompt_tokens_prefilled"))
                    advanced = (isinstance(marker[0], int) and isinstance(marker[1], int)
                                and marker[1] > marker[0])
                    if marker != last_progress_marker:
                        last_progress_marker = marker
                        if advanced:
                            progress_started = True
                            progress_stall_deadline = (time.monotonic() +
                                                       self.prefill_progress_stall_timeout_ms / 1000.0)
                    if on_progress:
                        on_progress(value)
                    if cancelled and cancelled():
                        request_cancel()
            elif kind == "data":
                if not cancel_sent:
                    if value:
                        first_output_deadline = float("inf")
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        progress_stall_deadline = float("inf")
                    decode(value)
                    if cancelled and cancelled():
                        request_cancel()
            elif kind == "done":
                finished = time.monotonic()
                tail = decoder.decode(b"", final=True)
                if tail:
                    if first_token_at is None:
                        first_token_at = finished
                    on_text(tail)
                first = first_token_at if first_token_at is not None else finished
                value["ttft_ms"] = (first - request_started) * 1000.0
                value["prefill_time_ms"] = (
                    float(prefill_elapsed_ms)
                    if prefill_elapsed_ms is not None
                    else value["ttft_ms"]
                )
                rate = value.get("tokens_per_second", 0.0)
                value["decode_time_ms"] = (
                    1000.0 * value["completion_tokens"] / rate
                    if rate and rate > 0
                    else max(0.0, (finished - first) * 1000.0)
                )
                value["wall_time_ms"] = (finished - request_started) * 1000.0
                return value
            elif cancel_sent and isinstance(value, RuntimeError) and str(value) == "CANCELLED":
                raise ClientCancelled()
            else:
                raise value

    def close(self):
        with self.pending_lock:
            if self.closed:
                return
            self.closed = True
        self._fail_pending(RuntimeError("shiftwing engine is shutting down"))
        if self.process.poll() is None:
            # EOF is the mux protocol's graceful-shutdown signal. Let the C
            # process drain, run its atexit handlers, and atomically persist
            # the learned expert atlas before falling back to a signal.
            with self.write_lock:
                if self.process.stdin is not None and not self.process.stdin.closed:
                    try:
                        self.process.stdin.close()
                    except BrokenPipeError:
                        pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:
                        # A process blocked in kernel I/O can take longer than
                        # five seconds to observe SIGKILL. Cleanup must not
                        # discard an already completed benchmark result.
                        self.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
        if self.dispatcher is not threading.current_thread():
            self.dispatcher.join(timeout=5)
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None and not pipe.closed:
                pipe.close()


def model_object(model_id, created):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "shiftwing"}


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, model_id, api_key=None, max_tokens=1024,
                 cors_origins=DEFAULT_CORS_ORIGINS, max_queue=8, queue_timeout=300,
                 kv_slots=1, model_path=None):
        super().__init__(address, APIHandler)
        self.engine = engine
        self.model_id = model_id
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.model_family = snapshot_model_family(self.model_path)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.scheduler = GenerationScheduler(max_queue, queue_timeout, kv_slots)
        self.kv_slots = kv_slots
        self.cors_origins = tuple(cors_origins)
        self.created = int(time.time())


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30   # per-request socket timeout: a slowloris client that dribbles its
                   # request line/body can't pin a worker thread (and a slot) forever
    server_version = "shiftwing"

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, body, request_id=None, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if request_id:
            self.send_header("x-request-id", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if not origin or ("*" not in self.server.cors_origins and origin not in self.server.cors_origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in self.server.cors_origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, anthropic-version")
        self.send_header("Access-Control-Expose-Headers",
                         "x-request-id, x-shiftwing-queue-wait-ms, x-colibri-queue-wait-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in self.server.cors_origins:
            self.send_header("Vary", "Origin")

    LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", ""}

    def _is_authed(self):
        """True if no key is configured, or a correct key was presented. Anthropic clients
        (Claude Code, the Anthropic SDKs) authenticate with `x-api-key`, not `Bearer` — both
        are accepted, and both are compared in constant time."""
        if not self.server.api_key:
            return True
        import hmac
        if hmac.compare_digest(self.headers.get("Authorization", ""),
                               f"Bearer {self.server.api_key}"):
            return True
        return hmac.compare_digest(self.headers.get("x-api-key", ""), self.server.api_key)

    def require_auth(self):
        if not self._is_authed():
            raise APIError(401, "Invalid or missing API key.", None, "invalid_api_key",
                           "authentication_error")

    def _check_host(self):
        """DNS-rebinding guard: a web page can resolve a hostname to 127.0.0.1 and
        drive this local server unless we pin the Host header to loopback / the bind
        address. Rejects requests whose Host is anything else. (#SEC-7)"""
        host = self.headers.get("Host", "")
        if host.startswith("["):
            name = host[1:].split("]", 1)[0]                       # [ipv6]:port
        elif host.count(":") == 1:
            name = host.rsplit(":", 1)[0]                          # host:port / ipv4:port
        else:
            name = host                                            # bare host / bracketless ipv6
        name = name.strip().lower()
        allowed = set(self.LOOPBACK_HOSTS)
        try:
            allowed.add(str(self.server.server_address[0]).strip("[]").lower())
        except Exception:
            pass
        if name not in allowed:
            raise APIError(403, "Host header not allowed.", None, "forbidden")

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid Content-Length header.")
        if length < 1 or length > MAX_BODY:
            raise APIError(400, f"Request body must be between 1 and {MAX_BODY} bytes.")
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "Request body must be a JSON object.")
        return body

    def check_model(self, body):
        model = body.get("model")
        if model != self.server.model_id:
            raise APIError(404, f"The model `{model}` does not exist.", "model", "model_not_found")

    _WEB_CANDIDATES = (
        Path(os.environ["COLI_WEB_DIST"]).expanduser()
        if os.environ.get("COLI_WEB_DIST")
        else None,
        Path(__file__).resolve().parent / "web" / "dist",
        Path(__file__).resolve().parent.parent / "web" / "dist",
    )
    WEB_DIST = next(
        (item for item in _WEB_CANDIDATES if item is not None and item.is_dir()),
        Path(__file__).resolve().parent.parent / "web" / "dist",
    )

    def serve_static(self, path):
        """Serve the built web UI (web/dist) so `coli web` is one process.
        Read-only, no auth (same trust level as /health), traversal-safe."""
        if path.startswith("/v1/") or path == "/health":
            return False
        base = self.WEB_DIST.resolve()
        if not base.is_dir():
            return False
        rel = unquote(path).lstrip("/") or "index.html"
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            target = None
        if target is None or not target.is_file():
            if path == "/" or "." not in rel:      # SPA fallback
                target = base / "index.html"
                if not target.is_file():
                    return False
            else:
                return False
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self._check_host()
            path = urlsplit(self.path).path
            if path == "/health":
                # Liveness is always public; hardware/scheduler internals only when a
                # request is authed (or no key set), so a configured key isn't leaked
                # past a bare 200 to an unauthenticated probe. (#SEC-8)
                payload = {"status": "ok"}
                if self._is_authed():
                    payload["scheduler"] = self.server.scheduler.snapshot()
                    payload["kv_slots"] = self.server.kv_slots
                    tiers = getattr(self.server.engine, "tiers", None) if self.server.engine else None
                    if tiers: payload["tiers"] = tiers
                    hwinfo = getattr(self.server.engine, "hwinfo", None) if self.server.engine else None
                    if hwinfo: payload["hwinfo"] = hwinfo
                self.send_json(200, payload, request_id)
                return
            if path == "/experts":
                payload = {"rows": 0, "cols": 0, "map": "", "hits": "", "seq": 0}
                eng = self.server.engine
                if self._is_authed() and eng and getattr(eng, "emap", None):   # (#SEC-8) hide routing telemetry unless authed
                    payload.update(eng.emap)
                    payload["hits"] = eng.hits or ""
                    payload["seq"] = eng.hits_seq
                self.send_json(200, payload, request_id)
                return
            if path == "/profile":
                eng = self.server.engine
                payload = {"seq": getattr(eng, "profile_seq", 0) if eng else 0,
                           "turns": list(getattr(eng, "profile", ()) or ()) if eng else [],
                           "resident": getattr(eng, "resident", None) if eng else None}
                self.send_json(200, payload, request_id)
                return
            if self.serve_static(path):
                return
            self.require_auth()
            if path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [model_object(
                    self.server.model_id, self.server.created)]}, request_id)
            elif path.startswith("/v1/models/") and unquote(path[11:]) == self.server.model_id:
                self.send_json(200, model_object(self.server.model_id, self.server.created), request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)

    def do_OPTIONS(self):
        try:                                   # (#SEC-7) apply the Host guard uniformly, incl. CORS preflight
            self._check_host()
        except APIError:
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self._check_host()
            self.require_auth()
            body = self.read_json()
            self.check_model(body)
            path = urlsplit(self.path).path
            if path == "/v1/chat/completions":
                self.chat_completion(body, request_id)
            elif path == "/v1/completions":
                self.completion(body, request_id)
            elif path == "/v1/messages":
                self.anthropic_messages(body, request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, self.error_body(error), request_id, error.headers)
        except FirstModelOutputTimeoutError as error:
            api_error = APIError(504, str(error), None,
                                 "first_model_output_timeout", "server_error")
            try:
                self.send_json(504, self.error_body(api_error), request_id)
            except OSError:
                pass
        except ClientCancelled:
            pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            api_error = APIError(500, "The shiftwing engine failed to process the request.",
                                 None, "engine_error", "server_error")
            try:
                self.send_json(500, self.error_body(api_error), request_id)
            except OSError:
                pass

    def error_body(self, error):
        """Anthropic clients parse a different error envelope; the OpenAI one is unchanged."""
        if urlsplit(self.path).path != "/v1/messages":
            return error_object(error)
        return {"type": "error", "error": {"type": error.error_type, "message": error.message}}

    def generation(self, body, prompt, request_id, chat, tools=None, tool_choice=None,
                   thinking=False):
        # COLI_DEBUG tees the engine transaction to stderr: 1 = decoded output stream only,
        # 2 = both sides (rendered prompt + output). render_chat already folds prior turns and
        # tool results into `prompt`, so level 2 is the full conversation the engine saw.
        try:
            dbg = int(os.environ.get("COLI_DEBUG", "0"))
        except ValueError:
            dbg = 0
        if dbg >= 2:
            sys.stderr.write(f"\n===== PROMPT [{request_id}] =====\n{prompt}\n===== OUTPUT [{request_id}] =====\n")
            sys.stderr.flush()
        maximum, temperature, top_p, grammar = generation_options(body, self.server.max_tokens)
        # tools and tool_choice come from chat_completion() already processed/filtered
        if chat and tool_choice == "none":
            tools = None          # client forbade tools: never surface tool_calls
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_ms = str(round(queue_wait * 1000))
            queue_headers = {
                "x-shiftwing-queue-wait-ms": queue_ms,
                "x-colibri-queue-wait-ms": queue_ms,
            }
            if not stream:
                output = []
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, output.append, cache_slot,
                    self.client_disconnected, grammar=grammar)
                text = "".join(output)
                length_finish = "length" if stats["length_limited"] else "stop"
                if chat:
                    reasoning, content, calls = parse_assistant_reply(
                        text, tools, assume_reasoning=thinking,
                        family=self.server.model_family
                    )
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    if calls:
                        message["tool_calls"] = calls
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                else:
                    choice = {"index": 0, "text": text, "logprobs": None,
                              "finish_reason": length_finish}
                self.send_json(200, {"id": completion_id, "object": object_name, "created": created,
                    "model": self.server.model_id, "choices": [choice], "usage": self.usage(stats),
                    "shiftwing_metrics": self.shiftwing_metrics(stats, queue_wait),
                    "colib_metrics": self.shiftwing_metrics(stats, queue_wait)},
                    request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items(): self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = True
            # KEEPALIVE: engine.generate() blocks SILENTLY during the (minutes-long) cold
            # prefill, and the client drops the socket after its idle timeout. A background pump
            # emits a reasoning_content "." delta (the channel that reliably resets the client's
            # timer and lands in the thinking panel, so answer content stays clean) whenever no
            # event has been written for KA_GAP seconds. All wfile writes share ka_lock so the
            # pump and event() never interleave; last_write gates the pump so it stays quiet
            # while real tokens are flowing (e.g. during decode).
            ka_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()
            KA_GAP = 10.0
            dbg_echo = dbg >= 1   # tee decoded tokens to stderr (COLI_DEBUG level parsed in generation())

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": self.server.model_id, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            last_progress = [None]

            def progress_event(progress):
                nonlocal connected
                last_progress[0] = dict(progress)
                if not connected:
                    return
                payload = {"object": "shiftwing.progress", "schema_version": 1,
                           "request_id": request_id, **progress}
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def _keepalive():
                ping = [{"index": 0, "delta": ({"reasoning_content": "."} if chat else {"content": ""}),
                         "logprobs": None, "finish_reason": None}]
                while not ka_stop.wait(1.0):
                    if not connected:
                        return
                    gap = time.time() - last_write[0]
                    if last_progress[0] is not None and gap >= 2.0:
                        progress_event(last_progress[0])
                    elif gap >= KA_GAP:
                        event(ping)

            if chat:
                event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "logprobs": None, "finish_reason": None}])

            def emit(text):
                last_progress[0] = None
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            def emit_reasoning(text):
                if text:
                    last_progress[0] = None
                    event([{"index": 0, "delta": {"reasoning_content": text},
                            "logprobs": None, "finish_reason": None}])

            ka_thread = threading.Thread(target=_keepalive, daemon=True)
            ka_thread.start()
            raw = []
            if chat and tools:
                # Suppress tool-call markers from the streamed content and parse the authoritative
                # calls from the FULL reply after generation. Hold back a marker-length tail so a
                # <tool_call> split across engine chunks is still caught.
                sp = {"buf": "", "tool": False}
                hold = len(BOX_START) - 1
                def emit_public(chunk):
                    if sp["tool"]:
                        return
                    sp["buf"] += chunk
                    cut = sp["buf"].find(BOX_START)
                    if cut >= 0:
                        if cut:
                            emit(sp["buf"][:cut])
                        sp["buf"] = ""
                        sp["tool"] = True
                        return
                    flush = max(0, len(sp["buf"]) - hold)
                    if flush:
                        emit(sp["buf"][:flush])
                        sp["buf"] = sp["buf"][flush:]
            else:
                def emit_public(chunk):
                    emit(chunk)

            # With thinking enabled, generation begins inside the opening
            # <think> supplied by the prompt. Route bytes before the closing tag
            # to reasoning_content while retaining a marker-sized suffix so a
            # tag split across tokens is recognized.
            reason = {"buf": "", "closed": not (chat and thinking)}
            reason_hold = len(THINK_CLOSE) - 1

            def emit_model(chunk):
                if tools:
                    raw.append(chunk)
                if dbg_echo:
                    sys.stderr.write(chunk)
                    sys.stderr.flush()
                if reason["closed"]:
                    emit_public(chunk)
                    return
                reason["buf"] += chunk
                cut = reason["buf"].find(THINK_CLOSE)
                if cut >= 0:
                    emit_reasoning(reason["buf"][:cut])
                    tail = reason["buf"][cut + len(THINK_CLOSE):]
                    reason["buf"] = ""
                    reason["closed"] = True
                    if tail:
                        emit_public(tail)
                    return
                flush = max(0, len(reason["buf"]) - reason_hold)
                if flush:
                    emit_reasoning(reason["buf"][:flush])
                    reason["buf"] = reason["buf"][flush:]

            try:
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit_model, cache_slot,
                    lambda: not connected, grammar=grammar, on_progress=progress_event)
            except FirstModelOutputTimeoutError as error:
                # The 200/SSE headers are already committed, so the outer HTTP error
                # handler cannot replace this response with a JSON 504. Surface the
                # terminal error in-band and close the stream cleanly instead.
                ka_stop.set()
                ka_thread.join(timeout=2)
                payload = error_object(APIError(
                    504, str(error), None, "first_model_output_timeout", "server_error"))
                if connected:
                    with ka_lock:
                        try:
                            data = json.dumps(payload, ensure_ascii=False,
                                              separators=(",", ":"))
                            self.wfile.write(f"data: {data}\n\n".encode())
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        except OSError:
                            pass
                self.close_connection = True
                return
            if not reason["closed"] and reason["buf"]:
                emit_reasoning(reason["buf"])
                reason["buf"] = ""

            calls = []
            if chat and tools:
                if not sp["tool"] and sp["buf"]:
                    emit(sp["buf"])                     # no tool call happened: flush held tail
                _reasoning, _content, calls = parse_assistant_reply(
                    "".join(raw), tools, assume_reasoning=thinking,
                    family=self.server.model_family
                )
                for i, tc in enumerate(calls):
                    event([{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"],
                             "type": "function", "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]}}]},
                            "logprobs": None, "finish_reason": None}])
            finish = ("tool_calls" if calls else
                      ("length" if stats["length_limited"] else "stop"))
            ka_stop.set()                          # generation done: stop the keepalive pump
            ka_thread.join(timeout=2)
            final_choice = ({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}
                            if chat else {"index": 0, "text": "", "logprobs": None,
                                          "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            metrics = self.shiftwing_metrics(stats, queue_wait)
            if connected:
                with ka_lock:
                    try:
                        data = json.dumps(metrics, ensure_ascii=False, separators=(",", ":"))
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                    except OSError:
                        connected = False
            if connected:
                with ka_lock:                          # (#B9) share the pump's lock so [DONE] can't interleave a keepalive write
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass
            self.close_connection = True

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    @staticmethod
    def shiftwing_metrics(stats, queue_wait=0.0):
        profile = stats.get("profile") or {}
        cpu_hits = int(profile.get("expert_cpu_hits", 0))
        gpu_hits = int(profile.get("expert_gpu_hits", 0))
        misses = int(profile.get("expert_misses", 0))
        return {
            "object": "shiftwing.metrics",
            "schema_version": 1,
            "ttft_ms": float(stats.get("ttft_ms", 0.0)) + queue_wait * 1000.0,
            "queue_wait_ms": queue_wait * 1000.0,
            "prefill_time_ms": float(stats.get("prefill_time_ms", 0.0)),
            "decode_time_ms": float(stats.get("decode_time_ms", 0.0)),
            "total_time_ms": float(stats.get("wall_time_ms", 0.0)) + queue_wait * 1000.0,
            "decode_tokens_per_second": float(stats.get("tokens_per_second", 0.0)),
            "prompt_tokens": int(stats.get("prompt_tokens", 0)),
            "completion_tokens": int(stats.get("completion_tokens", 0)),
            "cache_hits": cpu_hits + gpu_hits,
            "cache_misses": misses,
            "cache_hit_percent": float(stats.get("cache_hit_percent", 0.0)),
            "disk_bytes": int(profile.get("expert_read_bytes", 0)),
            "direct_io_bytes": int(profile.get("expert_direct_bytes", 0)),
        }

    def chat_completion(self, body, request_id):
        reasoning_effort = body.get("reasoning_effort")
        efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh")
        if reasoning_effort not in efforts:
            raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, or xhigh.",
                           "reasoning_effort")
        # COLI_THINK=1 makes thinking the default when the client sends NEITHER reasoning_effort
        # nor enable_thinking (a global switch, like the old server's --think). An explicit
        # client value always wins. Default off => exact OpenAI-standard behavior.
        if (reasoning_effort is None and "enable_thinking" not in body
                and os.environ.get("COLI_THINK", "0") == "1"):
            reasoning_effort = "high"
        enable_thinking = body.get("enable_thinking", reasoning_effort not in (None, "none"))
        if not isinstance(enable_thinking, bool):
            raise APIError(400, "`enable_thinking` must be a boolean.", "enable_thinking")
        tools = body.get("tools") or body.get("functions") or None
        tool_choice = body.get("tool_choice")
        prompt = render_model_chat(
            self.server.model_path,
            self.server.model_family,
            body.get("messages"),
            enable_thinking,
            reasoning_effort,
            tools,
            tool_choice,
            cache_prefix_compatible=True,
        )
        self.generation(body, prompt, request_id, True, tools, tool_choice,
                        enable_thinking)

    # ---- Anthropic /v1/messages (#343) ----------------------------------------------------
    ANTHROPIC_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}

    def anthropic_messages(self, body, request_id):
        for unsupported, why in (("stop_sequences", "custom stop sequences"),
                                 ("top_k", "top-k sampling")):
            if body.get(unsupported) not in (None, [], ""):
                raise APIError(400, f"Shiftwing does not support `{unsupported}` ({why}) yet.",
                               unsupported, "unsupported_value")
        messages = anthropic_to_openai(body)
        tools, tool_choice = anthropic_tools(body)
        thinking = body.get("thinking")
        if thinking is not None and not isinstance(thinking, dict):
            raise APIError(400, "`thinking` must be an object.", "thinking")
        enable_thinking = bool(thinking and thinking.get("type") == "enabled")
        if not enable_thinking and thinking is None and os.environ.get("COLI_THINK", "0") == "1":
            enable_thinking = True
        if body.get("max_tokens") is None:
            raise APIError(400, "`max_tokens` is required.", "max_tokens")
        # Reuse the OpenAI path's own validation by handing it an equivalent body.
        translated = {"messages": messages, "max_tokens": body.get("max_tokens"),
                      "temperature": body.get("temperature"), "top_p": body.get("top_p"),
                      "stream": body.get("stream", False), "cache_slot": body.get("cache_slot")}
        if tools:
            translated["tools"] = tools
        if tool_choice is not None:
            translated["tool_choice"] = tool_choice
        if tool_choice == "none":
            tools = None
        prompt = render_model_chat(
            self.server.model_path,
            self.server.model_family,
            messages,
            enable_thinking,
            "high" if enable_thinking else None,
            tools,
            tool_choice,
            cache_prefix_compatible=True,
        )
        self.anthropic_generation(translated, prompt, request_id, tools)

    def anthropic_generation(self, body, prompt, request_id, tools):
        maximum, temperature, top_p, grammar = generation_options(body, self.server.max_tokens)
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        message_id = "msg_" + uuid.uuid4().hex[:24]

        def blocks_and_stop(text, stats):
            """Split a finished reply into Anthropic content blocks + stop_reason."""
            calls = []
            if tools:
                text, calls = parse_model_tool_calls(
                    self.server.model_family, text, tools)
            content = []
            if text:
                content.append({"type": "text", "text": text})
            for call in calls:
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                content.append({"type": "tool_use", "id": call["id"],
                                "name": function["name"], "input": arguments})
            reason = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            return content, self.ANTHROPIC_STOP[reason]

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_ms = str(round(queue_wait * 1000))
            queue_headers = {
                "x-shiftwing-queue-wait-ms": queue_ms,
                "x-colibri-queue-wait-ms": queue_ms,
            }
            if not stream:
                output = []
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, output.append, cache_slot,
                    self.client_disconnected, grammar=grammar)
                content, stop_reason = blocks_and_stop("".join(output), stats)
                self.send_json(200, {
                    "id": message_id, "type": "message", "role": "assistant",
                    "model": self.server.model_id, "content": content,
                    "stop_reason": stop_reason, "stop_sequence": None,
                    "usage": {"input_tokens": stats["prompt_tokens"],
                              "output_tokens": stats["completion_tokens"]}},
                    request_id, queue_headers)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items():
                self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = [True]
            write_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()

            def send_event(name, payload):
                if not connected[0]:
                    return
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                with write_lock:
                    try:
                        self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected[0] = False

            def progress_event(progress):
                payload = {"object": "shiftwing.progress", "schema_version": 1,
                           "request_id": request_id, **progress}
                send_event("shiftwing.progress", payload)

            # Anthropic has a first-class keepalive event, so the cold prefill (minutes) does
            # not need the OpenAI path's reasoning-delta trick: `ping` is in the protocol.
            def keepalive():
                while not ka_stop.wait(1.0):
                    if not connected[0]:
                        return
                    if time.time() - last_write[0] >= 10.0:
                        send_event("ping", {"type": "ping"})

            send_event("message_start", {"type": "message_start", "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "model": self.server.model_id, "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
            send_event("content_block_start", {"type": "content_block_start", "index": 0,
                                               "content_block": {"type": "text", "text": ""}})
            ka_thread = threading.Thread(target=keepalive, daemon=True)
            ka_thread.start()

            raw = []
            state = {"buf": "", "in_tool": False}
            hold = len(BOX_START) - 1

            def on_text(chunk):
                raw.append(chunk)
                if not tools:
                    send_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": chunk}})
                    return
                if state["in_tool"]:
                    return                       # tool markers never reach the client as text
                state["buf"] += chunk
                cut = state["buf"].find(BOX_START)
                if cut >= 0:
                    if cut:
                        send_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": state["buf"][:cut]}})
                    state["buf"] = ""
                    state["in_tool"] = True
                    return
                flush = max(0, len(state["buf"]) - hold)
                if flush:
                    send_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": state["buf"][:flush]}})
                    state["buf"] = state["buf"][flush:]

            stats = self.server.engine.generate(
                prompt, maximum, temperature, top_p, on_text, cache_slot,
                lambda: not connected[0], grammar=grammar, on_progress=progress_event)
            if tools and not state["in_tool"] and state["buf"]:
                send_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": state["buf"]}})
            ka_stop.set()
            ka_thread.join(timeout=2)
            send_event("content_block_stop", {"type": "content_block_stop", "index": 0})

            content, stop_reason = blocks_and_stop("".join(raw), stats)
            index = 1
            for block in content:
                if block["type"] != "tool_use":
                    continue                     # text already streamed as block 0
                send_event("content_block_start", {"type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": block["id"],
                                      "name": block["name"], "input": {}}})
                send_event("content_block_delta", {"type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": json.dumps(block["input"], ensure_ascii=False)}})
                send_event("content_block_stop", {"type": "content_block_stop", "index": index})
                index += 1
            send_event("message_delta", {"type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": stats["completion_tokens"]}})
            send_event("message_stop", {"type": "message_stop"})
            self.close_connection = True

    def completion(self, body, request_id):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "Shiftwing currently requires `prompt` to be a string.", "prompt")
        if not prompt:
            raise APIError(400, "`prompt` must not be empty.", "prompt")
        self.generation(body, prompt, request_id, False)


def serve(model, host="127.0.0.1", port=8000, model_id="qwen3.5-shiftwing", api_key=None,
          cap=8, max_tokens=1024, engine=None, env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1):
    if engine is None:
        engine = default_engine()
    if not 1 <= max_tokens:
        raise ValueError("max_tokens must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_queue < 0:
        raise ValueError("max_queue cannot be negative")
    if queue_timeout <= 0:
        raise ValueError("queue_timeout must be positive")
    if not 1 <= kv_slots <= 16:
        raise ValueError("kv_slots must be between 1 and 16")
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        # (#SEC-6) Fail closed: an unauthenticated engine on a non-loopback bind exposes
        # a compute-heavy API to the network. Refuse unless explicitly overridden.
        if os.environ.get("COLI_ALLOW_INSECURE_BIND") == "1":
            print("WARNING: binding %s beyond localhost with NO auth (COLI_ALLOW_INSECURE_BIND=1)" % host,
                  file=sys.stderr)
        else:
            print("refusing to bind %s beyond localhost without COLI_API_KEY set "
                  "(set COLI_ALLOW_INSECURE_BIND=1 to override)" % host, file=sys.stderr)
            sys.exit(1)
    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    # Bind before starting the 744B engine. A stale/occupied port must fail in
    # milliseconds rather than loading hundreds of GB and leaking a child.
    server = APIServer((host, port), None, model_id, api_key, max_tokens, origins,
                       max_queue, queue_timeout, kv_slots, model)
    runtime = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        runtime = Engine(engine,model,cap,max_tokens,env,kv_slots)
        server.engine = runtime
        print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COLI_MODEL"), required=not os.environ.get("COLI_MODEL"))
    parser.add_argument("--engine", default=str(default_engine()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=os.environ.get("SHIFTWING_MODEL_ID", os.environ.get("COLI_MODEL_ID", "qwen3.5-shiftwing")))
    parser.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    parser.add_argument("--cors-origin", action="append", default=None,
                        help="allowed browser origin; repeat as needed (use '*' for any origin)")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queue", type=int, default=int(os.environ.get("COLI_MAX_QUEUE", "8")))
    parser.add_argument("--queue-timeout", type=float,
                        default=float(os.environ.get("COLI_QUEUE_TIMEOUT", "300")))
    parser.add_argument("--kv-slots", type=int, default=int(os.environ.get("COLI_KV_SLOTS", "1")))
    args = parser.parse_args()
    serve(args.model, args.host, args.port, args.model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout,kv_slots=args.kv_slots)


if __name__ == "__main__":
    main()
