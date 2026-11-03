#!/usr/bin/env python3
"""
LLM agent arm (llm_agents.py) — JURI Lab

Drop-in replacements for the four agent roles, backed by a language model through the Anthropic SDK.
The deterministic grader, governance game, packets and ground truth are unchanged: swapping a role
changes who produces an answer, not how it is scored. Information sets are enforced by what each
prompt contains, exactly as they are for the rule-based agents:

  compiler   sees the public handbook (turn 1) and labelled Tier 0-2 archive cases with its own
             assertion failures (turns 2 and 3). It writes `execute_compiled_routine(doc_intake, doc_sched,
             doc_annex)`. It never sees the solver or the registry.
  novice     sees the handbook and the three packet documents. No archive, no registry.
  incumbent  sees the full rulebook (public rules, the Tier 2 convention, the private registry and
             Tier 3 rules, the silent-collision gate) and the three documents; in collaborative mode it
             also sees the tool's answer and decides whether to accept or override it.
  critic     sees the annex and the tool's answer and returns an epistemic uncertainty in [0, 1].

Providers:
  AnthropicProvider  real calls (default model claude-opus-5-5, adaptive thinking, effort "high",
                     server-side refusal fallbacks, JSON-schema structured outputs for the answer roles).
                     Every call is appended to a JSONL transcript keyed by a hash of the prompt.
  ReplayProvider     serves responses from a transcript, so an LLM run can be re-executed exactly by
                     anyone holding the transcript file (the reproducibility mechanism for this arm).
  MockProvider       routes each role to the rule-based agent, so the protocol can be tested without
                     a key; with it, run_llm_arm.py must reproduce the registered-design numbers.

  StressProvider     a deterministic adversary that returns malformed JSON, refusals, missing fields,
                     syntax errors, infinite loops, exceptions, non-dict returns and file reads, mixed
                     with valid answers, to test that the harness degrades gracefully.

Model-written code is executed through SandboxedTool: a separate `python -I` process per task and
generation, with CPU-time and address-space limits and a per-case alarm, communicating by JSON files.
Resource limits contain crashes, hangs and excessive allocation. A self-tested macOS OS sandbox also restricts network and user-file access; on other platforms use a disposable container or VM for untrusted generated code.
"""
from __future__ import annotations


import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from model_config import CFG
from domain_cases import (
    ONETTaskSpec,
    MultiDocWorkPacket,
    PRECEDENT_REGISTRY,
    RULEBOOKS,
)
from negotiated_order_lab import (
    execute_unassisted_novice,
    execute_hybrid_tool_plus_critic,
    execute_incumbent_react_agent,
)
from compiler_agent import synthesize_compiled_tool_from_handbook, compute_ast_and_halstead_metrics

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "description": "Two bands joined by '__', e.g. HIGH_EXPOSURE__THRESHOLD_BREACH"},
        "inference": {"type": "string", "description": "The inference label of the rule that governs this case"},
        "treatment": {"type": "number", "description": "The treatment value, rounded to 2 decimals"},
        "rationale": {"type": "string", "description": "One or two sentences on which rule applied and why"},
    },
    "required": ["diagnosis", "inference", "treatment", "rationale"],
    "additionalProperties": False,
}

COLLAB_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["ACCEPT_TOOL", "OVERRIDE"]},
        "diagnosis": {"type": "string"},
        "inference": {"type": "string"},
        "treatment": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "diagnosis", "inference", "treatment", "rationale"],
    "additionalProperties": False,
}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "epistemic_uncertainty": {"type": "number", "description": "Probability in [0,1] that the tool's answer is wrong"},
        "critic_status": {"type": "string", "description": "Short label for what was noticed"},
        "rationale": {"type": "string"},
    },
    "required": ["epistemic_uncertainty", "critic_status", "rationale"],
    "additionalProperties": False,
}


@dataclass
class LLMConfig:
    model: str = "claude-opus-5-5"
    effort: str = "high"               # answer roles (novice, incumbent, critic)
    compiler_effort: str = "medium"    # code-writing turns: at "high" the repair turns spent 48k tokens thinking and returned no code
    max_tokens: int = 16000            # answer roles
    compiler_max_tokens: int = 64000   # code-writing turns: hard ceiling, streamed
    compiler_task_budget: int = 24000  # advisory token budget the compiler sees (beta task budgets; 0 disables; minimum 20000)
    transcript_path: Optional[str] = None
    code_timeout_s: int = 3
    archive_examples: int = 12     # labelled cases shown to the compiler per repair turn
    failure_examples: int = 8      # failing cases shown to the compiler per repair turn
    agentic_turns: int = 8         # tool-use turns allowed to the agentic compiler (test_module / show_cases / list_archive)
    run_id: str = ""               # identifies a live run; new transcripts and manifests live under data_outputs/llm_runs/<run_id>/
    code_isolation: str = "auto"   # generated code: "auto" = OS sandbox when available (macOS sandbox-exec), "required" = refuse without it, "none" = resource limits only


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
def _key(role: str, system: str, user: str, schema: Optional[dict]) -> str:
    h = hashlib.sha256()
    h.update(role.encode()); h.update(b"\x00"); h.update(system.encode()); h.update(b"\x00"); h.update(user.encode())
    h.update(b"\x00"); h.update(json.dumps(schema or {}, sort_keys=True).encode())
    return h.hexdigest()


KEY_VERSION = 2   # v2 transcript keys hash the prompt AND the complete request configuration


def build_request_kwargs(cfg: "LLMConfig", system: str, user: str, schema: Optional[dict], role: str = "") -> dict:
    """The complete Messages API request for one call. Used by AnthropicProvider to send the request and by the
    transcript key, so that any change to model, effort, token budgets, task budget, thinking mode, output format,
    beta flags or fallback policy changes the cache identity."""
    compiler = role.startswith("compiler")
    output_config: Dict[str, object] = {"effort": cfg.compiler_effort if compiler else cfg.effort}
    betas = ["server-side-fallback-2026-07-01"]
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if compiler and cfg.compiler_task_budget:
        # the model sees a countdown and wraps up instead of being cut off mid-thought by max_tokens
        output_config["task_budget"] = {"type": "tokens", "total": max(20000, int(cfg.compiler_task_budget))}
        betas.append("task-budgets-2026-03-13")
    return dict(
        model=cfg.model,
        max_tokens=cfg.compiler_max_tokens if compiler else cfg.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config=output_config,
        betas=betas,
        fallbacks="default",
    )


def request_fingerprint(cfg: "LLMConfig", role: str, schema: Optional[dict], tools: Optional[List[dict]] = None) -> dict:
    """Every request setting other than the prompt text: the configuration half of the v2 cache identity."""
    kw = build_request_kwargs(cfg, "", "", schema, role)
    kw.pop("system"); kw.pop("messages")
    if tools is not None:
        kw["tools"] = tools
    kw["key_version"] = KEY_VERSION
    return kw


def _key_v2(role: str, system: str, user: str, schema: Optional[dict], fingerprint: dict) -> str:
    h = hashlib.sha256()
    h.update(_key(role, system, user, schema).encode()); h.update(b"\x00")
    h.update(json.dumps(fingerprint, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _record_legacy_key(rec: dict) -> str:
    """The prompt-only key of a transcript record: records written before KEY_VERSION 2 store it as `key`."""
    return rec.get("legacy_key") or rec["key"] if rec.get("key_version") == KEY_VERSION else rec["key"]


class Provider:
    is_mock = False

    def complete(self, role: str, system: str, user: str, schema: Optional[dict] = None) -> str:
        raise NotImplementedError

    def tool_loop(self, role: str, system: str, user: str, tools: List[dict], handlers: Dict[str, Callable[[dict], str]],
                  max_turns: int) -> Tuple[str, int]:
        """Runs a tool-use conversation: the model may call any of `tools` (executed by `handlers`) up to `max_turns` times
        and then answers in text. Returns (final_text, turns_used). Providers without a model raise."""
        raise NotImplementedError


class MockProvider(Provider):
    """Protocol test double: the adapters route to the rule-based agents when they see `is_mock`."""
    is_mock = True

    def complete(self, role, system, user, schema=None):
        raise RuntimeError("MockProvider is handled by the adapters; complete() should not be called")


class StressProvider(Provider):
    """Deterministic adversary for harness testing. Cycles through malformed and hostile responses for
    each role, interleaved with valid ones. The valid compiler answer is the rule-based compiler's own
    emitted v1 program for the handbook in the prompt, so correct code is also exercised."""
    CODE_CASES = [
        "def execute_compiled_routine(doc_intake, doc_sched, doc_annex):\n    return {'diagnosis': 'X', 'inference': 'Y', 'treatment': 1.0\n",  # syntax error
        "def execute_compiled_routine(doc_intake, doc_sched, doc_annex):\n    while True:\n        pass\n",                                       # hang
        "def execute_compiled_routine(doc_intake, doc_sched, doc_annex):\n    raise ValueError('boom')\n",                                        # exception
        "def execute_compiled_routine(doc_intake, doc_sched, doc_annex):\n    return 42\n",                                                        # non-dict
        "def execute_compiled_routine(doc_intake, doc_sched, doc_annex):\n    open('/etc/hostname').read()\n    return {'diagnosis': 'A', 'inference': 'B', 'treatment': 0.0}\n",  # file read
        "import os\nos._exit(3)\n",                                                                                                              # kills the sandbox process at import
        "x = [0] * (10 ** 9)\n",                                                                                                                  # memory
        "VALID",
    ]
    JSON_CASES = [
        "not json at all",
        '{"diagnosis": "HIGH_EXPOSURE__THRESHOLD_BREACH"}',
        '{"error": "refusal", "category": "general_harms"}',
        '{"diagnosis": "A__B", "inference": "C", "treatment": "twelve", "rationale": ""}',
        '{"diagnosis": "A__B", "inference": "C", "treatment": 1e308, "rationale": ""}',
        "VALID",
    ]

    is_mock = False

    def __init__(self):
        self.n = 0
        self.valid_fraction_log: Dict[str, int] = {}

    def complete(self, role, system, user, schema=None):
        self.n += 1
        if role.startswith("compiler"):
            case = self.CODE_CASES[self.n % len(self.CODE_CASES)]
            if case == "VALID":
                from compiler_agent import parse_handbook_spec, emit_source
                m = re.search(r"PUBLIC HANDBOOK\n(.*?)\n\n(?:Write the module|LABELLED PAST CASES)", user, re.S)
                ir = parse_handbook_spec(m.group(1) if m else user)
                return "```python\n" + emit_source(ir, "stress", "stress", "stress", "v1") + "```"
            return "```python\n" + case + "```"
        case = self.JSON_CASES[self.n % len(self.JSON_CASES)]
        if case == "VALID":
            if role == "critic":
                return json.dumps({"epistemic_uncertainty": 0.5, "critic_status": "STRESS_VALID", "rationale": ""})
            if role == "incumbent_collab":
                return json.dumps({"decision": "ACCEPT_TOOL", "diagnosis": "HIGH_EXPOSURE__THRESHOLD_BREACH", "inference": "X", "treatment": 0.0, "rationale": ""})
            return json.dumps({"diagnosis": "HIGH_EXPOSURE__THRESHOLD_BREACH", "inference": "X", "treatment": 0.0, "rationale": ""})
        return case


class ReplayProvider(Provider):
    """Serves responses from a recorded transcript without any model request.

    mode="exact": every request must match a KEY_VERSION 2 record whose key hashes the prompt and the complete
    request configuration of `cfg`; tool loops are rebuilt turn by turn from the recorded request messages and
    assistant content, and each turn's key is verified. This verifies that the replayed run used the same requests.

    mode="legacy": for transcripts written before KEY_VERSION 2 (the shipped 2026 live run). Records are matched on
    the prompt-only key, so the request configuration (model, effort, budgets, fallback policy) is NOT verified, and
    tool loops are replayed as recorded events: the first turn is matched exactly on role, system prompt, user prompt
    and tool definitions, later turns are taken in file order, and every recorded tool call is re-executed through the
    live handlers so that the candidate modules the model submitted are actually run. The intermediate request
    histories were not recorded, so exact request verification of later turns is impossible in this mode."""

    def __init__(self, transcript_path: str, mode: str = "legacy", cfg: Optional["LLMConfig"] = None):
        if mode not in ("legacy", "exact"):
            raise ValueError(f"unknown replay mode {mode!r}")
        if mode == "exact" and cfg is None:
            raise ValueError("exact replay needs the LLMConfig of the recorded run")
        self.mode, self.cfg = mode, cfg
        self.store: Dict[str, str] = {}
        self.records: List[dict] = []
        self.tool_events: List[dict] = []
        self.prompt_mismatches: List[dict] = []   # legacy tool loops matched on the user prompt only
        self.ambiguous_starts: List[dict] = []    # legacy tool loops recorded more than once for the same prompt
        self.legacy_occurrence = 0                # which recorded conversation to replay when several share a prompt
        self.requested_keys: List[str] = []       # stored keys served by complete(), in request order (F13 selection)
        with open(transcript_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.records.append(rec)
                if mode == "exact":
                    if rec.get("key_version") == KEY_VERSION:
                        self.store[rec["key"]] = rec["response_text"]
                else:
                    self.store[_record_legacy_key(rec)] = rec["response_text"]
        if mode == "exact" and not self.store:
            raise ValueError(f"{transcript_path} has no KEY_VERSION {KEY_VERSION} records; it can only be replayed with mode='legacy'")
        self.provenance = ("exact replay: prompt and request configuration verified" if mode == "exact" else
                           "LEGACY replay: prompts matched, request configuration not verified; tool loops replayed as recorded events")

    def complete(self, role, system, user, schema=None):
        k = (_key_v2(role, system, user, schema, request_fingerprint(self.cfg, role, schema)) if self.mode == "exact"
             else _key(role, system, user, schema))
        if k not in self.store:
            raise KeyError(f"no transcript entry for role={role} key={k[:12]} ({self.mode} replay); the prompt"
                           f"{' or request configuration' if self.mode == 'exact' else ''} differs from the recorded run")
        self.requested_keys.append(k)
        return self.store[k]

    def _run_tools(self, calls: List[dict], handlers: Dict[str, Callable[[dict], str]], turn: int) -> List[str]:
        outs = []
        for c in calls:
            try:
                inp = c.get("input") if isinstance(c.get("input"), dict) else json.loads(c.get("input") or "{}")
                out = handlers[c["name"]](inp)
            except Exception as e:  # a failing tool is reported, never raised, exactly as in the live loop
                out = json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"})
            outs.append(out)
            self.tool_events.append({"turn": turn, "tool": c.get("name"), "result": out[:2000]})
        return outs

    def tool_loop(self, role, system, user, tools, handlers, max_turns):
        if self.mode == "exact":
            return self._tool_loop_exact(role, system, user, tools, handlers, max_turns)
        return self._tool_loop_legacy(role, system, user, tools, handlers, max_turns)

    def _tool_loop_exact(self, role, system, user, tools, handlers, max_turns):
        by_key = {r["key"]: r for r in self.records if r.get("key_version") == KEY_VERSION}
        messages: List[dict] = [{"role": "user", "content": user}]
        turns = 0
        while True:
            k = _key_v2(f"{role}_turn{turns}", system, json.dumps(messages, sort_keys=True, default=str), {"tools": tools},
                        request_fingerprint(self.cfg, role, None, tools))
            rec = by_key.get(k)
            if rec is None:
                raise KeyError(f"exact replay: no record for {role}_turn{turns}; the conversation diverged from the recorded run")
            calls = rec.get("tool_calls") or []
            if rec.get("stop_reason") == "refusal":
                return json.dumps({"error": "refusal"}), turns
            if rec.get("stop_reason") != "tool_use" or not calls or turns >= max_turns:
                return rec["response_text"], turns
            messages.append({"role": "assistant", "content": rec["assistant_content"]})
            outs = self._run_tools(calls, handlers, turns)
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": c["id"], "content": o}
                                                         for c, o in zip(calls, outs)]})
            turns += 1

    def _tool_loop_legacy(self, role, system, user, tools, handlers, max_turns):
        k0 = _key(f"{role}_turn0", system, json.dumps([{"role": "user", "content": user}], sort_keys=True, default=str), {"tools": tools})
        starts = [i for i, r in enumerate(self.records) if _record_legacy_key(r) == k0]
        if not starts:
            # the shipped agentic transcript was recorded under an earlier system prompt: match the user prompt only
            # and say so; the replay then follows the recorded system prompt, not the current one
            starts = [i for i, r in enumerate(self.records) if r.get("role") == f"{role}_turn0" and r.get("user") == user]
            if not starts:
                raise KeyError(f"legacy replay: no recorded {role}_turn0 for this prompt")
            self.prompt_mismatches.append({"role": role, "detail": "recorded system prompt or tool definitions differ from the current code"})
        if len(starts) > 1:
            self.ambiguous_starts.append({"role": role, "n_recorded_conversations": len(starts), "used": self.legacy_occurrence})
        i = starts[min(self.legacy_occurrence, len(starts) - 1)]
        rec_system = self.records[i].get("system")
        turns = 0
        while True:
            rec = self.records[i]
            if rec.get("role") != f"{role}_turn{turns}" or rec.get("system") != rec_system:
                raise KeyError(f"legacy replay: record {i} is {rec.get('role')}, expected {role}_turn{turns}; the event sequence is broken")
            calls = rec.get("tool_calls") or []
            if rec.get("stop_reason") == "refusal":
                return json.dumps({"error": "refusal"}), turns
            if rec.get("stop_reason") != "tool_use" or not calls or turns >= max_turns:
                return rec["response_text"], turns
            self._run_tools(calls, handlers, turns)
            turns += 1
            i += 1
            if i >= len(self.records):
                raise KeyError("legacy replay: transcript ended inside a tool loop")


class AnthropicProvider(Provider):
    def __init__(self, cfg: LLMConfig):
        import anthropic  # official SDK; credentials resolve from the environment or `ant auth login`
        self.cfg = cfg
        self.client = anthropic.Anthropic(max_retries=6)   # SDK backoff on 429/5xx/connection errors
        self.calls = 0
        self.errors = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        if cfg.transcript_path and os.path.exists(cfg.transcript_path):
            with open(cfg.transcript_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("key_version") != KEY_VERSION:
                        # never reuse or append to a transcript whose keys do not identify the request configuration
                        raise RuntimeError(f"{cfg.transcript_path} holds pre-v{KEY_VERSION} records; write new runs to a new transcript path")
                    self._cache[rec["key"]] = rec["response_text"]

    def request_kwargs(self, system: str, user: str, schema: Optional[dict], role: str = "") -> dict:
        return build_request_kwargs(self.cfg, system, user, schema, role)

    def tool_loop(self, role, system, user, tools, handlers, max_turns):
        """Manual agent loop over the beta Messages API. Every turn is recorded in the transcript under a key that hashes
        the full message history, so ReplayProvider can replay the loop turn by turn. Assistant content blocks (thinking
        included) are appended unchanged, as preserved thinking requires."""
        import anthropic
        messages: List[dict] = [{"role": "user", "content": user}]
        turns = 0
        final_text = ""
        while True:
            msg_json = json.dumps(messages, sort_keys=True, default=str)
            legacy_k = _key(f"{role}_turn{turns}", system, msg_json, {"tools": tools})
            fp = request_fingerprint(self.cfg, role, None, tools)
            k = _key_v2(f"{role}_turn{turns}", system, msg_json, {"tools": tools}, fp)
            kwargs = self.request_kwargs(system, user, None, role)
            kwargs["messages"] = messages
            kwargs["tools"] = tools
            t0 = time.time()
            try:
                with self.client.beta.messages.stream(**kwargs) as stream:
                    resp = stream.get_final_message()
            except anthropic.BadRequestError as e:
                raise RuntimeError(f"bad request (fix the request shape): {e.message}") from e
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                with self._lock:
                    self.errors += 1
                return json.dumps({"error": "api_error", "detail": str(getattr(e, "message", e))[:200]}), turns
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
            calls = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            rec = {"key": k, "role": f"{role}_turn{turns}", "model": resp.model, "requested_model": self.cfg.model,
                   "fallback": [f"{getattr(getattr(b, 'from_', None), 'model', '?')}->{getattr(getattr(b, 'to', None), 'model', '?')}"
                                for b in resp.content if getattr(b, "type", "") == "fallback"],
                   "stop_reason": resp.stop_reason, "seconds": round(time.time() - t0, 1),
                   "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
                   "request_id": getattr(resp, "_request_id", None), "system": system, "user": user if turns == 0 else "",
                   "tool_calls": [{"id": b.id, "name": b.name, "input": b.input} for b in calls], "response_text": text,
                   "key_version": KEY_VERSION, "legacy_key": legacy_k, "config": fp, "run_id": self.cfg.run_id,
                   "request_messages": json.loads(msg_json), "assistant_content": [b.model_dump() for b in resp.content]}
            with self._lock:
                self.calls += 1
                self.usage["input_tokens"] += resp.usage.input_tokens
                self.usage["output_tokens"] += resp.usage.output_tokens
                if self.cfg.transcript_path:
                    with open(self.cfg.transcript_path, "a") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
            if resp.stop_reason == "refusal":
                return json.dumps({"error": "refusal"}), turns
            if resp.stop_reason != "tool_use" or not calls or turns >= max_turns:
                final_text = text
                break
            messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
            results = []
            for b in calls:
                try:
                    out = handlers[b.name](b.input if isinstance(b.input, dict) else json.loads(b.input))
                except Exception as e:  # a failing tool is reported to the model, never raised
                    out = json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
            turns += 1
        return final_text, turns

    def complete(self, role, system, user, schema=None):
        import anthropic
        fp = request_fingerprint(self.cfg, role, schema)
        legacy_k = _key(role, system, user, schema)
        k = _key_v2(role, system, user, schema, fp)
        with self._lock:
            if k in self._cache:
                return self._cache[k]
        kwargs = self.request_kwargs(system, user, schema, role)
        t0 = time.time()
        try:
            # streamed so that large max_tokens on code-writing turns do not hit HTTP timeouts
            with self.client.beta.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
        except anthropic.BadRequestError as e:
            raise RuntimeError(f"bad request (fix the request shape): {e.message}") from e
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # after the SDK's own retries: record the failure and let the grader count it as wrong
            with self._lock:
                self.errors += 1
            return json.dumps({"error": "api_error", "detail": str(getattr(e, "message", e))[:200]})
        if resp.stop_reason == "refusal":
            text = json.dumps({"error": "refusal", "category": getattr(getattr(resp, "stop_details", None), "category", None)})
        else:
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
            if resp.stop_reason == "max_tokens" and not text.strip():
                text = json.dumps({"error": "max_tokens", "detail": "output budget exhausted before any text was produced"})
        # server-side fallback: `resp.model` is the model that served the call; a `fallback` block names the decline
        fallbacks = [f"{getattr(getattr(b, 'from_', None), 'model', '?')}->{getattr(getattr(b, 'to', None), 'model', '?')}"
                     for b in resp.content if getattr(b, "type", "") == "fallback"]
        rec = {"key": k, "role": role, "model": resp.model, "requested_model": self.cfg.model, "fallback": fallbacks,
               "stop_reason": resp.stop_reason, "seconds": round(time.time() - t0, 1),
               "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
               "request_id": getattr(resp, "_request_id", None), "system": system, "user": user, "response_text": text,
               "key_version": KEY_VERSION, "legacy_key": legacy_k, "config": fp, "run_id": self.cfg.run_id}
        with self._lock:
            self.calls += 1
            self.usage["input_tokens"] += resp.usage.input_tokens
            self.usage["output_tokens"] += resp.usage.output_tokens
            if self.cfg.transcript_path:
                with open(self.cfg.transcript_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            self._cache[k] = text
        return text


# ---------------------------------------------------------------------------
# Prompt material
# ---------------------------------------------------------------------------
def render_full_rulebook(spec: ONETTaskSpec) -> str:
    """The incumbent's information set: public rules plus the Tier 2 convention and the private registry."""
    rb = RULEBOOKS[spec.domain_family]
    reg = PRECEDENT_REGISTRY[spec.domain_family]
    L = [f"FULL RULEBOOK FOR {spec.domain_family} (incumbent professional knowledge)",
         f"Diagnostic cutoffs: PRIMARY_CUTOFF={spec.primary_cutoff:.2f}, RATIO_CUTOFF={spec.ratio_cutoff:.4f}",
         "Document fields: V1 = DOC1.V1_PRIMARY_MAGNITUDE, V2 = DOC1.V2_THRESHOLD_RATIO, V3 = DOC1.V3_RESERVE_ADJUSTMENT, "
         "V4 = DOC2.V4_SCHEDULE_PARAMETER, ANNEX_SEVERITY = DOC3.ANNEX_SEVERITY, SEV = the annex's severity-type number for the case",
         "Definitions:"]
    for n, ex in rb.lets:
        L.append(f"  LET {n} := {ex}")
    L.append("Diagnosis: HIGH_EXPOSURE if V1 >= PRIMARY_CUTOFF else MODERATE_EXPOSURE; THRESHOLD_BREACH if V2 >= RATIO_CUTOFF else WITHIN_TOLERANCE; join with '__'.")
    L.append("Which rule governs (check in this order):")
    L.append(f"  Tier 1 — annex states EXPLICIT_TIER1_OVERRIDE = TRUE with ANNEX_SEVERITY: INFER {rb.tier1_rule[0]}; TREAT {rb.tier1_rule[1]}")
    L.append(f"  Tier 2 — annex carries severity_index=<SEV> (semi-structured protocol exception): INFER {rb.tier2_rule[0]}; TREAT {rb.tier2_rule[1]}")
    L.append(f"  Tier 3 (overt) — annex is a PRECEDENT MEMO with base_haircut=<SEV>: apply the registry rule below.")
    L.append(f"  Tier 3 (silent) — annex looks routine but 'Archive Reference Metric' >= {reg['silent_metric_floor']} AND V2 > {reg['silent_ratio_gate']}: "
             f"treat the metric as SEV and apply the registry rule below. Routine archive metrics are below the floor.")
    L.append(f"  Routine, V2 >= RATIO_CUTOFF: INFER {rb.breach_rule[0]}; TREAT {rb.breach_rule[1]}")
    L.append(f"  Routine, otherwise: INFER {rb.base_rule[0]}; TREAT {rb.base_rule[1]}")
    L.append("Private precedent registry (REG.*):")
    for k, v in reg.items():
        L.append(f"  REG.{k} = {v}")
    for n, ex in rb.tier3_lets:
        L.append(f"  LET {n} := {ex}")
    L.append(f"  Tier 3 rule: INFER {rb.tier3_rule[0]}; TREAT {rb.tier3_rule[1]}")
    L.append("Arithmetic: MIN/MAX are min/max; 'a IF c ELSE b' is a conditional; round the treatment to 2 decimals.")
    return "\n".join(L)


def _docs_block(pkt: MultiDocWorkPacket) -> str:
    return (f"--- DOCUMENT 1 ---\n{pkt.doc_primary_intake}\n--- DOCUMENT 2 ---\n{pkt.doc_secondary_schedule}\n"
            f"--- DOCUMENT 3 ---\n{pkt.doc_tertiary_annex}\n")


def _parse_json(text: str) -> dict:
    """Parses a model response into a JSON object. Always returns a dict: a non-object (array, number, string, null),
    unparseable text or a non-finite number yields {"error": <class>, ...} so that every role grades it as an error."""
    if not isinstance(text, str):
        return {"error": "parse", "raw": repr(text)[:200]}

    def _reject_constant(c):  # NaN, Infinity and -Infinity are not JSON; treat them as non-finite answers
        raise ValueError(f"nonfinite:{c}")
    try:
        d = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:
        if str(e).startswith("nonfinite"):
            return {"error": "nonfinite", "raw": text[:500]}
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"error": "parse", "raw": text[:500]}
        try:
            d = json.loads(m.group(0), parse_constant=_reject_constant)
        except ValueError as e2:
            return {"error": "nonfinite" if str(e2).startswith("nonfinite") else "parse", "raw": text[:500]}
    if not isinstance(d, dict):
        return {"error": "schema", "detail": f"top-level {type(d).__name__}, expected an object", "raw": text[:500]}
    return d


def _finite_number(x) -> Optional[float]:
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _answer_dict(d: dict, fallback_label: str) -> Dict[str, object]:
    """Validates an answer-role object (novice, incumbent, incumbent_collab). A response that is an error, lacks a
    required field, carries a non-string label or a non-finite / non-numeric treatment is returned with an
    `error_class` and a sentinel label, so it is graded wrong; it is never silently coerced into a plausible answer."""
    if not isinstance(d, dict):
        d = {"error": "schema"}
    err = d.get("error")
    if err:
        cls = str(err) if str(err) in ("parse", "refusal", "api_error", "max_tokens", "nonfinite", "schema") else "parse"
        return {"diagnosis": f"ERROR_{cls.upper()}", "inference": f"ERROR_{cls.upper()}", "treatment": float("nan"),
                "rationale": str(d.get("detail", d.get("raw", "")))[:400], "error_class": cls}
    missing = [k for k in ("diagnosis", "inference", "treatment") if k not in d]
    t = _finite_number(d.get("treatment"))
    if missing or t is None or not isinstance(d.get("diagnosis"), str) or not isinstance(d.get("inference"), str):
        cls = "nonfinite" if (not missing and d.get("treatment") is not None and t is None and isinstance(d.get("treatment"), (int, float))) else "schema"
        return {"diagnosis": f"ERROR_{cls.upper()}", "inference": f"ERROR_{cls.upper()}", "treatment": float("nan"),
                "rationale": f"invalid answer object (missing {missing})"[:400], "error_class": cls}
    return {"diagnosis": d["diagnosis"], "inference": d["inference"] or fallback_label, "treatment": round(t, 2),
            "rationale": str(d.get("rationale", ""))[:400], "error_class": ""}


# ---------------------------------------------------------------------------
# Role adapters
# ---------------------------------------------------------------------------
NOVICE_SYSTEM = (
    "You are a newly hired analyst. You have the public task handbook and three workplace documents for one case. "
    "You have no access to past resolved cases and no access to any precedent registry. Apply the handbook's rules "
    "exactly as written, in order; if the annex carries something the handbook does not cover, do your best. "
    "Return only the JSON object requested."
)

INCUMBENT_SYSTEM = (
    "You are the licensed incumbent professional for this task. You hold the full rulebook, including the private "
    "precedent registry, and you work in three hops: (1) read the intake and schedule fields; (2) classify the annex, "
    "including the quantitative silent-collision gate; (3) if a precedent coupling applies, resolve it through the "
    "registry rule. Compute carefully and round the treatment to 2 decimals. Return only the JSON object requested."
)

CRITIC_SYSTEM = (
    "You are a one-pass anomaly critic reviewing a compiled script's answer to a case. You see only the annex document "
    "and the script's output. Estimate the probability that the script's answer is wrong because the annex contains "
    "something the script did not handle. Return only the JSON object requested."
)

COMPILER_SYSTEM = (
    "You are a code-writing agent. You will be given the public rule handbook for an occupational task and, on later "
    "turns, labelled past cases and the assertion failures of your previous program. Write a complete, standalone Python "
    "module that defines\n\n    def execute_compiled_routine(doc_intake: str, doc_sched: str, doc_annex: str) -> dict\n\n"
    "returning {'diagnosis': str, 'inference': str, 'treatment': float, 'validation_flags': list, 'fired': str, "
    "'unhandled_fields': list}. 'fired' names the rule or handler that produced the answer; 'unhandled_fields' lists any "
    "key=value fields in the annex that no rule consumed. Parse the documents with regular expressions, follow the "
    "handbook's rule order, round the treatment to 2 decimals, and use only the Python standard library. The module is the "
    "deliverable: when a repair turn does not let you pin a handler down exactly, implement the best-supported version, keep "
    "everything that already passes, and return the module rather than continuing to derive. Output the module inside one "
    "```python code block and nothing else."
)


def llm_novice(provider: Provider, spec: ONETTaskSpec, pkt: MultiDocWorkPacket) -> Dict[str, object]:
    if provider.is_mock:
        return execute_unassisted_novice(spec, pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
    user = f"PUBLIC HANDBOOK\n{pkt.doc_handbook}\n\nCASE DOCUMENTS\n{_docs_block(pkt)}\nReturn the diagnosis, inference label and treatment."
    return _answer_dict(_parse_json(provider.complete("novice", NOVICE_SYSTEM, user, ANSWER_SCHEMA)), f"{spec.domain_family}_NOVICE_GUESS")


def llm_incumbent(provider: Provider, spec: ONETTaskSpec, pkt: MultiDocWorkPacket, mode: str = "SOLO",
                  challenger_out: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if provider.is_mock:
        return execute_incumbent_react_agent(spec, pkt, mode=mode, challenger_out=challenger_out)
    rulebook = render_full_rulebook(spec)
    if mode == "SOLO":
        user = f"{rulebook}\n\nCASE DOCUMENTS\n{_docs_block(pkt)}\nResolve the case."
        out = _answer_dict(_parse_json(provider.complete("incumbent", INCUMBENT_SYSTEM, user, ANSWER_SCHEMA)), "INCUMBENT_UNRESOLVED")
        out["mode"] = "LLM_SOLO"
        return out
    tool = {k: challenger_out.get(k) for k in ("diagnosis", "inference", "treatment")} if challenger_out else {}
    user = (f"{rulebook}\n\nCASE DOCUMENTS\n{_docs_block(pkt)}\nA compiled script produced this answer backstage: {json.dumps(tool)}.\n"
            "You are the frontstage reviewer. Decide ACCEPT_TOOL if its answer is right, otherwise OVERRIDE with your own answer. "
            "Either way, return the final diagnosis, inference and treatment.")
    d = _parse_json(provider.complete("incumbent_collab", INCUMBENT_SYSTEM, user, COLLAB_SCHEMA))
    out = _answer_dict(d, "INCUMBENT_UNRESOLVED")
    decision = d.get("decision")
    if decision not in ("ACCEPT_TOOL", "OVERRIDE"):
        decision = "INVALID"
        if not out["error_class"]:
            out["error_class"] = "schema"
    out["mode"] = "LLM_COLLAB_" + decision
    return out


def llm_critic(provider: Provider, spec: ONETTaskSpec, compiled_fn: Callable, pkt: MultiDocWorkPacket) -> Dict[str, object]:
    if provider.is_mock:
        return execute_hybrid_tool_plus_critic(spec, compiled_fn, pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
    out = dict(compiled_fn(pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex))
    tool = {k: out.get(k) for k in ("diagnosis", "inference", "treatment", "fired", "unhandled_fields")}
    user = f"ANNEX DOCUMENT\n{pkt.doc_tertiary_annex}\n\nSCRIPT OUTPUT\n{json.dumps(tool)}\n\nEstimate the probability the script's answer is wrong."
    d = _parse_json(provider.complete("critic", CRITIC_SYSTEM, user, CRITIC_SCHEMA))
    u = _finite_number(d.get("epistemic_uncertainty")) if not d.get("error") else None
    if u is None or not 0.0 <= u <= 1.0:
        # a critic answer without a finite probability in [0, 1] (error, missing field, NaN, out of range) is
        # escalated, like a failed tool, and recorded as such; out-of-range values are not clipped into range
        u, status = CFG.u_p_unparsed, "CRITIC_UNPARSED_ESCALATED"
        out["error_class"] = str(d.get("error") or "schema")
    else:
        status = str(d.get("critic_status", "LLM_CRITIC"))[:60]
        out["error_class"] = ""
    out["epistemic_uncertainty"] = u
    out["critic_status"] = status
    return out


# ---------------------------------------------------------------------------
# Compiler role: write, test on the archive, repair
# ---------------------------------------------------------------------------
_SANDBOX_RUNNER = r"""
import importlib.util, json, signal, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
except Exception:
    pass
ERR = lambda tag: {"diagnosis": tag, "inference": tag, "treatment": 0.0, "fired": "ERROR", "unhandled_fields": [], "validation_flags": []}
def _alarm(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
mod_path, batch_path, out_path, per_case = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    import types
    with open(mod_path, encoding="utf-8") as _f:
        _src = _f.read()
    mod = types.ModuleType("generated_tool")
    mod.__file__ = mod_path
    signal.alarm(per_case)
    exec(compile(_src, mod_path, "exec", dont_inherit=True), mod.__dict__)  # exact source, no bytecode cache
    signal.alarm(0)
    fn = getattr(mod, "execute_compiled_routine")
except BaseException as e:
    signal.alarm(0)
    with open(out_path, "w") as f:
        json.dump({"import_error": type(e).__name__ + ": " + str(e)[:200]}, f)
    sys.exit(0)
results = []
with open(batch_path) as f:
    batch = json.load(f)
for case in batch:
    signal.alarm(per_case)
    try:
        out = fn(case["doc_intake"], case["doc_sched"], case["doc_annex"])
        if not isinstance(out, dict):
            raise TypeError("non-dict output")
        results.append({"diagnosis": str(out.get("diagnosis")), "inference": str(out.get("inference")),
                        "treatment": float(out.get("treatment", 0.0)), "fired": str(out.get("fired", "UNKNOWN")),
                        "unhandled_fields": [str(x) for x in (out.get("unhandled_fields") or [])],
                        "validation_flags": [str(x) for x in (out.get("validation_flags") or [])]})
    except BaseException as e:
        results.append(ERR("RUNTIME_ERROR:" + type(e).__name__))
    finally:
        signal.alarm(0)
with open(out_path, "w") as f:
    json.dump({"results": results}, f)
"""


SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def _sb_quote(path: str) -> str:
    return '"' + os.path.realpath(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def sandbox_profile(read_files: List[str], write_dir: str) -> str:
    """macOS Seatbelt profile for one generated-code batch: no network, no process creation, no writes outside the
    batch directory, and no reads under /Users or /Volumes except the Python runtime, the runner, the module and the
    batch directory. System paths outside /Users stay readable (the interpreter needs them)."""
    runtime = {os.path.realpath(sys.base_prefix), os.path.realpath(sys.prefix), os.path.realpath(os.path.dirname(sys.executable))}
    allow_read = " ".join([f"(subpath {_sb_quote(r)})" for r in sorted(runtime)] + [f"(literal {_sb_quote(f)})" for f in read_files]
                          + [f"(subpath {_sb_quote(write_dir)})"])
    return ("(version 1)\n(allow default)\n(deny network*)\n(deny process-fork)\n(deny file-write*)\n"
            f"(allow file-write* (subpath {_sb_quote(write_dir)}) (literal \"/dev/null\"))\n"
            "(deny file-read* (subpath \"/Users\") (subpath \"/Volumes\"))\n"
            f"(allow file-read* {allow_read})\n")


_ISOLATION_OK: Optional[bool] = None


def os_isolation_available() -> bool:
    """True when the macOS sandbox is present AND a self-test confirms it blocks a write outside the batch
    directory and a network connection. Other platforms: False (generated code then runs with resource limits only)."""
    global _ISOLATION_OK
    if _ISOLATION_OK is not None:
        return _ISOLATION_OK
    _ISOLATION_OK = False
    if sys.platform != "darwin" or not os.path.exists(SANDBOX_EXEC):
        return False
    probe = ("import socket, sys\nok = []\n"
             "try:\n    open(sys.argv[1], 'w').write('x'); ok.append('w')\nexcept Exception: pass\n"
             "try:\n    socket.create_connection(('127.0.0.1', 9), timeout=1); ok.append('n')\n"
             "except PermissionError: pass\nexcept Exception: ok.append('n?')\n"
             "open(sys.argv[2], 'w').write(','.join(ok))\n")
    with tempfile.TemporaryDirectory(prefix="juri_sbprobe_") as d, tempfile.TemporaryDirectory(prefix="juri_sbout_") as o:
        script = os.path.join(d, "probe.py")
        with open(script, "w") as f:
            f.write(probe)
        prof = sandbox_profile([script], o)
        try:
            subprocess.run([SANDBOX_EXEC, "-p", prof, sys.executable, "-I", "-B", script, os.path.join(d, "escape.txt"),
                            os.path.join(o, "res.txt")], timeout=30, capture_output=True)
            with open(os.path.join(o, "res.txt")) as f:
                res = f.read()
            _ISOLATION_OK = res == "" and not os.path.exists(os.path.join(d, "escape.txt"))
        except Exception:
            _ISOLATION_OK = False
    return _ISOLATION_OK


class SandboxedTool:
    """Evaluates a generated module on batches of cases in a separate interpreter (-I -B, empty environment) with CPU,
    memory and wall-clock limits. When `isolation` resolves to "os" (macOS sandbox-exec, self-tested) the subprocess
    also runs under a Seatbelt profile that denies network access, process creation, writes outside the batch
    directory and reads of user files. Without OS isolation the limits bound runaway code but are NOT a security
    boundary: the module could read or write files the user can access, so only trusted or reviewed code should be
    run that way. Results are cached by the three document strings."""

    def __init__(self, module_path: str, per_case_seconds: int = 3, batch_timeout_s: int = 300, isolation: str = "auto"):
        if isolation not in ("auto", "required", "none"):
            raise ValueError(f"unknown isolation {isolation!r}")
        if isolation == "none":
            self.isolation = "none"
        elif os_isolation_available():
            self.isolation = "os"
        elif isolation == "required":
            raise RuntimeError("OS isolation required but not available (macOS sandbox-exec self-test failed or absent)")
        else:
            self.isolation = "none"
        self.module_path = module_path
        self.per_case = per_case_seconds
        self.batch_timeout = batch_timeout_s
        self.cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        self.import_error: Optional[str] = None
        runner_dir = tempfile.mkdtemp(prefix="juri_sandbox_")
        self.runner_path = os.path.join(runner_dir, "sandbox_runner.py")
        with open(self.runner_path, "w") as f:
            f.write(_SANDBOX_RUNNER)

    @staticmethod
    def _err(tag: str) -> Dict[str, object]:
        return {"diagnosis": tag, "inference": tag, "treatment": 0.0, "fired": "ERROR", "unhandled_fields": [], "validation_flags": []}

    def prime(self, cases: List[Tuple[str, str, str]]) -> None:
        todo = [c for c in cases if c not in self.cache]
        if not todo:
            return
        if self.import_error:
            for c in todo:
                self.cache[c] = self._err("IMPORT_ERROR")
            return
        with tempfile.TemporaryDirectory(prefix="juri_batch_") as d:
            bpath, opath = os.path.join(d, "batch.json"), os.path.join(d, "out.json")
            with open(bpath, "w") as f:
                json.dump([{"doc_intake": a, "doc_sched": b, "doc_annex": c} for a, b, c in todo], f)
            try:
                cmd = [sys.executable, "-I", "-B", self.runner_path, self.module_path, bpath, opath, str(self.per_case)]
                if self.isolation == "os":
                    cmd = [SANDBOX_EXEC, "-p", sandbox_profile([self.runner_path, self.module_path], d)] + cmd
                subprocess.run(cmd,
                               timeout=self.batch_timeout, capture_output=True, env={"PATH": os.environ.get("PATH", "")})
                with open(opath) as f:
                    data = json.load(f)
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                data = {"results": None}
            if "import_error" in data:
                self.import_error = data["import_error"]
                for c in todo:
                    self.cache[c] = self._err("IMPORT_ERROR")
                return
            results = data.get("results")
            if not results or len(results) != len(todo):
                for c in todo:
                    self.cache[c] = self._err("SANDBOX_FAILURE")
                return
            for c, r in zip(todo, results):
                self.cache[c] = r

    def __call__(self, doc_intake: str, doc_sched: str, doc_annex: str) -> Dict[str, object]:
        key = (doc_intake, doc_sched, doc_annex)
        if key not in self.cache:
            self.prime([key])
        return self.cache[key]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _call_with_timeout(fn: Callable, args: tuple, seconds: int) -> Dict[str, object]:
    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(seconds)
    try:
        out = fn(*args)
        if not isinstance(out, dict):
            return {"diagnosis": "RUNTIME_ERROR", "inference": "RUNTIME_ERROR", "treatment": 0.0, "validation_flags": [], "fired": "ERROR", "unhandled_fields": []}
        out.setdefault("validation_flags", [])
        out.setdefault("fired", "UNKNOWN")
        out.setdefault("unhandled_fields", [])
        return out
    except _Timeout:
        return {"diagnosis": "TIMEOUT", "inference": "TIMEOUT", "treatment": 0.0, "validation_flags": [], "fired": "ERROR", "unhandled_fields": []}
    except Exception as e:  # noqa: BLE001 - model-written code may raise anything
        return {"diagnosis": "RUNTIME_ERROR", "inference": f"RUNTIME_ERROR:{type(e).__name__}", "treatment": 0.0, "validation_flags": [], "fired": "ERROR", "unhandled_fields": []}
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _wrap(fn: Callable, seconds: int) -> Callable:
    def wrapped(a, b, c):
        return _call_with_timeout(fn, (a, b, c), seconds)
    return wrapped


def _extract_code(text: str) -> Optional[str]:
    """The module inside the first fenced block; a bare module is accepted if it defines the entry point; else None."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    body = (m.group(1) if m else text).strip()
    if "def execute_compiled_routine" not in body:
        return None
    return body + "\n"


def _load_source(source: str, path: str) -> Optional[Callable]:
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    try:
        from compiler_agent import load_source_exactly
        module = load_source_exactly(source, path, "mod_" + hashlib.md5(path.encode()).hexdigest()[:10])
        return getattr(module, "execute_compiled_routine", None)
    except Exception:  # noqa: BLE001
        return None


AGENTIC_COMPILER_SYSTEM = COMPILER_SYSTEM + (
    "\n\nYou are working as an agent with tools. `list_archive` lists every resolved case in the public archive with its "
    "annex and expected outcome; `show_cases` returns the full documents of named cases; `test_module` runs a candidate module "
    "on the whole archive and returns the pass count and the failing cases. Use them to discover the rule the handbook omits "
    "and to verify your module before you finish. Treat failing cases as data: hypothesise a functional form in the handbook's "
    "quantities and the unhandled annex field, fit its constants from a few cases (several cases give several equations), check the "
    "fit on all the others, and revise; expect the omitted rule to be piecewise. Keep testing until the module passes the archive "
    "or your turns run out. When you are done, reply with the final module inside one ```python block."
)

AGENTIC_TOOLS = [
    {"name": "list_archive", "description": "List every resolved case in the public archive: id, annex line, expected diagnosis, inference and treatment.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "show_cases", "description": "Return the full three documents and the expected outcome for up to 10 archive cases.",
     "input_schema": {"type": "object", "properties": {"packet_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
                      "required": ["packet_ids"], "additionalProperties": False}},
    {"name": "test_module", "description": "Run a candidate module (full Python source defining execute_compiled_routine) on every archive case; returns passed/total and up to 10 failures with your output and the expected outcome.",
     "input_schema": {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"], "additionalProperties": False}},
]


def llm_compile_agentic(provider: Provider, cfg: LLMConfig, spec: ONETTaskSpec, handbook: str,
                        archive: List[MultiDocWorkPacket], out_dir: str) -> Dict[str, LLMCompiledTool]:
    """One agent conversation with a code-execution loop over the FULL archive, instead of three single-shot generations.
    The deployed module is the best-tested one (highest archive pass count), or the final answer if it was never tested.
    Returned under the key 'v3' so the runner treats it like the third generation; 'v1' and 'v2' are the first two
    tested candidates when they exist."""
    os.makedirs(out_dir, exist_ok=True)
    if provider.is_mock:
        return llm_compile_generations(provider, cfg, spec, handbook, archive, out_dir)
    by_id = {p.packet_id: p for p in archive}
    tested: List[Tuple[int, str]] = []   # (passed, source)

    def list_archive(_inp):
        return "\n".join(f"{p.packet_id} | annex: {p.doc_tertiary_annex.strip().splitlines()[-1][:90]} | expected: {p.gt_diagnosis} {p.gt_inference} {p.gt_treatment}"
                         for p in archive)

    def show_cases(inp):
        ids = [i for i in inp.get("packet_ids", []) if i in by_id][:10]
        return _archive_block([by_id[i] for i in ids], 10) if ids else "no matching case ids"

    def test_module(inp):
        src = _extract_code(inp.get("source", "") or "")
        if src is None:
            return json.dumps({"error": "no execute_compiled_routine in source"})
        path = os.path.join(out_dir, f"tool_{spec.task_id.replace('-', '_')}_cand{len(tested)}_llm.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        fn = SandboxedTool(path, per_case_seconds=cfg.code_timeout_s, isolation=cfg.code_isolation)
        fails = _failures(fn, archive)
        tested.append((len(archive) - len(fails), src))
        return json.dumps({"passed": len(archive) - len(fails), "total": len(archive), "import_error": fn.import_error,
                           "failures": [{"packet_id": p.packet_id, "your_output": {k: o.get(k) for k in ("diagnosis", "inference", "treatment")},
                                         "expected": {"diagnosis": p.gt_diagnosis, "inference": p.gt_inference, "treatment": p.gt_treatment},
                                         "annex": p.doc_tertiary_annex.strip()} for p, o in fails[:10]]})

    user = (f"PUBLIC HANDBOOK\n{handbook}\n\nThe public archive holds {len(archive)} resolved cases. Write the module, test it, "
            f"repair what the handbook omits, and return the final module.")
    text, turns = provider.tool_loop("compiler_agentic", AGENTIC_COMPILER_SYSTEM, user, AGENTIC_TOOLS,
                                     {"list_archive": list_archive, "show_cases": show_cases, "test_module": test_module}, cfg.agentic_turns)
    final_src = _extract_code(text or "")
    if final_src is not None:
        path = os.path.join(out_dir, f"tool_{spec.task_id.replace('-', '_')}_final_llm.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(final_src)
        fn = SandboxedTool(path, per_case_seconds=cfg.code_timeout_s, isolation=cfg.code_isolation)
        tested.append((len(archive) - len(_failures(fn, archive)), final_src))
    history = [f"agentic: {turns} tool turns, {len(tested)} candidates tested"]
    if not tested:
        src = "def execute_compiled_routine(doc_intake, doc_sched, doc_annex):\n    raise RuntimeError('no module')\n"
        history.append("agentic: no module returned")
        tested.append((0, src))
    tools: Dict[str, LLMCompiledTool] = {}
    best = max(range(len(tested)), key=lambda i: tested[i][0])
    picks = {"v1": 0, "v2": min(1, len(tested) - 1), "v3": best}
    for v, idx in picks.items():
        passed, src = tested[idx]
        path = os.path.join(out_dir, f"tool_{spec.task_id.replace('-', '_')}_{v}_llm.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        fn = SandboxedTool(path, per_case_seconds=cfg.code_timeout_s, isolation=cfg.code_isolation)
        try:
            n, cy, hv, _ = compute_ast_and_halstead_metrics(src)
        except SyntaxError:
            n, cy, hv = 0, 0, 0.0
        fails = _failures(fn, archive)
        h = history + [f"{v}: candidate {idx}: {len(fails)}/{len(archive)} archive assertion failures; {len(src.splitlines())} source lines"]
        tools[v] = LLMCompiledTool(spec.task_id, v, path, src, fn, len(fails), n, cy, hv, h)
    return tools


def _archive_block(cases: List[MultiDocWorkPacket], n: int) -> str:
    L = []
    for p in cases[:n]:
        L.append(f"### Resolved case {p.packet_id}\n{_docs_block(p)}EXPECTED: diagnosis={p.gt_diagnosis} inference={p.gt_inference} treatment={p.gt_treatment}\n")
    return "\n".join(L)


def _failures(fn: Callable, archive: List[MultiDocWorkPacket]) -> List[Tuple[MultiDocWorkPacket, Dict[str, object]]]:
    if isinstance(fn, SandboxedTool):
        fn.prime([(p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex) for p in archive])
    fails = []
    for p in archive:
        out = fn(p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex)
        ok = (out.get("diagnosis") == p.gt_diagnosis and out.get("inference") == p.gt_inference
              and abs(float(out.get("treatment", 0.0)) - p.gt_treatment) <= 0.01)
        if not ok:
            fails.append((p, out))
    return fails


@dataclass
class LLMCompiledTool:
    task_id: str
    version: str
    filepath: str
    source: str
    fn: Callable
    archive_failures: int
    ast_node_count: int
    cyclomatic_complexity: int
    halstead_volume: float
    log: List[str] = field(default_factory=list)


def llm_compile_generations(provider: Provider, cfg: LLMConfig, spec: ONETTaskSpec, handbook: str,
                            archive: List[MultiDocWorkPacket], out_dir: str) -> Dict[str, LLMCompiledTool]:
    """Three generations: v1 from the handbook; v2 and v3 repair against archive assertion failures."""
    os.makedirs(out_dir, exist_ok=True)
    tools: Dict[str, LLMCompiledTool] = {}
    if provider.is_mock:
        prev = None
        for v in ("v1", "v2", "v3"):
            art = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, out_dir, v,
                                                         sandbox=archive if v != "v1" else None, previous=prev)
            prev = art
            fn = _load_source(art.source_code, art.filepath)
            tools[v] = LLMCompiledTool(spec.task_id, v, art.filepath, art.source_code, fn, len(_failures(fn, archive)),
                                       art.ast_node_count, art.cyclomatic_complexity, art.halstead_volume, list(art.repair_log))
        return tools

    history: List[str] = []
    prev_source: Optional[str] = None
    prev_fails: List[Tuple[MultiDocWorkPacket, Dict[str, object]]] = []
    for v in ("v1", "v2", "v3"):
        if v == "v1":
            user = f"PUBLIC HANDBOOK\n{handbook}\n\nWrite the module."
        else:
            fail_txt = "\n".join(f"- {p.packet_id}: your output {json.dumps({k: o.get(k) for k in ('diagnosis', 'inference', 'treatment')})} "
                                 f"vs expected diagnosis={p.gt_diagnosis} inference={p.gt_inference} treatment={p.gt_treatment}\n  ANNEX: {p.doc_tertiary_annex.strip()}"
                                 for p, o in prev_fails[:cfg.failure_examples])
            user = (f"PUBLIC HANDBOOK\n{handbook}\n\nLABELLED PAST CASES (public archive, resolved outcomes)\n{_archive_block(archive, cfg.archive_examples)}\n"
                    f"YOUR PREVIOUS MODULE\n```python\n{prev_source}```\n\nASSERTION FAILURES OF YOUR PREVIOUS MODULE ON THE ARCHIVE ({len(prev_fails)} of {len(archive)})\n{fail_txt}\n\n"
                    "Diagnose what the failing cases have in common, add the handler the handbook omits, and return the full corrected module.")
        text = provider.complete(f"compiler_{v}", COMPILER_SYSTEM, user, None)
        source = _extract_code(text)
        if source is None:
            # no module came back (budget exhausted, refusal or API error): the generation is recorded as failed and the
            # previous module is carried forward unchanged, so a failed repair never regresses the deployed tool
            reason = text.strip()[:80].replace("\n", " ") or "empty response"
            history.append(f"{v}: no module returned ({reason}); previous module carried forward")
            source = prev_source if prev_source else "def execute_compiled_routine(doc_intake, doc_sched, doc_annex):\n    raise RuntimeError('no module')\n"
        path = os.path.join(out_dir, f"tool_{spec.task_id.replace('-', '_')}_{v}_llm.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        fn = SandboxedTool(path, per_case_seconds=cfg.code_timeout_s, isolation=cfg.code_isolation)
        try:
            n, cy, hv, _ = compute_ast_and_halstead_metrics(source)
        except SyntaxError:
            n, cy, hv = 0, 0, 0.0
        fails = _failures(fn, archive)
        if fn.import_error:
            history.append(f"{v}: module failed to import in the sandbox ({fn.import_error[:80]})")
        history.append(f"{v}: {len(fails)}/{len(archive)} archive assertion failures; {len(source.splitlines())} source lines")
        tools[v] = LLMCompiledTool(spec.task_id, v, path, source, fn, len(fails), n, cy, hv, list(history))
        prev_source, prev_fails = source, fails
    return tools
