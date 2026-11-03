#!/usr/bin/env python3
"""
LLM arm runner (run_llm_arm.py) — JURI Lab

Runs the laboratory with one or more agent roles filled by a language model (llm_agents.py) and reports
the same Table 1 layout for the LLM arm next to the rule-based baseline computed in the same run, plus
ring-level governance under both. The grader, packets, ground truth and governance game are identical.

    python3 run_llm_arm.py --provider mock                       # protocol test: must equal the reference configuration
    python3 run_llm_arm.py --provider anthropic --corpus primary  # 6 occupations, 18 tasks, 360 packets
    python3 run_llm_arm.py --provider replay --replay-mode legacy --n-eval 10 --transcript data_outputs/llm_transcripts_primary.jsonl
    python3 run_llm_arm.py --provider replay --replay-mode exact --run-id <id>   # a run written with KEY_VERSION 2 records
    python3 run_llm_arm.py --provider anthropic --dry-run        # print the first prompt of each role, no calls

Credentials for --provider anthropic resolve from ANTHROPIC_API_KEY or `ant auth login`.
Model-written code runs in a separate interpreter with CPU, memory and time limits. On macOS the subprocess also
runs under a self-tested sandbox-exec profile (no network, no process creation, no writes outside its batch
directory, no reads of user files); elsewhere, or with --code-isolation none, the limits are not a security
boundary and model-written code should be run only inside a disposable container or VM.

New live runs write a per-run transcript and manifest under data_outputs/llm_runs/<run_id>/; transcript keys
hash both the prompt and the complete request configuration (KEY_VERSION 2). The shipped transcripts predate
this and can only be replayed with --replay-mode legacy, which does not verify the request configuration.
"""

import argparse
import csv
import json
import os
import juri_paths
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import Dict, List

from model_config import CFG
from domain_cases import PRIMARY_CORPUS, FULL_EXTENDED_CORPUS, build_multi_doc_packets_for_task, build_public_sandbox_archive
from compiler_agent import synthesize_compiled_tool_from_handbook, load_compiled_callable
from negotiated_order_lab import (execute_unassisted_novice, execute_hybrid_tool_plus_critic, execute_incumbent_react_agent,
                                  grade_agent_output, negotiate_workplace_governance)
from llm_agents import (LLMConfig, MockProvider, ReplayProvider, AnthropicProvider, StressProvider, SandboxedTool, llm_novice, llm_incumbent, llm_critic,
                        llm_compile_generations, llm_compile_agentic, COMPILER_SYSTEM, NOVICE_SYSTEM, INCUMBENT_SYSTEM, CRITIC_SYSTEM, render_full_rulebook)

BASE_DIR = juri_paths.ROOT
OUT_DIR = os.path.join(BASE_DIR, "data_outputs")
ROLES = ("compiler", "novice", "incumbent", "critic")


def rate(rows, col):
    return round(100.0 * sum(int(r[col]) for r in rows) / len(rows), 1) if rows else 0.0


def write_run_manifest(run_dir: str, args, cfg, corpus, n_eval, transcript: str) -> None:
    """Writes data_outputs/llm_runs/<run_id>/manifest.json before any request: the command line, the complete request
    configuration, code hashes, interpreter and SDK versions, the seed salt, corpus and packet counts. Refuses to reuse
    a run directory, so every live run has its own manifest and transcript."""
    import hashlib, platform
    from dataclasses import asdict
    os.makedirs(run_dir, exist_ok=False)
    def h(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    try:
        import anthropic
        sdk = getattr(anthropic, "__version__", "?")
    except Exception:
        sdk = None
    from llm_agents import KEY_VERSION, request_fingerprint
    man = {"run_id": cfg.run_id, "argv": sys.argv, "llm_config": asdict(cfg), "key_version": KEY_VERSION,
           "request_fingerprints": {r: request_fingerprint(cfg, r, None) for r in ("compiler", "novice", "incumbent", "critic")},
           "seed_salt": CFG.seed_salt, "corpus": args.corpus, "tasks": [t.task_id for t in corpus], "n_eval": n_eval,
           "transcript": os.path.relpath(transcript, BASE_DIR), "python": sys.version.split()[0], "platform": platform.platform(),
           "anthropic_sdk": sdk, "code_sha256": {f: h(os.path.join(BASE_DIR, f)) for f in sorted(os.listdir(BASE_DIR)) if f.endswith(".py")},
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["mock", "anthropic", "replay", "stress"], default="mock")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", help="effort for the answer roles (novice, incumbent, critic)")
    ap.add_argument("--compiler-effort", default="medium", help="effort for the code-writing turns")
    ap.add_argument("--compiler-task-budget", type=int, default=24000, help="advisory token budget shown to the compiler (0 disables)")
    ap.add_argument("--compiler-mode", choices=["generations", "agentic"], default="generations",
                    help="generations: three single-shot turns (v1, v2, v3); agentic: one tool-use conversation with a code-execution loop over the full archive")
    ap.add_argument("--agentic-turns", type=int, default=8)
    ap.add_argument("--roles", default="compiler,novice,incumbent,critic")
    ap.add_argument("--corpus", choices=["primary", "extended"], default="primary")
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--max-calls", type=int, default=1500)
    ap.add_argument("--yes", action="store_true", help="proceed past the call-budget check")
    ap.add_argument("--tag", default="", help="suffix for the output files, so that smoke tests never overwrite a full run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12, help="concurrent model calls")
    ap.add_argument("--max-tasks", type=int, default=None, help="limit to the first N tasks (smoke tests)")
    ap.add_argument("--replay-mode", choices=["legacy", "exact"], default="legacy",
                    help="legacy: prompt-only keys, request configuration NOT verified (shipped transcripts); exact: KEY_VERSION 2 keys incl. configuration")
    ap.add_argument("--run-id", default=None, help="live run identifier (default: UTC timestamp); its transcript and manifest go to data_outputs/llm_runs/<run_id>/")
    ap.add_argument("--code-isolation", choices=["auto", "required", "none"], default="auto",
                    help="generated code: OS sandbox when available (auto), refuse without it (required), or resource limits only (none)")
    args = ap.parse_args()

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    for r in roles:
        if r not in ROLES:
            sys.exit(f"unknown role {r}; choose from {ROLES}")
    corpus = PRIMARY_CORPUS if args.corpus == "primary" else FULL_EXTENDED_CORPUS
    if args.max_tasks:
        corpus = corpus[:args.max_tasks]
    n_eval = args.n_eval or CFG.n_eval_packets
    run_id = args.run_id or (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) if args.provider == "anthropic" else "")
    run_dir = os.path.join(OUT_DIR, "llm_runs", run_id) if run_id else None
    if args.transcript:
        transcript = args.transcript
    elif args.provider == "anthropic" or (args.provider == "replay" and args.replay_mode == "exact"):
        if not run_dir:
            sys.exit("--replay-mode exact needs --run-id or --transcript")
        transcript = os.path.join(run_dir, "transcript.jsonl")
    else:
        transcript = os.path.join(OUT_DIR, f"llm_transcripts_{args.corpus}.jsonl")   # the shipped legacy transcript
    tag = f"{args.provider}_{args.corpus}" + (f"_{args.compiler_mode}" if args.compiler_mode != "generations" else "") + (f"_{args.tag}" if args.tag else "") + (f"_{run_id}" if args.provider == "anthropic" else "")
    tools_dir = os.path.join(BASE_DIR, "compiled_tools_llm", tag)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.compiler_mode == "agentic" and "--compiler-effort" not in sys.argv and "--compiler-task-budget" not in sys.argv:
        args.compiler_effort, args.compiler_task_budget = "high", 48000   # tool turns bound the thinking; the single-shot limits were for generations mode
    cfg = LLMConfig(model=args.model, effort=args.effort, compiler_effort=args.compiler_effort,
                    compiler_task_budget=args.compiler_task_budget, agentic_turns=args.agentic_turns,
                    transcript_path=transcript if args.provider == "anthropic" else None, run_id=run_id,
                    code_isolation=args.code_isolation)
    if args.provider == "replay" and args.replay_mode == "exact" and run_dir and os.path.exists(os.path.join(run_dir, "manifest.json")):
        with open(os.path.join(run_dir, "manifest.json")) as f:
            rec_cfg = json.load(f)["llm_config"]
        rec_cfg.update(transcript_path=None, code_isolation=args.code_isolation)
        cfg = LLMConfig(**rec_cfg)   # exact replay verifies against the recorded configuration, not the CLI defaults
    if args.provider == "mock":
        provider = MockProvider()
    elif args.provider == "stress":
        provider = StressProvider()
    elif args.provider == "replay":
        provider = ReplayProvider(transcript, mode=args.replay_mode, cfg=cfg)
        print(f"REPLAY PROVENANCE: {provider.provenance}")
    else:
        provider = AnthropicProvider(cfg)
        if not args.dry_run:
            write_run_manifest(run_dir, args, cfg, corpus, n_eval, transcript)

    n_tasks, n_pk = len(corpus), len(corpus) * n_eval
    est = (3 * n_tasks if "compiler" in roles else 0) + n_pk * sum(1 for r in roles if r != "compiler") + (n_pk if "incumbent" in roles else 0)
    print(f"LLM arm: provider={args.provider} model={args.model} effort={args.effort}/{args.compiler_effort} roles={roles} corpus={args.corpus} "
          f"tasks={n_tasks} packets={n_pk} estimated model calls={est}")
    if args.provider == "anthropic" and est > args.max_calls and not args.yes and not args.dry_run:
        sys.exit(f"estimated {est} calls exceeds --max-calls {args.max_calls}; re-run with --yes or a smaller corpus/role set")

    if args.dry_run:
        spec = corpus[0]
        pk = build_multi_doc_packets_for_task(spec, n_eval, split="eval")[0]
        print("\n=== compiler system ===\n" + COMPILER_SYSTEM)
        print("\n=== compiler v1 user (handbook) ===\n" + pk.doc_handbook)
        print("\n=== novice system ===\n" + NOVICE_SYSTEM)
        print("\n=== incumbent system ===\n" + INCUMBENT_SYSTEM + "\n\n=== incumbent rulebook ===\n" + render_full_rulebook(spec))
        print("\n=== critic system ===\n" + CRITIC_SYSTEM)
        if args.provider == "anthropic":
            print("\n=== request kwargs (answer role) ===")
            kw = provider.request_kwargs(NOVICE_SYSTEM, "<user prompt>", {"type": "object"})
            print(json.dumps({k: v for k, v in kw.items() if k != "messages"}, indent=1))
        return

    traces: List[Dict[str, object]] = []
    comp_rows: List[Dict[str, object]] = []
    t_start = time.time()

    def prepare(spec):
        packets = build_multi_doc_packets_for_task(spec, n_eval, split="eval")
        archive = build_public_sandbox_archive(spec)
        handbook = packets[0].doc_handbook
        base_dir = os.path.join(tools_dir, "baseline")
        a1 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, base_dir, "v1")
        a2 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, base_dir, "v2", sandbox=archive, previous=a1)
        a3 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, base_dir, "v3", sandbox=archive, previous=a2)
        base_fn = load_compiled_callable(a3)
        comp_row = None
        if "compiler" in roles:
            compile_fn = llm_compile_agentic if args.compiler_mode == "agentic" else llm_compile_generations
            llm_tools = compile_fn(provider, cfg, spec, handbook, archive, os.path.join(tools_dir, "llm"))
            arm_fn = llm_tools["v3"].fn
            if isinstance(arm_fn, SandboxedTool):
                arm_fn.prime([(p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex) for p in packets])
            comp_row = {"task_id": spec.task_id, "ring": spec.ring, "archive_n": len(archive),
                        **{f"{v}_archive_failures": llm_tools[v].archive_failures for v in ("v1", "v2", "v3")},
                        **{f"{v}_halstead": llm_tools[v].halstead_volume for v in ("v1", "v2", "v3")},
                        "log": " | ".join(llm_tools["v3"].log)}
            print(f"  compiled {spec.task_id}: " + " ; ".join(llm_tools["v3"].log) + f"  [{time.time() - t_start:.0f}s]", flush=True)
        else:
            arm_fn = base_fn
        return spec, packets, base_fn, arm_fn, comp_row

    def eval_packet(job):
        spec, pkt, base_fn, arm_fn = job
        d_in, d_sc, d_an = pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex
        b_nov = execute_unassisted_novice(spec, d_in, d_sc, d_an)
        b_tool = base_fn(d_in, d_sc, d_an)
        b_hyb = execute_hybrid_tool_plus_critic(spec, base_fn, d_in, d_sc, d_an)
        b_solo = execute_incumbent_react_agent(spec, pkt, "SOLO")
        b_col = execute_incumbent_react_agent(spec, pkt, "COLLAB_AUDIT", challenger_out=b_tool)
        b_set = negotiate_workplace_governance(spec, pkt, b_hyb, b_col, b_solo)
        l_tool = arm_fn(d_in, d_sc, d_an)
        l_nov = llm_novice(provider, spec, pkt) if "novice" in roles else b_nov
        l_hyb = llm_critic(provider, spec, arm_fn, pkt) if "critic" in roles else execute_hybrid_tool_plus_critic(spec, arm_fn, d_in, d_sc, d_an)
        if "incumbent" in roles:
            l_solo = llm_incumbent(provider, spec, pkt, "SOLO")
            l_col = llm_incumbent(provider, spec, pkt, "COLLAB_AUDIT", challenger_out=l_tool)
        else:
            l_solo = b_solo
            l_col = execute_incumbent_react_agent(spec, pkt, "COLLAB_AUDIT", challenger_out=l_tool)
        l_set = negotiate_workplace_governance(spec, pkt, l_hyb, l_col, l_solo)
        g = lambda o: grade_agent_output(o, pkt)["e2e_ok"]
        return {
            "task_id": spec.task_id, "occ_title": spec.occ_title, "ring": spec.ring, "packet_id": pkt.packet_id,
            "complexity_tier": pkt.complexity_tier, "is_silent": pkt.is_silent_collision,
            "base_novice": g(b_nov), "base_tool_v3": g(b_tool), "base_inc_solo": g(b_solo), "base_collab": g(b_col),
            "base_governance": b_set.selected_governance, "base_sq_accurate": b_set.final_output_accurate,
            "arm_novice": g(l_nov), "arm_tool_v3": g(l_tool), "arm_inc_solo": g(l_solo), "arm_collab": g(l_col),
            "arm_u_p": l_hyb.get("epistemic_uncertainty"), "arm_critic_status": l_hyb.get("critic_status"),
            "arm_governance": l_set.selected_governance, "arm_sq_accurate": l_set.final_output_accurate,
            "arm_inc_mode": l_solo.get("mode", ""), "arm_collab_mode": l_col.get("mode", ""),
            "arm_novice_out": {k: l_nov.get(k) for k in ("diagnosis", "inference", "treatment")},
            "arm_solo_out": {k: l_solo.get(k) for k in ("diagnosis", "inference", "treatment")},
            "gt": {"diagnosis": pkt.gt_diagnosis, "inference": pkt.gt_inference, "treatment": pkt.gt_treatment},
        }

    workers = max(1, args.workers if args.provider == "anthropic" else 1)
    with ThreadPoolExecutor(max_workers=min(workers, len(corpus))) as ex:
        prepared = list(ex.map(prepare, corpus))
    comp_rows = [c for _, _, _, _, c in prepared if c]
    jobs = [(spec, pkt, base_fn, arm_fn) for spec, packets, base_fn, arm_fn, _ in prepared for pkt in packets]
    print(f"  compiled {len(prepared)} tasks in {time.time() - t_start:.0f}s; evaluating {len(jobs)} packets with {workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tr in ex.map(eval_packet, jobs):
            traces.append(tr)
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print(f"  evaluated {done}/{len(jobs)} packets [{time.time() - t_start:.0f}s]", flush=True)

    # ---- outputs ----
    with open(os.path.join(OUT_DIR, f"results_llm_arm_{tag}_traces.json"), "w") as f:
        json.dump(traces, f, indent=1)
    if comp_rows:
        with open(os.path.join(OUT_DIR, f"results_llm_arm_{tag}_compiler.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys())); w.writeheader(); w.writerows(comp_rows)

    tiers = {t: [x for x in traces if x["complexity_tier"] == t] for t in range(4)}
    print("\n" + "=" * 112)
    print(f"TABLE 6: RULE-BASED BASELINE vs LLM ARM ({args.provider}, {args.model if args.provider != 'mock' else 'rule-based stand-in'}; roles={','.join(roles)}; N = {len(traces)})")
    print("=" * 112)
    print(f"{'Regime':<42} | {'Novice b/a':>13} | {'Tool_v3 b/a':>13} | {'Inc_Solo b/a':>13} | {'Collab b/a':>13}")
    print("-" * 112)
    rows = [("Tier 0 routine", tiers[0]), ("Tier 1 explicit alert", tiers[1]), ("Tier 2 piecewise collision", tiers[2]),
            ("Tier 3 precedent collision", tiers[3]), ("Silent collisions", [x for x in traces if x["is_silent"]]), ("All packets", traces)]
    for name, sub in rows:
        cells = [f"{rate(sub, 'base_' + c):5.1f}/{rate(sub, 'arm_' + c):5.1f}" for c in ("novice", "tool_v3", "inc_solo", "collab")]
        print(f"{name + f' (N={len(sub)})':<42} | " + " | ".join(f"{c:>13}" for c in cells))
    print("-" * 112)
    for ring in ("Outer_Periphery", "Middle_Buffer", "Inner_Core"):
        rr = [x for x in traces if x["ring"] == ring]
        if not rr:
            continue
        gb, ga = Counter(x["base_governance"] for x in rr), Counter(x["arm_governance"] for x in rr)
        n = len(rr)
        print(f"{ring:<16} governance baseline poach/settle/gate = {100*gb['DIRECT_CHALLENGER_POACHING']/n:4.1f}/{100*gb['ACCOUNTABILITY_SETTLEMENT']/n:4.1f}/{100*gb['INCUMBENT_FULL_GATEKEEPING']/n:4.1f}  "
              f"arm = {100*ga['DIRECT_CHALLENGER_POACHING']/n:4.1f}/{100*ga['ACCOUNTABILITY_SETTLEMENT']/n:4.1f}/{100*ga['INCUMBENT_FULL_GATEKEEPING']/n:4.1f}  "
              f"SQ error baseline/arm = {100*sum(1 for x in rr if not x['base_sq_accurate'])/n:4.1f}/{100*sum(1 for x in rr if not x['arm_sq_accurate'])/n:4.1f}")
    if args.provider == "replay":
        prov = {"provenance": provider.provenance, "replay_mode": args.replay_mode, "transcript": os.path.relpath(transcript, BASE_DIR),
                "tool_events_reexecuted": len(provider.tool_events), "prompt_mismatches": provider.prompt_mismatches,
                "ambiguous_tool_loop_starts": provider.ambiguous_starts}
        with open(os.path.join(OUT_DIR, f"results_llm_arm_{tag}_replay_provenance.json"), "w") as f:
            json.dump({**prov, "n_requested_keys": len(provider.requested_keys), "requested_keys": provider.requested_keys,
                       "tool_events": provider.tool_events}, f, indent=1)
        print(f"\nREPLAY PROVENANCE: {json.dumps(prov)}")
    if args.provider == "anthropic":
        print(f"\nmodel calls: {provider.calls}; API errors after retries: {provider.errors}; tokens in/out: {provider.usage}; "
              f"transcript: {transcript}; wall time {time.time() - t_start:.0f}s")
    if args.provider == "stress":
        bad = Counter(x["arm_tool_v3"] for x in traces)
        print(f"\nstress test: completed without a crash; arm tool outcomes (1 = correct) = {dict(bad)}; "
              f"provider calls = {provider.n}")
    if args.provider == "mock":
        same = all(x["base_" + c] == x["arm_" + c] for x in traces for c in ("novice", "tool_v3", "inc_solo", "collab"))
        print("\nmock protocol test:", "PASS (arm reproduces the rule-based baseline exactly)" if same else "FAIL")


if __name__ == "__main__":
    main()
