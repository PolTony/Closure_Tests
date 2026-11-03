#!/usr/bin/env python3
"""
Bumping chains (run_bumping.py) — JURI Lab, proposition P6

Adds a SECOND profession to every task: an adjacent subordinate occupation that held a division-of-labour
settlement with the incumbent before any tool existed (paralegal beside lawyer, registered nurse beside nurse
practitioner, technician beside engineer). The subordinate has weaker closure (lower wage, little licensing,
no statutory sign-off) and a narrower information set: the public handbook, the public archive and the
profession's conventions (Tier 2), but not the private registry (Tier 3). When it recognises an overt
precedent memo it refers the case upward; a silent collision it treats as routine.

The governance game gains three arrangements, and a pre-tool wave (v0) is evaluated alongside the three
compiler generations, so the table answers P6 directly: when the buffer is compiled, who loses first?

Arrangements (cost = wage bill + expected error cost; penalty applies when a gated task is done without the
licensed incumbent's signature):
  POACH          tool alone                       w_chall + penalty + u_p L
  SETTLE_INC     tool backstage, incumbent signs  w_chall + r_inc w_inc + r(u_p) L
  GATE           incumbent alone                  w_inc + 0.008 L
  SUB_ALONE      subordinate alone                w_sub + penalty + u_sub L
  SUB_SUPERVISED subordinate works, incumbent signs  w_sub + r_inc w_inc + residual_sup L
  SETTLE_SUB     tool backstage, subordinate signs   w_chall + r_sub w_sub + (r(u_p) + u_sub) L + penalty
v0 (no tool) offers GATE, SUB_ALONE and SUB_SUPERVISED only.

Outputs: Table 10 (by ring and wave: arrangement shares, incumbent and subordinate hours relative to v0,
error) and data_outputs/bumping_<corpus>.csv.
"""

import argparse
import csv
import math
import os
import juri_paths
import shutil
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from model_config import CFG, reset_config, set_config
from domain_cases import (ONETTaskSpec, PRIMARY_CORPUS, FULL_EXTENDED_CORPUS, MultiDocWorkPacket, PRECEDENT_REGISTRY,
                          build_multi_doc_packets_for_task, build_public_sandbox_archive, parse_doc_fields, evaluate_rulebook)
from compiler_agent import synthesize_compiled_tool_from_handbook, load_compiled_callable
from negotiated_order_lab import execute_hybrid_tool_plus_critic, execute_incumbent_react_agent, grade_agent_output

BASE_DIR = juri_paths.ROOT
OUT_DIR = os.path.join(BASE_DIR, "data_outputs")
TOOLS_DIR = os.path.join(BASE_DIR, "bumping_tools", f"run_{os.getpid()}")   # unique per process: concurrent runs must not share emitted modules
RINGS = ("Outer_Periphery", "Middle_Buffer", "Inner_Core")
WAVES = ("v0", "v1", "v2", "v3")
ARRANGEMENTS = ("POACH", "SETTLE_INC", "GATE", "SUB_ALONE", "SUB_SUPERVISED", "SETTLE_SUB")

# Named adjacent pairs where the corpus holds the subordinate's own attributes (SOC, wage ratio, licensing, z, job zone).
# Other occupations get a generic subordinate derived from the incumbent (see subordinate_profile).
NAMED_PAIRS = {
    "Lawyers": "Paralegals",
    "Nurse_Practitioners": "Registered_Nurses",
}


@dataclass
class Subordinate:
    title: str
    wage_ratio: float
    licensing_rate: float
    accountability_z: float
    job_zone: int
    source: str   # "corpus" or "generic"


SOURCED_CHALLENGERS: Dict[str, Subordinate] = {}


def load_sourced_challengers() -> Dict[str, Subordinate]:
    """The empirical programme's named challenger for each incumbent (data_inputs/lab18_closure_values.csv: dyad analysis or
    proposed pair) with the challenger's own attributes from the native SOC-2018 closure file: OEWS 2024 wage over the
    all-occupations median, CPS licence share (unit level), O*NET 27.0 accountability z. Job zone is the incumbent's minus one."""
    import csv, io
    base = juri_paths.ROOT
    def load(path):
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(io.StringIO("".join(l for l in f if not l.startswith("#")))))
    lab = load(os.path.join(base, "data_inputs", "lab18_closure_values.csv"))
    native = {r["soc2018"]: r for r in load(os.path.join(base, "data_inputs", "closure_native_soc2018_codes.csv"))}
    out = {}
    for r in lab:
        c = native.get(r["challenger_soc2018"])
        if not c:
            continue
        def f(x, d=None):
            try:
                return float(x)
            except (TypeError, ValueError):
                return d
        w = f(c["wage_ratio_to_all_occ_median_2024"]) or f(c["wage_ratio_to_all_occ_median_2019"])
        if w is None:
            continue
        out[r["occ_title"]] = Subordinate(c["occ_title"].strip().replace(" ", "_"), round(w, 3), round(f(c["lic_cps_1519_unit"], 0.05), 3),
                                          round(f(c["accountability_z_native22"], 0.0), 3), 0, f"sourced:{r['challenger_basis'][:20]}")
    return out


def subordinate_profile(spec: ONETTaskSpec, corpus: List[ONETTaskSpec]) -> Subordinate:
    if SOURCED_CHALLENGERS:
        sub = SOURCED_CHALLENGERS.get(spec.occ_title)
        if sub is not None:
            return Subordinate(sub.title, sub.wage_ratio, sub.licensing_rate, sub.accountability_z, max(1, spec.job_zone - 1), sub.source)
    if spec.occ_title in NAMED_PAIRS:
        sub = next((s for s in corpus if s.occ_title == NAMED_PAIRS[spec.occ_title]), None)
        if sub is not None:
            return Subordinate(sub.occ_title, sub.wage_ratio_inc_chall, sub.cps_licensing_rate, sub.accountability_z, sub.job_zone, "corpus")
    return Subordinate(f"{spec.occ_title}_assistant", 1.0 + CFG.sub_wage_share * (spec.wage_ratio_inc_chall - 1.0),
                       CFG.sub_licensing_rate, max(0.0, spec.accountability_z - 1.0), max(1, spec.job_zone - 1), "generic")


def execute_subordinate(spec: ONETTaskSpec, sub: Subordinate, pkt: MultiDocWorkPacket) -> Dict[str, object]:
    """Handbook + conventions, no registry. Overt precedent memos are referred upward; silent ones look routine.
    Slips above the subordinate's own (lower) noise threshold, set by its job zone."""
    f = parse_doc_fields(pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
    v1, v2, v3, v4, noise = f["V1"], f["V2"], f["V3"], f["V4"], f["NOISE"]
    if f["TIER3_MEMO"] or f["base_haircut"] is not None:
        return {"diagnosis": "", "inference": "REFERRED_TO_INCUMBENT", "treatment": 0.0, "refer": True}
    if f["EXPLICIT_TIER1_OVERRIDE"]:
        tier, sev = 1, f["ANNEX_SEVERITY"]
    elif f["severity_index"] is not None:
        tier, sev = 2, f["severity_index"]
    else:
        tier, sev = 0, 0.0
    d, i, t = evaluate_rulebook(spec, v1, v2, v3, v4, sev, tier)
    if noise > (CFG.solo_slip_base + CFG.solo_slip_per_jz * sub.job_zone):
        t = round(t * CFG.solo_slip_multiplier, 2)
    return {"diagnosis": d, "inference": i, "treatment": t, "refer": False}


def arrangement_costs(spec: ONETTaskSpec, sub: Subordinate, u_p: Optional[float], has_tool: bool) -> Dict[str, float]:
    w_chall, w_inc, w_sub = CFG.w_chall, spec.wage_ratio_inc_chall, sub.wage_ratio
    L = CFG.liability_scale * math.exp(CFG.liability_exp * spec.accountability_z)
    penalty = (CFG.penalty_base + CFG.penalty_per_lic * spec.cps_licensing_rate) if spec.statutory_gate == 1 else 0.0
    r_inc = CFG.review_fraction_base + CFG.review_fraction_per_lic * spec.cps_licensing_rate
    costs = {
        "GATE": w_inc + CFG.residual_incumbent * L,
        "SUB_ALONE": w_sub + penalty + CFG.sub_u_prior * L,
        "SUB_SUPERVISED": w_sub + r_inc * w_inc + CFG.sub_supervised_residual * L,
    }
    if has_tool and u_p is not None:
        r_u = (CFG.residual_settle_low * u_p) if u_p <= CFG.residual_settle_cut else (CFG.residual_settle_high * u_p)
        costs["POACH"] = w_chall + penalty + u_p * L
        costs["SETTLE_INC"] = w_chall + r_inc * w_inc + r_u * L
        costs["SETTLE_SUB"] = w_chall + CFG.sub_review_fraction * w_sub + (r_u + CFG.sub_u_prior) * L + penalty
    return costs


def resolve(spec: ONETTaskSpec, sub: Subordinate, pkt: MultiDocWorkPacket, fn, has_tool: bool) -> Tuple[str, int, float, float, float]:
    """Returns (arrangement, accurate, incumbent_hours, subordinate_hours, labour_cost)."""
    r_inc = CFG.review_fraction_base + CFG.review_fraction_per_lic * spec.cps_licensing_rate
    w_inc, w_sub = spec.wage_ratio_inc_chall, sub.wage_ratio
    solo = execute_incumbent_react_agent(spec, pkt, "SOLO")
    sub_out = execute_subordinate(spec, sub, pkt)
    u_p, tool_out, hyb = None, None, None
    if has_tool:
        tool_out = fn(pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
        hyb = execute_hybrid_tool_plus_critic(spec, fn, pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
        u_p = float(hyb.get("epistemic_uncertainty", 0.10))
    costs = arrangement_costs(spec, sub, u_p, has_tool)
    arr = min(costs, key=costs.get)
    if arr == "GATE":
        final, hi, hs, cost = solo, 1.0, 0.0, w_inc
    elif arr == "SUB_ALONE":
        if sub_out["refer"]:   # referral: the incumbent resolves; the subordinate's intake time is sunk
            final, hi, hs, cost = solo, 1.0, 0.25, w_inc + 0.25 * w_sub
        else:
            final, hi, hs, cost = sub_out, 0.0, 1.0, w_sub
    elif arr == "SUB_SUPERVISED":
        if sub_out["refer"]:
            final, hi, hs, cost = solo, 1.0, 0.25, w_inc + 0.25 * w_sub
        else:
            final = execute_incumbent_react_agent(spec, pkt, "COLLAB_AUDIT", challenger_out=sub_out)
            hi, hs, cost = r_inc, 1.0, w_sub + r_inc * w_inc
    elif arr == "POACH":
        final, hi, hs, cost = tool_out, 0.0, 0.0, CFG.w_chall
    elif arr == "SETTLE_INC":
        final = execute_incumbent_react_agent(spec, pkt, "COLLAB_AUDIT", challenger_out=tool_out)
        hi, hs, cost = r_inc, 0.0, CFG.w_chall + r_inc * w_inc
    else:   # SETTLE_SUB: the subordinate reviews the tool with what it knows (handbook and conventions)
        if sub_out["refer"]:
            final, hi, hs, cost = solo, 1.0, CFG.sub_review_fraction, w_inc + CFG.sub_review_fraction * w_sub
        else:
            agree = (tool_out.get("diagnosis") == sub_out["diagnosis"] and tool_out.get("inference") == sub_out["inference"]
                     and abs(float(tool_out.get("treatment", 0.0)) - float(sub_out["treatment"])) <= 0.01)
            final = tool_out if agree else sub_out
            hi, hs, cost = 0.0, CFG.sub_review_fraction, CFG.w_chall + CFG.sub_review_fraction * w_sub
    return arr, grade_agent_output(final, pkt)["e2e_ok"], hi, hs, cost


def run(corpus: List[ONETTaskSpec], n_eval: int, overrides: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    reset_config()
    if overrides:
        set_config(**overrides)
    os.makedirs(TOOLS_DIR, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for spec in corpus:
        sub = subordinate_profile(spec, FULL_EXTENDED_CORPUS)
        packets = build_multi_doc_packets_for_task(spec, n_eval, split="eval")
        archive = build_public_sandbox_archive(spec)
        hb = packets[0].doc_handbook
        a1 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, hb, TOOLS_DIR, "v1")
        a2 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, hb, TOOLS_DIR, "v2", sandbox=archive, previous=a1)
        a3 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, hb, TOOLS_DIR, "v3", sandbox=archive, previous=a2)
        fns = {"v0": None, "v1": load_compiled_callable(a1), "v2": load_compiled_callable(a2), "v3": load_compiled_callable(a3)}
        for wave in WAVES:
            for pkt in packets:
                arr, ok, hi, hs, cost = resolve(spec, sub, pkt, fns[wave], has_tool=(wave != "v0"))
                rows.append({"task_id": spec.task_id, "occ_title": spec.occ_title, "subordinate": sub.title, "sub_source": sub.source,
                             "ring": spec.ring, "statutory_gate": spec.statutory_gate, "wave": wave, "packet_id": pkt.packet_id,
                             "tier": pkt.complexity_tier, "silent": int(pkt.is_silent_collision), "arrangement": arr,
                             "accurate": ok, "inc_hours": hi, "sub_hours": hs, "labor_cost": round(cost, 4),
                             "weight": spec.task_importance_wt})
        print(f"  {spec.task_id} done ({sub.title}, {sub.source})", flush=True)
    shutil.rmtree(TOOLS_DIR, ignore_errors=True)
    return rows


def print_table10(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 150)
    print("TABLE 10: BUMPING CHAINS — two professions per task; arrangement shares and hours by ring and tool wave (v0 = before any tool)")
    print("=" * 150)
    hdr = f"{'Ring':<16} | {'wave':>4} | " + " | ".join(f"{a:>14}" for a in ARRANGEMENTS) + f" | {'Inc hrs':>7} | {'Sub hrs':>7} | {'Error':>5}"
    print(hdr); print("-" * 150)
    for ring in RINGS:
        base_i = base_s = None
        for wave in WAVES:
            rr = [r for r in rows if r["ring"] == ring and r["wave"] == wave]
            n = len(rr)
            if not n:
                continue
            shares = {a: 100.0 * sum(1 for r in rr if r["arrangement"] == a) / n for a in ARRANGEMENTS}
            hi = sum(float(r["inc_hours"]) for r in rr) / n
            hs = sum(float(r["sub_hours"]) for r in rr) / n
            if wave == "v0":
                base_i, base_s = hi, hs
            err = 100.0 * sum(1 for r in rr if not int(r["accurate"])) / n
            fi = f"{100*hi/base_i:6.0f}" if base_i > 1e-6 else "   n/a"
            fs = f"{100*hs/base_s:6.0f}" if base_s > 1e-6 else "   n/a"
            print(f"{ring:<16} | {wave:>4} | " + " | ".join(f"{shares[a]:13.1f}%" for a in ARRANGEMENTS)
                  + f" | {fi} | {fs} | {err:4.1f}%")
    print("-" * 150)
    print("Inc hrs / Sub hrs = incumbent and subordinate hours per packet, indexed to 100 at wave v0 (the pre-tool division of labour); n/a when v0 hours are zero.")
    # who loses first: hours lost v0 -> v3, by ring
    print("\nWho loses first (hours at v3 as a share of hours at v0):")
    for ring in RINGS:
        r0 = [r for r in rows if r["ring"] == ring and r["wave"] == "v0"]; r3 = [r for r in rows if r["ring"] == ring and r["wave"] == "v3"]
        if not r0: continue
        hi0 = sum(float(r["inc_hours"]) for r in r0); hi3 = sum(float(r["inc_hours"]) for r in r3)
        hs0 = sum(float(r["sub_hours"]) for r in r0); hs3 = sum(float(r["sub_hours"]) for r in r3)
        si = f"{100*hi3/hi0:5.0f}%" if hi0 > 1e-6 else "  n/a (no pre-tool hours)"
        print(f"  {ring:<16} incumbent {si}   subordinate {100*hs3/max(hs0,1e-9):5.0f}%")
    print("\nNamed pairs (hours at v3 as a share of v0):")
    for occ, subt in NAMED_PAIRS.items():
        r0 = [r for r in rows if r["occ_title"] == occ and r["wave"] == "v0"]; r3 = [r for r in rows if r["occ_title"] == occ and r["wave"] == "v3"]
        if not r0: continue
        hi0 = sum(float(r["inc_hours"]) for r in r0); hi3 = sum(float(r["inc_hours"]) for r in r3)
        hs0 = sum(float(r["sub_hours"]) for r in r0); hs3 = sum(float(r["sub_hours"]) for r in r3)
        print(f"  {occ} / {subt}: incumbent {100*hi3/max(hi0,1e-9):5.0f}%   subordinate {100*hs3/max(hs0,1e-9):5.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["primary", "extended"], default="extended")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a model_config field, e.g. --set sub_u_prior=0.02")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    ap.add_argument("--subordinates", choices=["lab", "sourced"], default="lab",
                    help="lab: named pairs from the corpus plus a generic assistant; sourced: the empirical programme's named challenger per occupation with its own attributes")
    ap.add_argument("--gates", choices=["registered", "all", "none"], default="registered",
                    help="statutory sign-off as registered, imposed on every core and buffer task, or removed everywhere (closure-formation counterfactuals)")
    args = ap.parse_args()
    if args.subordinates == "sourced":
        SOURCED_CHALLENGERS.update(load_sourced_challengers())
        print(f"sourced challengers loaded for {len(SOURCED_CHALLENGERS)} occupations", flush=True)
        if not args.tag:
            args.tag = "_sourced"
    corpus = PRIMARY_CORPUS if args.corpus == "primary" else FULL_EXTENDED_CORPUS
    if args.gates != "registered":
        corpus = [replace(s, statutory_gate=(1 if (args.gates == "all" and s.ring != "Outer_Periphery") else 0)) for s in corpus]
        if not args.tag:
            args.tag = f"_gates_{args.gates}"
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(CFG, k)
        overrides[k] = (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v)
    rows = run(corpus, args.n_eval, overrides)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"bumping_{args.corpus}{args.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print_table10(rows)


if __name__ == "__main__":
    main()
