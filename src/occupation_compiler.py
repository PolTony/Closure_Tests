#!/usr/bin/env python3
"""
Occupation compiler (occupation_compiler.py) — JURI Lab, research-program stream 1

An AI-enabled research loop under human direction: for an occupation that the laboratory does not yet contain,
a language model drafts an executable rulebook in the laboratory's rule language from the occupation's O*NET task
statements and detailed work activities; the draft is validated deterministically (does it parse, does it use only
the allowed identifiers, does each tier produce a distinct, finite treatment, is the convention piecewise, does the
precedent rule couple a registry constant with the ratio); a validated draft is registered as a new domain family
and the registered pipeline (compiler v1-v3, agents, governance) is run on three ring tasks built from the O*NET
proposal; and the draft is written out for a practitioner to validate. Nothing the model writes is treated as a
fact about the occupation until a practitioner has signed it off: the file carries `validated_by: ""`.

    python3 occupation_compiler.py --provider mock --occupations 29-1051          # protocol test, no credentials
    python3 occupation_compiler.py --provider anthropic --occupations 29-1051,29-1215,29-1021 --yes

Inputs: data_inputs/proposed_corpus_tasks.csv (ring tasks), onet31_tasks_to_dwas_lab18_and_additions.csv (work
activities), closure_table_units_753.csv and closure_native_soc2018_codes.csv (attributes). Outputs:
data_inputs/drafted_rulebooks/<provider>/<run_id>/<soc>.json (immutable) and data_outputs/occupation_compiler_<provider>_<run_id>.csv.
The five drafts at the top of data_inputs/drafted_rulebooks/ are the legacy Anthropic run; `--provider saved` re-validates and
re-evaluates them under the current code without a model call and writes data_outputs/occupation_compiler_saved_drafts.csv.
"""

import argparse
import time
import csv
import hashlib
import io
import json
import math
import os
import juri_paths
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import domain_cases as dc
from domain_cases import DomainRulebook, ONETTaskSpec, _compile_expr, _ev, FULL_EXTENDED_CORPUS
from llm_agents import LLMConfig, AnthropicProvider, MockProvider

BASE = juri_paths.ROOT
INP = os.path.join(BASE, "data_inputs")
DRAFTS = os.path.join(INP, "drafted_rulebooks")
OUT_DIR = os.path.join(BASE, "data_outputs")
TOOLS_DIR = os.path.join(BASE, "occupation_compiler_tools")

RULEBOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "family": {"type": "string", "description": "Short upper-case family name, letters and underscores, e.g. PHARMACY"},
        "quantities": {"type": "object", "description": "What V1, V2, V3, V4 and ANNEX_SEVERITY stand for in this occupation",
                       "properties": {"V1": {"type": "string"}, "V2": {"type": "string"}, "V3": {"type": "string"}, "V4": {"type": "string"}, "ANNEX_SEVERITY": {"type": "string"}},
                       "required": ["V1", "V2", "V3", "V4", "ANNEX_SEVERITY"], "additionalProperties": False},
        "lets": {"type": "array", "description": "One to three [name, expression] pairs", "items": {"type": "array", "items": {"type": "string"}}},
        "tier1_rule": {"type": "array", "description": "[label, expression]", "items": {"type": "string"}},
        "breach_rule": {"type": "array", "description": "[label, expression]", "items": {"type": "string"}},
        "base_rule": {"type": "array", "description": "[label, expression]", "items": {"type": "string"}},
        "tier2_rule": {"type": "array", "description": "[label, expression]", "items": {"type": "string"}},
        "tier3_lets": {"type": "array", "description": "One or two [name, expression] pairs", "items": {"type": "array", "items": {"type": "string"}}},
        "tier3_rule": {"type": "array", "description": "[label, expression]", "items": {"type": "string"}},
        "registry": {"type": "array", "description": "Exactly three named precedent constants, as {name, value} pairs; snake_case names, values with two to four decimals",
                     "items": {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                               "required": ["name", "value"], "additionalProperties": False}},
        "rationale": {"type": "string", "description": "Why these quantities and rules fit the occupation's tasks, in plain language"},
    },
    "required": ["family", "quantities", "lets", "tier1_rule", "breach_rule", "base_rule", "tier2_rule", "tier3_lets", "tier3_rule", "registry", "rationale"],
    "additionalProperties": False,
}

SYSTEM = (
    "You draft rulebooks for a research laboratory that studies how AI compilation reshapes professional work. A rulebook "
    "describes ONE professional task family in a small formal rule language, in four layers: public rules that a handbook "
    "would print; one unwritten convention that practitioners apply (never printed, present only in resolved cases); and one "
    "precedent rule that depends on private registry constants. You will be given the occupation, its three ring tasks and "
    "their detailed work activities, and a complete worked example from accounting.\n\n"
    "The rule language. Expressions use numbers, the variables V1 V2 V3 V4 (case magnitudes from the documents), SEV or "
    "ANNEX_SEVERITY (a severity in [0,1] from the annex), PRIMARY_CUTOFF and RATIO_CUTOFF, names defined by earlier LETs, "
    "registry constants written REG.<name>, the operators + - * / **, parentheses, MIN(a, b), MAX(a, b), and the conditional "
    "`a IF condition ELSE b` where the condition compares two expressions with >= > <= < ==. Nothing else: no functions, no "
    "text, no other variables.\n\n"
    "Constraints that the validator enforces. (1) Every expression must parse. (2) Public LETs and the three public rules "
    "use only V1..V4, ANNEX_SEVERITY, the cutoffs and LET names. (3) The convention (tier2_rule) must be piecewise in V2: it "
    "contains exactly one `(c_a IF V2 >= b ELSE c_b)` with a breakpoint b between 0.08 and 0.20 that is NOT a published "
    "cutoff, and it must involve SEV, V3 and V4. (4) The precedent rule (tier3_rule) must use at least two REG.<name> "
    "constants and a tier3 LET that couples SEV with V2 (an interaction). (5) Treatments must be finite and of the same "
    "order as the example for V1 in [100, 900], V2 in [0.02, 0.25], V3 in [20, 120], V4 in [100, 500], SEV in [0, 1], and "
    "the four rules must give different treatments on typical cases. (6) Labels are upper-case, prefixed by the family name, and distinct. (7) The "
    "registry holds exactly three named constants with plausible magnitudes (multipliers near 1 to 2, rates between 0 and 1).\n\n"
    "Make the quantities and labels fit the occupation (what V1..V4 would be in a real case file of this work), keep the "
    "arithmetic of the same scale as the example, and explain your choices briefly in `rationale`. Return only the JSON object."
)

EXAMPLE = {
    "family": "CPA", "quantities": {"V1": "transaction price to allocate", "V2": "deferred-tax threshold ratio", "V3": "reserve adjustment", "V4": "schedule parameter (standalone selling price basis)", "ANNEX_SEVERITY": "severity of the annex exception"},
    "lets": [["ssp_weight", "V4 / (V1 + V4)"], ["allocated_rev", "V1 * (0.65 + 0.35 * ssp_weight)"]],
    "tier1_rule": ["CPA_EXPLICIT_SUBSEQUENT_EVENT", "allocated_rev * (1 - 0.38 * ANNEX_SEVERITY) - 1.15 * V3"],
    "breach_rule": ["CPA_ASC606_DEFERRAL_VALUATION", "allocated_rev * (1 - 0.40 * V2) + 0.40 * V3"],
    "base_rule": ["CPA_UNQUALIFIED_LEDGER_RECON", "allocated_rev + 0.30 * V3"],
    "tier2_rule": ["CPA_PIECEWISE_TAX_VALUATION_ALLOWANCE", "allocated_rev * (1 - (0.48 IF V2 >= 0.12 ELSE 0.34) * SEV) - 1.25 * V3 + 0.10 * V4"],
    "tier3_lets": [["eff_sev", "SEV * (1 + REG.vie_impairment_mult * V2)"]],
    "tier3_rule": ["CPA_ASC810_VIE_GOING_CONCERN_DERECOGNITION", "allocated_rev * (1 - REG.valuation_allowance_rate * eff_sev) - REG.covenant_clawback * REG.vie_impairment_mult * V3"],
    "registry": [{"name": "vie_impairment_mult", "value": 1.44}, {"name": "valuation_allowance_rate", "value": 0.38}, {"name": "covenant_clawback", "value": 1.32}],
    "rationale": "Revenue allocation under ASC 606 with a deferred-tax valuation allowance as the unwritten convention and a VIE derecognition precedent.",
}


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------
def _load(name):
    with open(os.path.join(INP, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(io.StringIO("".join(l for l in f if not l.startswith("#")))))


def occupation_inputs(soc: str) -> Dict[str, object]:
    tasks = [r for r in _load("proposed_corpus_tasks.csv") if r["soc2018"] == soc]
    if len(tasks) != 3:
        raise SystemExit(f"{soc}: proposed_corpus_tasks.csv holds {len(tasks)} ring tasks (need 3); run propose_corpus.py")
    dwas = [r for r in _load("onet31_tasks_to_dwas_lab18_and_additions.csv") if r["soc2018"] == soc]
    by_task = {}
    for r in dwas:
        by_task.setdefault(r["Task ID"], []).append(r["DWA Element Name"])
    unit = next((u for u in _load("closure_table_units_753.csv") if u["unit"] == soc), {})
    native = next((n for n in _load("closure_native_soc2018_codes.csv") if n["soc2018"] == soc), {})
    return {"soc": soc, "title": tasks[0]["occ_title"], "tasks": tasks, "dwas": by_task, "unit": unit, "native": native}


def _f(x, d):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


# ---------------------------------------------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------------------------------------------
def draft_rulebook(provider, inputs: Dict[str, object]) -> Dict[str, object]:
    if provider.is_mock:
        fam = "MOCK_" + re.sub(r"[^A-Z]", "", inputs["title"].upper())[:8]
        d = json.loads(json.dumps(EXAMPLE))
        d["family"] = fam
        for k in ("tier1_rule", "breach_rule", "base_rule", "tier2_rule", "tier3_rule"):
            d[k][0] = d[k][0].replace("CPA_", fam + "_")
        d["rationale"] = "mock: the accounting example relabelled; protocol test only"
        return d
    lines = [f"OCCUPATION: {inputs['title']} (SOC {inputs['soc']})"]
    for t in sorted(inputs["tasks"], key=lambda r: ["Inner_Core", "Middle_Buffer", "Outer_Periphery"].index(r["ring"])):
        lines.append(f"\n{t['ring']} task (O*NET importance {t['importance_IM']}, task exposure {t['task_exposure_e_g_2025']}):\n  {t['task']}")
        for a in inputs["dwas"].get(t["task_id_onet"], [])[:6]:
            lines.append(f"    - {a}")
    user = ("\n".join(lines) + "\n\nWORKED EXAMPLE (accounting):\n" + json.dumps(EXAMPLE, indent=1) +
            "\n\nDraft the rulebook for this occupation's task family as a JSON object of the same shape.")
    text = provider.complete("rulebook", SYSTEM, user, RULEBOOK_SCHEMA)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {"error": "unparseable", "raw": text[:500]}


# ---------------------------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------------------------
ALLOWED_PUBLIC = {"V1", "V2", "V3", "V4", "SEV", "ANNEX_SEVERITY", "PRIMARY_CUTOFF", "RATIO_CUTOFF"}


def _idents(expr: str) -> set:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", expr)) - {"IF", "ELSE", "MIN", "MAX"}


def _sample_cases(n: int = 60) -> List[Tuple[float, float, float, float, float]]:
    out = []
    for i in range(n):
        h = hashlib.sha256(f"occ_compiler_case_{i}".encode()).hexdigest()
        u = [int(h[k * 8:(k + 1) * 8], 16) / 0xFFFFFFFF for k in range(5)]
        out.append((100 + 800 * u[0], 0.02 + 0.23 * u[1], 20 + 100 * u[2], 100 + 400 * u[3], u[4]))
    return out


def validate(draft: Dict[str, object], profile: str = "g1_template") -> Tuple[bool, List[str], Optional[DomainRulebook], Dict[str, float]]:
    """Deterministic validation of a drafted rulebook.

    profile "g1_template" (the configuration used for the shipped drafts) also requires the precedent-family form:
    a tier-3 LET coupling SEV with V2 and a tier-2 rule piecewise in V2 that involves SEV, V3 and V4. profile
    "structural" checks only grammar, identifiers, registry shape, finiteness and tier separation, so that a domain
    whose hidden rule has a different functional form is not rejected for failing to imitate G1."""
    if profile not in ("g1_template", "structural"):
        raise ValueError(f"unknown validator profile {profile}")
    problems: List[str] = []
    if "error" in draft:
        return False, [f"no draft: {draft.get('error')}"], None, {}
    fam = str(draft.get("family", "")).strip()
    if not re.fullmatch(r"[A-Z][A-Z_]{1,24}", fam):
        problems.append(f"family name '{fam}' is not upper-case letters and underscores")
    if fam in dc.RULEBOOKS:
        problems.append(f"family {fam} already exists")
    try:
        rb = DomainRulebook(fam, [tuple(x) for x in draft["lets"]], tuple(draft["tier1_rule"]), tuple(draft["breach_rule"]),
                            tuple(draft["base_rule"]), tuple(draft["tier2_rule"]), [tuple(x) for x in draft["tier3_lets"]], tuple(draft["tier3_rule"]))
    except Exception as e:
        return False, [f"shape: {e}"], None, {}
    raw = draft.get("registry", {})
    try:
        reg = {str(e["name"]): float(e["value"]) for e in raw} if isinstance(raw, list) else {str(k): float(v) for k, v in raw.items()}
    except (KeyError, TypeError, ValueError) as e:
        return False, [f"registry shape: {e}"], None, {}
    if len(reg) != 3:
        problems.append(f"registry holds {len(reg)} constants, need exactly 3")
    for k, v in reg.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", k):
            problems.append(f"registry key '{k}' is not snake_case")
        if not (0.0 < v < 5.0):
            problems.append(f"registry constant {k} = {v} outside (0, 5)")
    # expressions parse and use allowed identifiers
    let_names = []
    for name, ex in rb.lets:
        try:
            _compile_expr(ex)
        except Exception as e:
            problems.append(f"LET {name}: does not parse ({e})")
        bad = _idents(ex) - ALLOWED_PUBLIC - set(let_names)
        if bad:
            problems.append(f"LET {name}: uses {sorted(bad)}")
        let_names.append(name)
    labels = []
    for key in ("tier1_rule", "breach_rule", "base_rule", "tier2_rule"):
        label, ex = getattr(rb, key)
        labels.append(label)
        try:
            _compile_expr(ex)
        except Exception as e:
            problems.append(f"{key}: does not parse ({e})")
        bad = _idents(ex) - ALLOWED_PUBLIC - set(let_names)
        if bad:
            problems.append(f"{key}: uses {sorted(bad)}")
        if not label.startswith(fam + "_") or not re.fullmatch(r"[A-Z0-9_]+", label):
            problems.append(f"{key}: label '{label}' is not an upper-case label prefixed by {fam}_")
    t3_names = []
    for name, ex in rb.tier3_lets:
        try:
            _compile_expr(ex)
        except Exception as e:
            problems.append(f"tier3 LET {name}: does not parse ({e})")
        t3_names.append(name)
    label3, ex3 = rb.tier3_rule
    labels.append(label3)
    try:
        _compile_expr(ex3)
    except Exception as e:
        problems.append(f"tier3_rule: does not parse ({e})")
    if len(set(labels)) != len(labels):
        problems.append("labels are not distinct")
    reg_used = set(re.findall(r"REG\.([a-z0-9_]+)", ex3 + " " + " ".join(e for _, e in rb.tier3_lets)))
    if len(reg_used) < 2:
        problems.append("tier3 uses fewer than two registry constants")
    if not reg_used <= set(reg):
        problems.append(f"tier3 refers to unknown registry constants {sorted(reg_used - set(reg))}")
    if profile == "g1_template":
        if not any("SEV" in e and "V2" in e for _, e in rb.tier3_lets):
            problems.append("no tier3 LET couples SEV with V2")
        # convention: piecewise in V2 with a breakpoint that is not a cutoff
        m = re.search(r"\(\s*([0-9.]+)\s+IF\s+V2\s*>=\s*([0-9.]+)\s+ELSE\s+([0-9.]+)\s*\)", rb.tier2_rule[1])
        if not m:
            problems.append("tier2_rule has no `(a IF V2 >= b ELSE c)` piece")
        else:
            bp = float(m.group(2))
            if not (0.08 <= bp <= 0.20):
                problems.append(f"tier2 breakpoint {bp} outside [0.08, 0.20]")
            if float(m.group(1)) == float(m.group(3)):
                problems.append("tier2 branches are identical")
        for v in ("SEV", "V3", "V4"):
            if v not in rb.tier2_rule[1]:
                problems.append(f"tier2_rule does not involve {v}")
    if problems:
        return False, problems, rb, reg
    # numeric behaviour on sampled cases
    env_reg = {"REG_" + k: v for k, v in reg.items()}
    dist = {"tier2_vs_public": 0, "tier3_vs_public": 0, "breach_vs_base": 0}
    n = 0
    for v1, v2, v3, v4, sev in _sample_cases():
        env = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "SEV": sev, "ANNEX_SEVERITY": sev, "PRIMARY_CUTOFF": 500.0, "RATIO_CUTOFF": 0.10, **env_reg}
        try:
            for name, ex in rb.lets:
                env[name] = _ev(_compile_expr(ex), env)
            vals = {k: _ev(_compile_expr(getattr(rb, k)[1]), env) for k in ("tier1_rule", "breach_rule", "base_rule", "tier2_rule")}
            for name, ex in rb.tier3_lets:
                env[name] = _ev(_compile_expr(ex), env)
            vals["tier3_rule"] = _ev(_compile_expr(rb.tier3_rule[1]), env)
        except Exception as e:
            problems.append(f"evaluation failed: {e}")
            break
        for k, v in vals.items():
            if not math.isfinite(v) or abs(v) > 1e5:
                problems.append(f"{k}: treatment {v:.3g} not finite and bounded on a typical case")
                break
        pub = vals["breach_rule"] if v2 >= 0.10 else vals["base_rule"]
        dist["tier2_vs_public"] += abs(vals["tier2_rule"] - pub) > 0.011
        dist["tier3_vs_public"] += abs(vals["tier3_rule"] - pub) > 0.011
        dist["breach_vs_base"] += abs(vals["breach_rule"] - vals["base_rule"]) > 0.011
        n += 1
        if problems:
            break
    if n:
        for k, c in dist.items():
            if c / n < 0.9:
                problems.append(f"{k}: tiers coincide on {100 * (1 - c / n):.0f}% of sampled cases")
    return (not problems), problems, rb, reg


# ---------------------------------------------------------------------------------------------------------------
# Instantiation and evaluation
# ---------------------------------------------------------------------------------------------------------------
def ring_defaults() -> Dict[str, Dict[str, float]]:
    """Ring means of the registered corpus for the design attributes a new occupation has no source for."""
    out = {}
    for ring in ("Inner_Core", "Middle_Buffer", "Outer_Periphery"):
        rr = [s for s in FULL_EXTENDED_CORPUS if s.ring == ring]
        out[ring] = {k: sum(getattr(s, k) for s in rr) / len(rr) for k in
                     ("overlap_j", "n_subclauses", "collision_density", "tier2_piecewise_share", "tier3_deep_share", "silent_collision_prob")}
        out[ring]["n_subclauses"] = int(round(out[ring]["n_subclauses"]))
    return out


def instantiate(inputs: Dict[str, object], rb: DomainRulebook, reg: Dict[str, float]) -> List[ONETTaskSpec]:
    fam = rb.family
    dc.RULEBOOKS[fam] = rb
    dc.PRECEDENT_REGISTRY[fam] = dict(reg, silent_ratio_gate=0.13, silent_metric_floor=0.50)
    unit, native = inputs["unit"], inputs["native"]
    jz = int(_f(unit.get("job_zone"), 4))
    lic = _f(unit.get("lic_cps_1519"), 0.1)
    w = _f(native.get("wage_ratio_to_all_occ_median_2024"), None) or _f(native.get("wage_ratio_to_all_occ_median_2019"), 2.0)
    z = _f(native.get("accountability_z_native22"), 0.0)
    signoff = int(_f(native.get("signoff_handcoded"), 0) >= 0.5)
    h = int(hashlib.sha256(inputs["soc"].encode()).hexdigest()[:8], 16)
    primary_cutoff = round(450 + (h % 120), 1)
    ratio_cutoff = round(0.090 + ((h >> 8) % 30) / 1000.0, 4)
    code = re.sub(r"[^A-Z]", "", inputs["title"].upper())[:3]
    defaults = ring_defaults()
    specs = []
    for t in inputs["tasks"]:
        ring = t["ring"]; d = defaults[ring]
        suffix = {"Inner_Core": "C1", "Middle_Buffer": "B2", "Outer_Periphery": "P3"}[ring]
        specs.append(ONETTaskSpec(inputs["soc"], inputs["title"].replace(" ", "_").replace(",", ""), jz,
                                  "Sourced_" + ("Signoff" if signoff else "NoSignoff"), f"{code}-{suffix}", ring, t["task"],
                                  "; ".join(inputs["dwas"].get(t["task_id_onet"], [])[:3]) or t["task"],
                                  round(d["overlap_j"], 2), float(t["task_importance_wt"]), signoff if ring != "Outer_Periphery" else 0,
                                  round(lic, 2), round(w, 2), round(z, 2), d["n_subclauses"], round(d["collision_density"], 2),
                                  round(d["tier2_piecewise_share"], 2), round(d["tier3_deep_share"], 2), round(d["silent_collision_prob"], 2),
                                  fam, primary_cutoff, ratio_cutoff))
    return specs


def _draft_dir(provider: str, run_id: str) -> str:
    """Drafts are namespaced by provider and run and never overwritten. The five drafts shipped at the top of
    data_inputs/drafted_rulebooks/ are the legacy Anthropic run and are only ever read."""
    return os.path.join(DRAFTS, provider, run_id)


def _write_immutable(path: str, rec: Dict[str, object]) -> None:
    if os.path.exists(path):
        raise SystemExit(f"{os.path.relpath(path, BASE)} exists; drafts are immutable, choose a new --run-id")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x") as f:
        json.dump(rec, f, indent=1)


def evaluate_draft(soc: str, draft: Dict[str, object], provider_name: str, profile: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    inputs = occupation_inputs(soc)
    ok, problems, rb, reg = validate(draft, profile)
    rec = {"soc2018": soc, "occ_title": inputs["title"], "provider": provider_name, "family": draft.get("family", ""),
           "validator_profile": profile, "validated": ok, "problems": problems, "validated_by": "", "draft": draft}
    print(f"\n{soc} {inputs['title']}: family {draft.get('family', '?')}: {'VALID' if ok else 'REJECTED'}" + ("" if ok else f" ({'; '.join(problems)})"), flush=True)
    if ok:
        from run_live_benchmark import run_corpus_evaluation
        specs = instantiate(inputs, rb, reg)
        tasks, traces, logs, winners = run_corpus_evaluation(specs, f"NEW_{soc}", os.path.join(TOOLS_DIR, soc.replace("-", "_")))
        for t in tasks:
            rows.append({"soc2018": soc, "occ_title": inputs["title"], "family": rb.family, "task_id": t["task_id"], "ring": t["ring"],
                         **{k: t[k] for k in ("A_Base_Pct", "A_Tool_v1_Pct", "A_Tool_v2_Pct", "A_Tool_v3_Pct", "A_Inc_Solo_Pct", "A_Collab_Pct",
                                              "Direct_Poaching_Pct", "Accountability_Settlement_Pct", "Incumbent_Gatekeeping_Pct", "SQ_Error_Rate_Pct", "CF1_Dereg_Error_Pct")},
                         "v2_handler_template": t["v2_handler_template"], "v3_split": t["v3_split"]})
        rec["evaluation"] = [r for r in rows if r["soc2018"] == soc]
    return rec


def main():
    ap = argparse.ArgumentParser(description="Drafts, validates and evaluates rulebooks for new occupations.")
    ap.add_argument("--provider", choices=["mock", "anthropic", "saved"], default="mock",
                    help="saved = re-validate and re-evaluate saved drafts without any model call")
    ap.add_argument("--occupations", default="29-1051", help="comma-separated SOC-2018 codes present in proposed_corpus_tasks.csv")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--run-id", default="", help="namespace for new drafts (default: provider-specific timestamp)")
    ap.add_argument("--draft-dir", default=DRAFTS, help="directory of saved drafts for --provider saved (default: the legacy shipped run)")
    ap.add_argument("--validator-profile", choices=["g1_template", "structural"], default="g1_template")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    socs = [s.strip() for s in args.occupations.split(",") if s.strip()]
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    if args.provider == "anthropic":
        if not args.yes:
            print(f"about to make {len(socs)} model calls; pass --yes to proceed"); return
        provider = AnthropicProvider(LLMConfig(model=args.model, transcript_path=os.path.join(OUT_DIR, f"llm_transcripts_rulebooks_{run_id}.jsonl")))
    elif args.provider == "mock":
        provider = MockProvider()
    else:
        provider = None
    os.makedirs(TOOLS_DIR, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for soc in socs:
        if provider is None:
            path = os.path.join(args.draft_dir, f"{soc}.json")
            with open(path) as f:
                saved = json.load(f)
            rec = evaluate_draft(soc, saved["draft"], f"saved:{saved.get('provider', '?')}", args.validator_profile, rows)
            rec["source_draft"] = os.path.relpath(path, BASE)
            continue  # saved drafts are read-only; results go to the CSV below
        draft = draft_rulebook(provider, occupation_inputs(soc))
        rec = evaluate_draft(soc, draft, args.provider, args.validator_profile, rows)
        rec["run_id"] = run_id
        _write_immutable(os.path.join(_draft_dir(args.provider, run_id), f"{soc}.json"), rec)
    shutil.rmtree(TOOLS_DIR, ignore_errors=True)
    if rows:
        name = "saved_drafts" if provider is None else f"{args.provider}_{run_id}"
        out = os.path.join(OUT_DIR, f"occupation_compiler_{name}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("\n" + "=" * 128)
        print(f"TABLE 13: NEW OCCUPATIONS INSTANTIATED FROM MODEL-DRAFTED RULEBOOKS ({name}); registered pipeline, 20 packets per task")
        print("=" * 128)
        print(f"{'Task':<8} {'Occupation':<30} {'Ring':<16} | {'Novice':>6} | {'v1':>5} | {'v2':>5} | {'v3':>5} | {'Inc':>5} | {'Settle':>6} | {'Poach':>5} | {'Settl':>5} | {'Gate':>5} | {'SQerr':>5} | {'CF1err':>6}")
        print("-" * 128)
        for r in rows:
            print(f"{r['task_id']:<8} {r['occ_title'][:30]:<30} {r['ring']:<16} | {float(r['A_Base_Pct']):5.0f}% | {float(r['A_Tool_v1_Pct']):4.0f}% | {float(r['A_Tool_v2_Pct']):4.0f}% | {float(r['A_Tool_v3_Pct']):4.0f}% | "
                  f"{float(r['A_Inc_Solo_Pct']):4.0f}% | {float(r['A_Collab_Pct']):5.0f}% | {float(r['Direct_Poaching_Pct']):4.0f}% | {float(r['Accountability_Settlement_Pct']):4.0f}% | {float(r['Incumbent_Gatekeeping_Pct']):4.0f}% | {float(r['SQ_Error_Rate_Pct']):4.1f}% | {float(r['CF1_Dereg_Error_Pct']):5.1f}%")
        print("-" * 128)
        print(f"Output: {os.path.relpath(out, BASE)}. `validated_by` is empty until a practitioner signs off.")
    if args.provider == "anthropic":
        print(f"model calls: {provider.calls}; tokens in/out: {provider.usage}")


if __name__ == "__main__":
    main()
