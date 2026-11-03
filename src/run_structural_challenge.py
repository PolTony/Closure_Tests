#!/usr/bin/env python3
"""
Held-out structural challenge: synthetic rule families whose Tier 2 convention is not of the G1 form.

The precedent conventions of the shipped corpus all belong to the G1 family B(1 - aS) + bX + cY, and the compiler's
grammar was written with that family in view. This script checks the compiler on two synthetic families that were
not used to design the grammar or the drafting examples:

  SYNTH_ADDITIVE  Tier 2 is additive in severity, B + a(V2) S + bX + cY: inside the declared grammar (G2) but not G1.
  SYNTH_PRODUCT   Tier 2 multiplies severity with a field ratio, B(1 - a(V2) S)(1 + b V3 / V4): outside every template.

Both are synthetic fixtures. They borrow the LEGAL task parameters, handbook quantities and private registry, so the
only change is the functional form of the archived convention. They are not new occupations and carry no claim about
any real profession. Each is run under the registered grammar ("full") and the true family ablation ("no-G1-family").

Output: data_outputs/structural_challenge.csv (one row per family x grammar x ring task).
"""
import argparse
import csv
import dataclasses
import os
import juri_paths
import shutil

import domain_cases as dc
from model_config import set_config, reset_config

BASE = juri_paths.ROOT
OUT = os.path.join(BASE, "data_outputs")
TOOLS = os.path.join(BASE, "structural_challenge_tools")

_LEGAL = dc.RULEBOOKS["LEGAL"]
FAMILIES = {
    "SYNTH_ADDITIVE": "net_claim + (95 IF V2 >= 0.12 ELSE 60) * SEV - 1.10 * V3 + 0.20 * V4",
    "SYNTH_PRODUCT": "net_claim * (1 - (0.50 IF V2 >= 0.12 ELSE 0.30) * SEV) * (1 + 0.40 * V3 / (V3 + V4))",
}


def register(family: str, expr: str) -> None:
    rb = dataclasses.replace(_LEGAL, family=family, tier1_rule=(f"{family}_EXPLICIT_OVERRIDE", _LEGAL.tier1_rule[1]),
                             breach_rule=(f"{family}_BREACH", _LEGAL.breach_rule[1]), base_rule=(f"{family}_BASE", _LEGAL.base_rule[1]),
                             tier2_rule=(f"{family}_CONVENTION", expr), tier3_rule=(f"{family}_PRECEDENT", _LEGAL.tier3_rule[1]))
    dc.RULEBOOKS[family] = rb
    dc.PRECEDENT_REGISTRY[family] = dict(dc.PRECEDENT_REGISTRY["LEGAL"])


def specs_for(family: str):
    base = [s for s in dc.PRIMARY_CORPUS if s.domain_family == "LEGAL"]
    code = family.split("_")[1][:3]
    return [dataclasses.replace(s, domain_family=family, occ_title=f"Synthetic_{family}", task_id=f"{code}-{s.task_id}") for s in base]


def main() -> None:
    argparse.ArgumentParser(description="Held-out structural challenge: synthetic non-G1 rule families").parse_args()
    from run_live_benchmark import run_corpus_evaluation
    rows = []
    for family, expr in FAMILIES.items():
        register(family, expr)
        for grammar in ("full", "no-G1-family"):
            set_config(grammar=grammar)
            try:
                tasks, traces, logs, winners = run_corpus_evaluation(specs_for(family), f"CHALLENGE_{family}", os.path.join(TOOLS, f"{family}_{grammar}"))
            finally:
                reset_config()
            for t in tasks:
                convention = [p for p in traces if p["task_id"] == t["task_id"] and p["complexity_tier"] == 2]
                rows.append({"family": family, "tier2_form": expr, "grammar": grammar, "task_id": t["task_id"], "ring": t["ring"],
                             "tier2_n": len(convention), "tier2_v3_correct": sum(p["A_Tool_v3_e2e"] for p in convention),
                             "v2_handler_template": t["v2_handler_template"], "v3_split": t["v3_split"],
                             **{k: t[k] for k in ("A_Base_Pct", "A_Tool_v1_Pct", "A_Tool_v2_Pct", "A_Tool_v3_Pct", "A_Inc_Solo_Pct")}})
    shutil.rmtree(TOOLS, ignore_errors=True)
    path = os.path.join(OUT, "structural_challenge.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'Family':<16} {'Grammar':<13} {'Task':<8} {'Ring':<16} | {'v1':>5} | {'v2':>5} | {'v3':>5} | {'Inc':>5} | template")
    for r in rows:
        print(f"{r['family']:<16} {r['grammar']:<13} {r['task_id']:<8} {r['ring']:<16} | {float(r['A_Tool_v1_Pct']):4.0f}% | {float(r['A_Tool_v2_Pct']):4.0f}% | "
              f"{float(r['A_Tool_v3_Pct']):4.0f}% | {float(r['A_Inc_Solo_Pct']):4.0f}% | {r['v2_handler_template']}")
    print(f"wrote {os.path.relpath(path, BASE)}; synthetic fixtures, not occupations")


if __name__ == "__main__":
    main()
