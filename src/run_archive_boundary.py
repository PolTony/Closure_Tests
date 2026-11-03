#!/usr/bin/env python3
"""
Archive-boundary experiment (run_archive_boundary.py) — JURI Lab

Is the precedent tier uncompilable because it is withheld from the public archive, because the compiler's
hypothesis class cannot express it, or because its constants are not on a published grid? The reference configuration
withholds Tier 3 from the archive. This script archives it and varies the compiler's grammar and coefficient
grid, holding everything else fixed.

Reading of the registered result: the ceiling is set by three things together, what is archived, what the grammar
can express, and whether the rule's constants lie on the coefficient grid the compiler assumes schedules are
published on. Silent collisions remain uncompilable in every condition because their field is not structured.
"""
import os
import juri_paths
import shutil
import sys

from model_config import set_config, reset_config
from run_live_benchmark import run_full
from domain_cases import FULL_EXTENDED_CORPUS, build_multi_doc_packets_for_task
from compiler_agent import load_source_exactly

BASE = juri_paths.ROOT
SCRATCH = os.path.join(BASE, "archive_boundary_tools")

CONDITIONS = [
    ("registered: Tier 3 withheld, grammar full, 2-dp grid", {}),
    ("Tier 3 archived, grammar full (no interaction template), 2-dp", {"archive_includes_tier3": True}),
    ("Tier 3 archived, grammar full+G5, 2-dp grid", {"archive_includes_tier3": True, "grammar": "full+G5"}),
    ("Tier 3 archived, grammar full+G5, 4-dp grid", {"archive_includes_tier3": True, "grammar": "full+G5", "coef_decimals": 4}),
    ("Tier 3 archived, full+G5, 4-dp, archive 320", {"archive_includes_tier3": True, "grammar": "full+G5", "coef_decimals": 4, "archive_size": 320}),
]


def main():
    print("=" * 128)
    print("ARCHIVE-BOUNDARY EXPERIMENT (Tier 3 = 190 evaluation packets, of which 42 silent)")
    print("=" * 128)
    print(f"{'Condition':<58} | {'Tier2 v3':>8} | {'Tier3 v2':>8} | {'Tier3 v3':>8} | {'E2E v3':>7} | {'handlers':>8} | winners")
    print("-" * 128)
    rows = []
    for label, ov in CONDITIONS:
        reset_config()
        set_config(**ov)
        summ = run_full(tools_dir=SCRATCH, out_dir=None, quiet=True)
        overt, silent = [], []
        for spec in FULL_EXTENDED_CORPUS:
            path = os.path.join(SCRATCH, f"tool_{spec.task_id.replace('-', '_')}_v3.py")
            with open(path) as f:
                fn = load_source_exactly(f.read(), path).execute_compiled_routine
            for p in build_multi_doc_packets_for_task(spec):
                if p.complexity_tier == 3:
                    out = fn(p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex)
                    correct = out["diagnosis"] == p.gt_diagnosis and out["inference"] == p.gt_inference and abs(out["treatment"] - p.gt_treatment) <= 0.01
                    (silent if p.is_silent_collision else overt).append(int(correct))
        t = summ["table1"]
        print(f"{label:<58} | {t['2c. Tier 2: 1st-Order Piecewise Collision']['A_Tool_v3']:7.1f}% | "
              f"{t['2d. Tier 3: 2nd-Order Precedent Collision']['A_Tool_v2']:7.1f}% | {t['2d. Tier 3: 2nd-Order Precedent Collision']['A_Tool_v3']:7.1f}% | "
              f"{t['3. All Multi-Doc Packets E2E']['A_Tool_v3']:6.1f}% | {summ['compiler']['n_v2']:>8} | {summ['compiler']['winners']}", flush=True)
        rows.append({"condition": label, "overrides": repr(ov),
                     "tier3_overt_n": len(overt), "tier3_overt_v3_correct": sum(overt),
                     "tier3_silent_n": len(silent), "tier3_silent_v3_correct": sum(silent),
                     "tier2_v3": round(t['2c. Tier 2: 1st-Order Piecewise Collision']['A_Tool_v3'], 1),
                     "tier3_v2": round(t['2d. Tier 3: 2nd-Order Precedent Collision']['A_Tool_v2'], 1),
                     "tier3_v3": round(t['2d. Tier 3: 2nd-Order Precedent Collision']['A_Tool_v3'], 1),
                     "e2e_v3": round(t['3. All Multi-Doc Packets E2E']['A_Tool_v3'], 1),
                     "handlers": summ['compiler']['n_v2'], "winners": str(summ['compiler']['winners'])})
        shutil.rmtree(SCRATCH, ignore_errors=True)
    reset_config()
    import csv
    out = os.path.join(BASE, "data_outputs", "archive_boundary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {os.path.relpath(out, BASE)}")
    print("=" * 128)


if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(description='Archive-boundary experiment: archives the precedent tier and varies the compiler grammar and grid (appendix section 7)').parse_args()   # no options; rejects stray arguments and serves --help
    main()
