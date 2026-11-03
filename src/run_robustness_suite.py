#!/usr/bin/env python3
"""
Robustness & Sensitivity Suite (run_robustness_suite.py) — JURI Lab

Re-runs the full laboratory under alternative configurations and records the headline cells of each run:
Table 1 end-to-end rates, Tier 2 compiler rates, Tier 3 incumbent/collab rates, ring-level governance and
status-quo error, and the compiler's template winners. Blocks:

  A. Monte Carlo over packet universes (seed_salt r01..r10): mean and SD of every headline cell.
  B. Archive-size learning curve (20, 40, 80, 160, 320 draws per task).
  C. Compiler hypothesis class: single-template grammar vs full grammar; coefficient grid 1 / 2 / 3 decimals.
  D. Institutional comparative statics: liability scale, statutory penalty scale, settlement review fraction.
  E. Behavioural thresholds: solo-slip and audit-depth noise limits, critic u_p scale, manual override arithmetic.
  F. Design variants: decoy supplemental notes, silent gate not sufficient, 100 evaluation packets per task.

Outputs: data_outputs/robustness_summary.csv (one row per run) and a printed digest.
Standard library only. Each run takes a few seconds; the whole suite takes several minutes.
"""

import csv
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

from model_config import CFG, set_config, reset_config, config_diff
from run_live_benchmark import run_full, BASE_DIR, OUT_DIR

SCRATCH_TOOLS = os.path.join(BASE_DIR, "robustness_tools")


def headline(summary: Dict[str, object]) -> Dict[str, object]:
    t1 = summary["table1"]
    allrow = t1["3. All Multi-Doc Packets E2E"]
    t2 = t1["2c. Tier 2: 1st-Order Piecewise Collision"]
    t3 = t1["2d. Tier 3: 2nd-Order Precedent Collision"]
    rings = summary["rings"]
    h = {
        "E2E_Novice": allrow["A_Base"], "E2E_Tool_v1": allrow["A_Tool_v1"], "E2E_Tool_v2": allrow["A_Tool_v2"], "E2E_Tool_v3": allrow["A_Tool_v3"],
        "E2E_Inc_Solo": allrow["A_Inc_Solo"], "E2E_Collab": allrow["A_Collab"],
        "Tier2_Tool_v2": t2["A_Tool_v2"], "Tier2_Tool_v3": t2["A_Tool_v3"], "Tier3_Inc_Solo": t3["A_Inc_Solo"], "Tier3_Collab": t3["A_Collab"],
    }
    for ring, short in (("Outer_Periphery", "Per"), ("Middle_Buffer", "Buf"), ("Inner_Core", "Core")):
        h[f"{short}_Poach"] = rings[ring]["Direct_Poaching_Pct"]
        h[f"{short}_Settle"] = rings[ring]["Accountability_Settlement_Pct"]
        h[f"{short}_Gate"] = rings[ring]["Incumbent_Gatekeeping_Pct"]
        h[f"{short}_SQErr"] = rings[ring]["SQ_Error_Rate_Pct"]
        h[f"{short}_CF2Err"] = rings[ring]["CF2_NoLiab_Error_Pct"]
    h["ISR_beta_auto"] = summary["table4"].get("macro_auto_delta_19_26", {}).get("isr_beta", float("nan"))
    h["n_v2_handlers"] = summary["compiler"]["n_v2"]
    h["n_v3_splits"] = summary["compiler"]["n_v3"]
    h["template_winners"] = ";".join(f"{k}:{v}" for k, v in sorted(summary["compiler"]["winners"].items()))
    return h


def run_one(block: str, name: str, overrides: Dict[str, object]) -> Dict[str, object]:
    reset_config()
    set_config(**overrides)
    tools = os.path.join(SCRATCH_TOOLS, name)
    t0 = time.time()
    summ = run_full(tools_dir=tools, out_dir=None, quiet=True)
    row = {"block": block, "run": name, "overrides": str(config_diff()), "seconds": round(time.time() - t0, 1)}
    row.update(headline(summ))
    reset_config()
    shutil.rmtree(tools, ignore_errors=True)
    print(f"  [{block}] {name:<28} {row['seconds']:5.1f}s  E2E v3={row['E2E_Tool_v3']:5.1f}  Tier2 v2/v3={row['Tier2_Tool_v2']:5.1f}/{row['Tier2_Tool_v3']:5.1f}  "
          f"IncSolo={row['E2E_Inc_Solo']:5.1f}  Collab={row['E2E_Collab']:5.1f}  Core poach/settle/gate={row['Core_Poach']:4.1f}/{row['Core_Settle']:4.1f}/{row['Core_Gate']:4.1f}  CoreSQErr={row['Core_SQErr']:4.1f}", flush=True)
    return row


def experiments() -> List[Tuple[str, str, Dict[str, object]]]:
    ex: List[Tuple[str, str, Dict[str, object]]] = [("base", "registered_design", {})]
    ex += [("A_montecarlo", f"universe_r{i:02d}", {"seed_salt": f"r{i:02d}"}) for i in range(1, 11)]
    ex += [("B_archive", f"archive_{n}", {"archive_size": n}) for n in (20, 40, 80, 160, 320)]
    ex += [("C_grammar", "grammar_G1_only", {"grammar": "G1"}),
           ("C_grammar", "grammar_without_G1_reparam", {"grammar": "no-G1-reparam"}),
           ("C_grammar", "grammar_without_G1_family", {"grammar": "no-G1-family"}),
           ("C_grammar", "selection_train_validation", {"selection_validation_fraction": 0.25}),
           ("C_grammar", "coef_grid_1dp", {"coef_decimals": 1}),
           ("C_grammar", "coef_grid_3dp", {"coef_decimals": 3}),
           ("C_grammar", "triple_sample_12", {"triple_sample": 12})]
    ex += [("D_institutions", f"liability_x{m}", {"liability_scale": 5.4 * m}) for m in (0.5, 2.0)]
    ex += [("D_institutions", f"penalty_x{m}", {"penalty_base": 35.0 * m, "penalty_per_lic": 15.0 * m}) for m in (0.0, 0.5, 2.0)]
    ex += [("D_institutions", f"review_frac_{b}", {"review_fraction_base": b}) for b in (0.15, 0.35)]
    ex += [("E_behaviour", f"solo_slip_base_{b}", {"solo_slip_base": b}) for b in (595.0, 675.0)]
    ex += [("E_behaviour", f"audit_depth_base_{b}", {"audit_depth_base": b}) for b in (430.0, 510.0)]
    ex += [("E_behaviour", f"u_p_scale_x{m}", {"u_p_clean": 0.03 * m, "u_p_override": 0.12 * m, "u_p_learned": 0.19 * m, "u_p_unparsed": min(0.99, 0.86 * m)}) for m in (0.5, 1.5)]
    ex += [("E_behaviour", "override_manual", {"override_arithmetic": "manual"}),
           ("E_behaviour", "u_p_clean_realised_0.064", {"u_p_clean": 0.064})]   # critic's clean-run doubt set to the realised core error
    ex += [("F_design", f"decoy_notes_{r}", {"decoy_note_rate": r}) for r in (0.1, 0.25)]
    ex += [("F_design", "silent_gate_insufficient", {"silent_gate_sufficient": False}),
           ("F_design", "eval_100_per_task", {"n_eval_packets": 100})]
    return ex


def main(only_blocks: List[str] = None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for block, name, ov in experiments():
        if only_blocks and block != "base" and block[0] not in only_blocks:
            continue
        rows.append(run_one(block, name, ov))
    path = os.path.join(OUT_DIR, "robustness_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Monte Carlo digest
    mc = [r for r in rows if r["block"] == "A_montecarlo"]
    if mc:
        print("\nMONTE CARLO OVER PACKET UNIVERSES (n = %d): mean +/- SD of headline cells" % len(mc))
        for k in ("E2E_Novice", "E2E_Tool_v1", "E2E_Tool_v2", "E2E_Tool_v3", "E2E_Inc_Solo", "E2E_Collab", "Tier2_Tool_v2", "Tier2_Tool_v3",
                  "Per_Poach", "Buf_Poach", "Core_Poach", "Buf_Settle", "Core_Settle", "Core_Gate", "Core_SQErr", "Core_CF2Err", "ISR_beta_auto"):
            vals = [float(r[k]) for r in mc]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
            base = float(rows[0][k])
            print(f"  {k:<16} registered = {base:7.2f}   MC mean = {mu:7.2f}   SD = {sd:5.2f}   range = [{min(vals):.1f}, {max(vals):.1f}]")
    print(f"\nwrote {os.path.relpath(path, BASE_DIR)} ({len(rows)} runs)")
    shutil.rmtree(SCRATCH_TOOLS, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=f"Robustness suite: reruns the laboratory {len(experiments())} times (reference configuration plus {len(experiments()) - 1} alternatives), one change at a time, and writes data_outputs/robustness_summary.csv")
    ap.add_argument("blocks", nargs="*", help="optional block letters to run (A-F); default: all blocks")
    args = ap.parse_args()
    valid = {"A", "B", "C", "D", "E", "F"}
    chosen = [b.upper() for b in args.blocks]
    bad = [b for b in chosen if b not in valid]
    if bad:
        ap.error(f"unknown block(s): {bad}; choose from {sorted(valid)}")
    main(chosen or None)
