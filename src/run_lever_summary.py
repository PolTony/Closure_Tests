#!/usr/bin/env python3
"""
Lever summary (run_lever_summary.py) — JURI Lab, Tables 17 and 18

Reads the dynamics written with the two levers added in the final build phase and prints

    Table 17  the fee-setting incumbent (`run_dynamics.py --incumbent-fee`): the review fee charged on settled work by
              ring and period under each disclosure regime, the excess review charge over the registered fee (not economic rent; review time held fixed), how the client's
              arrangement choice responds, and the incumbent's revenue against the registered-fee run;
    Table 18  the labour block when the program replaces junior hours (`--set labor_junior_operates_tool=false`):
              junior demand, hiring, apprenticeship and competence at promotion against the registered assumption
              that juniors operate the program.

Every number is recomputed from data_outputs/ so the tables in README.md and the appendix can be checked here.

    python3 run_lever_summary.py
"""

import argparse
import csv
import os
import juri_paths
import statistics as st
from typing import Dict, List

BASE = juri_paths.ROOT
OUT = os.path.join(BASE, "data_outputs")
RINGS = ("Outer_Periphery", "Middle_Buffer", "Inner_Core")


def load(name: str) -> List[Dict[str, str]]:
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def fl(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def m(rows, key) -> float:
    vals = [fl(r[key]) for r in rows if fl(r[key]) == fl(r[key])]
    return st.mean(vals) if vals else float("nan")


def table17() -> None:
    regs = [r for r in ("mandatory", "discretion", "strategic") if os.path.exists(os.path.join(OUT, f"dynamics_{r}_fee.csv"))]
    if not regs:
        print("Table 17: no fee runs found"); return
    print("=" * 160)
    print("TABLE 17: THE FEE-SETTING INCUMBENT — fee charged on settled work, by regime, ring and period; the registered-fee run of the same regime alongside")
    print("=" * 160)
    print(f"{'Regime':<11} | {'Ring':<10} | {'t':>2} | {'Fee':>6} | {'Excess':>6} | {'at max':>6} | {'at min':>6} | {'Settle% fee/reg':>15} | {'Gate% fee/reg':>13} | {'Poach% fee/reg':>14} | {'Inc rev fee/reg':>15} | {'T2 parity fee/reg':>17} | {'SQ err fee/reg':>14}")
    print("-" * 160)
    for reg in regs:
        fee = load(f"dynamics_{reg}_fee.csv"); base = load(f"dynamics_{reg}.csv")
        grid_max = max(fl(r["fee"]) for r in fee); grid_min = min(fl(r["fee"]) for r in fee)
        for ring in RINGS:
            for t in (2, 8, 16):
                ff = [r for r in fee if r["ring"] == ring and int(r["period"]) == t]
                bb = [r for r in base if r["ring"] == ring and int(r["period"]) == t]
                if not ff:
                    continue
                hi = 100 * sum(1 for r in ff if abs(fl(r["fee"]) - grid_max) < 1e-9) / len(ff)
                lo = 100 * sum(1 for r in ff if abs(fl(r["fee"]) - grid_min) < 1e-9) / len(ff)
                print(f"{reg:<11} | {ring[:10]:<10} | {t:>2} | {m(ff, 'fee'):6.3f} | {m(ff, 'fee') - m(ff, 'fee_registered'):+6.3f} | {hi:5.0f}% | {lo:5.0f}% | "
                      f"{m(ff, 'settle'):6.1f}/{m(bb, 'settle'):6.1f} | {m(ff, 'gate'):5.1f}/{m(bb, 'gate'):5.1f} | {m(ff, 'poach'):6.1f}/{m(bb, 'poach'):6.1f} | "
                      f"{sum(fl(r['incumbent_revenue']) for r in ff):7.1f}/{sum(fl(r['incumbent_revenue']) for r in bb):7.1f} | {m(ff, 'tier2_parity'):7.1f}/{m(bb, 'tier2_parity'):7.1f} | {m(ff, 'sq_error'):6.1f}/{m(bb, 'sq_error'):6.1f}")
    print("-" * 160)
    print("Fee = review fraction charged on settled work (the registered fee is 0.25 + 0.08 x licensing rate); at max / at min = share of tasks at the top / bottom of the grid;")
    print("Inc rev = incumbent revenue summed over the ring's tasks in that period (challenger-wage units); fee/reg = the fee-setting run against the registered-fee run of the same regime.")


def table18() -> None:
    if not os.path.exists(os.path.join(OUT, "dynamics_labor_mandatory_junior_replaced.csv")):
        print("Table 18: no junior-replacement run found"); return
    from domain_cases import FULL_EXTENDED_CORPUS
    gate = {s.occ_title: int(any(x.statutory_gate for x in FULL_EXTENDED_CORPUS if x.occ_title == s.occ_title)) for s in FULL_EXTENDED_CORPUS}
    print("\n" + "=" * 150)
    print("TABLE 18: WHEN THE PROGRAM REPLACES JUNIOR HOURS — the labour block and the learning profession under the two assumptions (mandatory disclosure)")
    print("=" * 150)
    print(f"{'Assumption':<34} | {'Group':<10} | {'t':>2} | {'Jun demand':>10} | {'Hires':>6} | {'Apprent.':>8} | {'Backlog%':>8} | {'Debt%':>6} | {'Inc demand':>10} | {'Emp inside':>10}")
    print("-" * 150)
    for label, tag in (("juniors operate the program (registered)", "mandatory"), ("the program replaces junior hours", "mandatory_junior_replaced")):
        lab = load(f"dynamics_labor_{tag}.csv")
        for grp, val in (("licensed", 1), ("unlicensed", 0)):
            rr = [r for r in lab if gate.get(r["occ_title"], int(r["statutory_gate"])) == val]
            base = {k: m([r for r in rr if int(r["period"]) == 1], k) for k in ("J_demand", "I_demand", "emp_inside")}
            for t in (1, 8, 16):
                tt = [r for r in rr if int(r["period"]) == t]
                pe = [fl(r["periods_to_eligibility"]) for r in tt if fl(r["periods_to_eligibility"]) < 1e6]
                print(f"{label:<34} | {grp:<10} | {t:>2} | {100*m(tt,'J_demand')/base['J_demand']:9.0f} | {sum(fl(r['hires']) for r in tt):6.2f} | {st.mean(pe) if pe else float('nan'):8.1f} | "
                      f"{100*m(tt,'eligible_backlog'):7.0f}% | {100*m(tt,'apprenticeship_debt'):5.1f}% | {100*m(tt,'I_demand')/base['I_demand']:9.0f} | {100*m(tt,'emp_inside')/base['emp_inside']:9.0f}")
    print("-" * 150)
    print("Jun demand, Inc demand, Emp inside = indices, period 1 = 100; Hires = new junior heads in the period (group sum); Apprent. = periods to eligibility at the current pace.")
    if os.path.exists(os.path.join(OUT, "dynamics_knowledge_mandatory_learning_junior_replaced.csv")):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ks", os.path.join(BASE, "run_knowledge_summary.py")); ks = importlib.util.module_from_spec(spec); spec.loader.exec_module(ks)
        print(f"\n{'Learning profession':<34} | {'Group':<10} | {'Promoted':>8} | {'Hold conv':>9} | {'Sen conv 16':>11} | {'Unk conv%':>9} | {'Core err%':>9}")
        for label, tag in (("juniors operate the program", "mandatory_learning"), ("the program replaces junior hours", "mandatory_learning_junior_replaced")):
            s = ks.summarise(tag)
            for grp in ("licensed", "unlicensed"):
                r = s[grp]
                print(f"{label:<34} | {grp:<10} | {r['promoted']:8.1f} | {r['hold_conv']:8.1f}% | {r['s_conv_16']:10.1f}% | {r['unknowing_conv_pct']:8.1f}% | {r['core_err_late']:8.1f}%")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Tables 17 and 18: the fee-setting incumbent and the junior-replacement labour variant").parse_args()
    table17()
    table18()
