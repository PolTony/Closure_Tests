#!/usr/bin/env python3
"""
Sourced calibration (run_sourced_calibration.py) — JURI Lab

Re-runs the reference configuration with the occupation attributes replaced by the sourced values of the empirical
program's frozen closure table (data_inputs/lab18_closure_values.csv, hand-off of 2026-10-31):

  cps_licensing_rate    <- cps_licensing_rate_unit        (CPS 2015-19 share holding an active government licence; unit level)
  wage_ratio_inc_chall  <- wage_ratio_to_all_occ_median_2024  (OEWS May 2024 median hourly wage over the all-occupations median,
                                                            i.e. the incumbent's rate against a generic worker; 2019 if 2024 is missing)
  accountability_z      <- accountability_z_native22      (O*NET 27.0 native SOC-2018 code; unit19 if missing)
  statutory_gate        <- signoff_handcoded on the occupation's core and buffer tasks, 0 on the periphery
                           (the programme's hand coding of professions whose core work needs a licensed signature)

The reference configuration is left untouched: the specifications are modified only inside this process and the
outputs go to data_outputs/sourced/. The printed table compares the headline cells of README Tables 1 and 2
under the registered constants and under the sourced ones, so that a reader can see which findings depend on
the hand-set attributes and which do not.
"""

import csv
import io
import json
import os
import juri_paths
import sys

BASE_DIR = juri_paths.ROOT
SRC = os.path.join(BASE_DIR, "data_inputs", "lab18_closure_values.csv")
OUT = os.path.join(BASE_DIR, "data_outputs", "sourced")


def load_sourced():
    with open(SRC, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(io.StringIO("".join(l for l in f if not l.startswith("#")))))
    return {r["occ_title"]: r for r in rows}


def fnum(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def apply(corpus, src):
    """Overrides the attributes in place. Returns a change log."""
    log = []
    for s in corpus:
        r = src.get(s.occ_title) or next((v for v in src.values() if v["occ_soc"] == s.occ_soc or v["soc2018"] == s.occ_soc), None)
        if r is None:
            log.append((s.task_id, "NO SOURCED ROW"))
            continue
        lic = fnum(r["cps_licensing_rate_unit"])
        w = fnum(r["wage_ratio_to_all_occ_median_2024"]) or fnum(r["wage_ratio_to_all_occ_median_2019"])
        z = fnum(r["accountability_z_native22"])
        if z is None:
            z = fnum(r["accountability_z_unit19"])
        gate = int(fnum(r["signoff_handcoded"], 0) >= 0.5) if s.ring != "Outer_Periphery" else 0
        before = (s.cps_licensing_rate, s.wage_ratio_inc_chall, s.accountability_z, s.statutory_gate)
        if lic is not None:
            s.cps_licensing_rate = round(lic, 3)
        if w is not None:
            s.wage_ratio_inc_chall = round(w, 3)
        if z is not None:
            s.accountability_z = round(z, 3)
        s.statutory_gate = gate
        log.append((s.task_id, before, (s.cps_licensing_rate, s.wage_ratio_inc_chall, s.accountability_z, s.statutory_gate),
                    "" if w is not None else "wage kept (no OEWS hourly median)"))
    return log


def headline(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    def m(col, sel=None):
        rr = [r for r in rows if sel is None or sel(r)]
        return sum(float(r[col]) for r in rr) / len(rr)
    core = lambda r: r["ring"] == "Inner_Core"
    buf = lambda r: r["ring"] == "Middle_Buffer"
    per = lambda r: r["ring"] == "Outer_Periphery"
    return {
        "Tool v3 end to end": m("A_Tool_v3_Pct"), "Incumbent alone": m("A_Inc_Solo_Pct"), "Settlement": m("A_Collab_Pct"),
        "Periphery poach": m("Direct_Poaching_Pct", per), "Buffer settle": m("Accountability_Settlement_Pct", buf),
        "Buffer gate": m("Incumbent_Gatekeeping_Pct", buf), "Core poach": m("Direct_Poaching_Pct", core),
        "Core settle": m("Accountability_Settlement_Pct", core), "Core gate": m("Incumbent_Gatekeeping_Pct", core),
        "Core SQ error": m("SQ_Error_Rate_Pct", core), "Core CF1 error": m("CF1_Dereg_Error_Pct", core),
        "Core CF2 error": m("CF2_NoLiab_Error_Pct", core), "Mean shield gap (pp)": m("Institutional_Shield_Gap_pp"),
        "Mean d ln wage bill": m("Implied_Delta_Ln_WageBill"),
    }


def main():
    import domain_cases as dc
    import run_live_benchmark as rlb
    src = load_sourced()
    log = apply(dc.FULL_EXTENDED_CORPUS, src)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "attribute_changes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task_id", "lic_before", "wage_before", "z_before", "gate_before", "lic_after", "wage_after", "z_after", "gate_after", "note"])
        for e in log:
            if len(e) == 4:
                w.writerow([e[0], *e[1], *e[2], e[3]])
    print("Sourced attributes applied to", sum(1 for e in log if len(e) == 4), "tasks; running the registered pipeline on them ...", flush=True)
    rlb.run_full(tools_dir=os.path.join(BASE_DIR, "sourced_tools"), out_dir=OUT, quiet=True)
    reg = headline(os.path.join(BASE_DIR, "data_outputs", "results_extended_18occ_tasks.csv"))
    new = headline(os.path.join(OUT, "results_extended_18occ_tasks.csv"))
    print("\n" + "=" * 92)
    print("TABLE 12: REGISTERED CONSTANTS vs SOURCED CLOSURE ATTRIBUTES (extended corpus, 1,080 packets)")
    print("=" * 92)
    print(f"{'Headline cell':<28} | {'Registered':>10} | {'Sourced':>10} | {'Change':>8}")
    print("-" * 92)
    for k in reg:
        print(f"{k:<28} | {reg[k]:10.2f} | {new[k]:10.2f} | {new[k]-reg[k]:+8.2f}")
    print("-" * 92)
    print("Sourced: CPS licence share (unit level), OEWS 2024 wage over the all-occupations median, O*NET 27.0 accountability (native code),")
    print("statutory gate = the programme's sign-off hand coding on core and buffer tasks. Outputs in data_outputs/sourced/.")
    with open(os.path.join(OUT, "headline_comparison.json"), "w") as f:
        json.dump({"registered": reg, "sourced": new}, f, indent=1)


if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(description='Table 12: the reference configuration on sourced closure attributes').parse_args()   # no options; rejects stray arguments and serves --help
    main()
