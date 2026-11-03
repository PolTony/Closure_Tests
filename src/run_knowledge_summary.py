#!/usr/bin/env python3
"""
Knowledge-race summary (run_knowledge_summary.py) — JURI Lab

Reads the learning-profession dynamics written by `run_dynamics.py --profession learning` (and its sensitivity runs)
and prints Table 14 (the knowledge race by regime, licensed and unlicensed occupations, periods 1, 8 and 16) and
Table 14b (where knowledge reproduction breaks: the sensitivity runs side by side). Every number is recomputed from
the CSV files in data_outputs/, so the tables in README.md and the appendix can be checked against this output.

    python3 run_knowledge_summary.py
"""

import csv
import os
import juri_paths
import statistics as st
from typing import Dict, List

BASE = juri_paths.ROOT
OUT = os.path.join(BASE, "data_outputs")


def load(name: str) -> List[Dict[str, str]]:
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def fl(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def ok(v: float) -> bool:
    return v == v


def gate_map(know: List[Dict[str, str]]) -> Dict[str, int]:
    """Licensed and unlicensed always mean the REGISTERED statute, also in runs that impose or remove gates."""
    from domain_cases import FULL_EXTENDED_CORPUS
    reg = {s.occ_title: int(any(x.statutory_gate for x in FULL_EXTENDED_CORPUS if x.occ_title == s.occ_title)) for s in FULL_EXTENDED_CORPUS}
    return {r["occ_title"]: reg.get(r["occ_title"], int(r["statutory_gate"])) for r in know}


def summarise(tag: str) -> Dict[str, object]:
    """Headline quantities of one learning run: late-period error against its oracle twin, knowledge at period 16,
    competence at promotion over periods 5 to 16, and the share of hard cases signed without the knowledge."""
    d = load(f"dynamics_{tag}.csv")
    k = load(f"dynamics_knowledge_{tag}.csv")
    g = gate_map(k)
    T = max(int(r["period"]) for r in d)
    res: Dict[str, object] = {"T": T}
    for grp, val in (("licensed", 1), ("unlicensed", 0)):
        dd = [r for r in d if g.get(r["occ_title"], 0) == val]
        kk = [r for r in k if g.get(r["occ_title"], 0) == val]
        late_k = [r for r in kk if int(r["period"]) >= 5]
        late_d = [r for r in dd if int(r["period"]) >= 9]
        hard = sum(int(r["hard_n"]) for r in late_d) or 1
        hard_conv = sum(int(r.get("hard_conv_n", 0)) for r in late_d) or 1
        hard_reg = sum(int(r.get("hard_reg_n", 0)) for r in late_d) or 1
        # competence at promotion is HEAD-WEIGHTED: promoted heads who held the component over all promoted heads
        prom = sum(fl(r["promoted"]) for r in late_k if ok(fl(r["promoted"])))
        hc = sum(fl(r["promoted"]) * fl(r["promoted_hold_convention"]) for r in late_k if ok(fl(r["promoted_hold_convention"])) and ok(fl(r["promoted"])))
        hr = sum(fl(r["promoted"]) * fl(r["promoted_hold_registry"]) for r in late_k if ok(fl(r["promoted_hold_registry"])) and ok(fl(r["promoted"])))
        l16 = [r for r in dd if int(r["period"]) == T]
        res[grp] = {
            "s_conv_16": 100 * st.mean(fl(r["s_conv"]) for r in l16),
            "s_reg_16": 100 * st.mean(fl(r["s_reg"]) for r in l16),
            "parity_16": st.mean(fl(r["tier2_parity"]) for r in l16 if ok(fl(r["tier2_parity"]))),
            "promoted": prom,
            "hold_conv": 100 * hc / prom if prom > 1e-9 else float("nan"),
            "hold_reg": 100 * hr / prom if prom > 1e-9 else float("nan"),
            "unknowing_pct": 100 * sum(int(r["unknowing_handled"]) for r in late_d) / hard,
            "unknowing_err_pct": 100 * sum(int(r["unknowing_errors"]) for r in late_d) / hard,
            "unknowing_conv_pct": 100 * sum(int(r.get("unknowing_conv", 0)) for r in late_d) / hard_conv,
            "unknowing_conv_err_pct": 100 * sum(int(r.get("unknowing_conv_err", 0)) for r in late_d) / hard_conv,
            "unknowing_reg_pct": 100 * sum(int(r.get("unknowing_reg", 0)) for r in late_d) / hard_reg,
            "unknowing_reg_err_pct": 100 * sum(int(r.get("unknowing_reg_err", 0)) for r in late_d) / hard_reg,
            "sq_err_late": st.mean(fl(r["sq_error"]) for r in late_d),
            "core_err_late": st.mean(fl(r["sq_error"]) for r in late_d if r["ring"] == "Inner_Core"),
            "min_s_reg": 100 * min(fl(r["s_reg"]) for r in dd if int(r["period"]) >= 2),
        }
    return res


def table14(tags: List[str]) -> None:
    print("=" * 150)
    print("TABLE 14: THE KNOWLEDGE RACE — learning profession, by regime (periods 1, 8, 16; licensed and unlicensed occupations)")
    print("=" * 150)
    print(f"{'Regime':<12} | {'Group':<10} | {'t':>2} | {'T2 parity':>9} | {'Sen conv':>8} | {'Sen reg':>7} | {'Jun reg':>7} | {'Unknowing%':>10} | {'Unk err%':>8} | {'SQ err%':>7} | {'Oracle SQ%':>10} | {'Escal.':>6} | {'Defer':>5}")
    print("-" * 150)
    for tag in tags:
        regime = tag.split("_")[0]
        d = load(f"dynamics_{tag}.csv")
        oracle_name = f"dynamics_{tag.replace('_learning', '')}.csv"
        o = load(oracle_name) if os.path.exists(os.path.join(OUT, oracle_name)) else None
        g = gate_map(load(f"dynamics_knowledge_{tag}.csv"))
        for grp, val in (("licensed", 1), ("unlicensed", 0)):
            for t in (1, 8, 16):
                rr = [r for r in d if g.get(r["occ_title"], 0) == val and int(r["period"]) == t]
                if not rr:
                    continue
                oo = [r for r in o if g.get(r["occ_title"], 0) == val and int(r["period"]) == t] if o else []
                hard = sum(int(r["hard_n"]) for r in rr) or 1
                def m(key, rows):
                    vals = [fl(r[key]) for r in rows if ok(fl(r[key]))]
                    return st.mean(vals) if vals else float("nan")   # junior shares are undefined at period 1
                print(f"{regime:<12} | {grp:<10} | {t:>2} | {m('tier2_parity', rr):8.1f}% | {100*m('s_conv', rr):7.1f}% | {100*m('s_reg', rr):6.1f}% | {100*m('j_reg', rr):6.1f}% | "
                      f"{100*sum(int(r['unknowing_handled']) for r in rr)/hard:9.1f}% | {100*sum(int(r['unknowing_errors']) for r in rr)/hard:7.1f}% | {m('sq_error', rr):6.1f}% | "
                      f"{(m('sq_error', oo) if oo else float('nan')):9.1f}% | {sum(int(r['escalations']) for r in rr):6d} | {sum(int(r['deferrals']) for r in rr):5d}")
    print("-" * 150)
    print("Sen conv / Sen reg = share of seniors holding the convention / the registry at the start of the period; Jun reg = share of juniors holding the registry;")
    print("Unknowing% = hard cases (Tiers 2 and 3) signed by a professional who lacked the component the case required, as a share of hard cases; Unk err% = errors among them;")
    print("Oracle SQ% = status-quo error of the same regime with the registered (all-knowing) incumbent.")


def table14b(runs: List[tuple]) -> None:
    print("\n" + "=" * 160)
    print("TABLE 14b: WHERE KNOWLEDGE REPRODUCTION BREAKS — competence at promotion (periods 5-16), seniors' knowledge at period 16, and errors signed without the knowledge (periods 9-16)")
    print("=" * 160)
    print(f"{'Run':<34} | {'Group':<10} | {'Promoted':>8} | {'Hold conv':>9} | {'Hold reg':>8} | {'Sen conv 16':>11} | {'Sen reg 16':>10} | {'Unk conv%':>9} | {'err%':>5} | {'Unk reg%':>8} | {'err%':>5} | {'Core err%':>9} | {'T2 par 16':>9}")
    print("-" * 160)
    for label, tag in runs:
        if not os.path.exists(os.path.join(OUT, f"dynamics_{tag}.csv")):
            print(f"{label:<34} | (not run)")
            continue
        s = summarise(tag)
        for grp in ("licensed", "unlicensed"):
            r = s[grp]
            print(f"{label:<34} | {grp:<10} | {r['promoted']:8.1f} | {r['hold_conv']:8.1f}% | {r['hold_reg']:7.1f}% | {r['s_conv_16']:10.1f}% | {r['s_reg_16']:9.1f}% | "
                  f"{r['unknowing_conv_pct']:8.1f}% | {r['unknowing_conv_err_pct']:4.1f}% | {r['unknowing_reg_pct']:7.1f}% | {r['unknowing_reg_err_pct']:4.1f}% | {r['core_err_late']:8.1f}% | {r['parity_16']:8.1f}%")
    print("-" * 160)
    print("Promoted = heads promoted over periods 5-16; Hold conv / Hold reg = head-weighted share of them who held the component at promotion;")
    print("Unk conv% = convention cases signed by a professional who lacked the convention, as a share of convention cases (periods 9-16), and the errors among them as a share of convention cases;")
    print("Unk reg% = the same for precedent cases and the registry; Core err% = inner-core status-quo error over periods 9-16 (the registered oracle gives 7.1% under mandatory disclosure).")


if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(description='Tables 14 and 14b from the learning-profession dynamics').parse_args()   # no options; rejects stray arguments and serves --help
    present = [t for t in ("mandatory_learning", "discretion_learning", "strategic_learning") if os.path.exists(os.path.join(OUT, f"dynamics_{t}.csv"))]
    table14(present)
    table14b([
        ("registered calibration, mandatory", "mandatory_learning"),
        ("registered calibration, discretion", "discretion_learning"),
        ("registered calibration, strategic", "strategic_learning"),
        ("learning twice as fast (kappa x0.5)", "mandatory_learning_k05"),
        ("learning slower (kappa x1.5)", "mandatory_learning_k15"),
        ("learning twice as slow (kappa x2)", "mandatory_learning_k20"),
        ("faster turnover (retirement 10%)", "mandatory_learning_delta10"),
        ("faster demand growth (5%)", "mandatory_learning_growth05"),
        ("no escalation of precedent memos", "mandatory_learning_noescalate"),
        ("reviewers override instead of defer", "mandatory_learning_nodefer"),
        ("regulator feedback, mandatory", "mandatory_fb_regulator_learning"),
        ("regulator feedback, strategic", "strategic_fb_regulator_learning"),
        ("no statute anywhere (gates none)", "mandatory_learning_gates_none"),
        ("statute everywhere (gates all)", "mandatory_learning_gates_all"),
        ("kappa x1.5, no statute anywhere", "mandatory_learning_k15_gates_none"),
        ("kappa x1.5, statute everywhere", "mandatory_learning_k15_gates_all"),
        ("thresholds on the period-1 pace", "mandatory_learning_kbasis_p1"),
        ("registered calibration, 32 periods", "mandatory_learning_t32"),
        ("fee-setting incumbent", "mandatory_learning_fee"),
        ("program replaces junior hours", "mandatory_learning_junior_replaced"),
    ])
