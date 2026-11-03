#!/usr/bin/env python3
"""
Closure formation (run_closure_formation.py) — JURI Lab, face validity before the AI shock

Where do statutory gates come from, and would the laboratory put them where the law does? The reference configuration
takes each occupation's statutory sign-off from the empirical programme's hand coding. This script lets closure
FORM inside the laboratory instead, in a jurisdictional battle that predates any compiled tool, and asks whether
the emergent gate set matches the real one. Only then does it add the tool and ask what the same rules do.

The battle. `run_bumping.py` gives every task a second profession, an adjacent subordinate occupation with weaker
closure and no registry (paralegal beside lawyer, registered nurse beside nurse practitioner). Its pre-tool wave
(v0) is the division of labour between the two; its third-generation wave (v3) adds the compiled tool. Both were run
twice: with no statutory gate anywhere (`--gates none`) and with a gate on every core and buffer task
(`--gates all`). From the two runs this script computes, per occupation and wave,

    gate gain        the incumbent's revenue per case with the gate minus without it (challenger-wage units):
                     what the profession would pay to obtain a statute,
    visible error    the share of cases wrong AND overt without the gate: what a regulator could see going wrong,
    silent error     the share wrong and silent without the gate: what no regulator sees,
    client price     the client's cost per case with the gate minus without it: the rent the statute creates.

The closure game. Two actors with declared rules, swept over their constants. The PROFESSION seeks a gate on its
core and buffer work when the gate gain exceeds a lobbying cost, a fraction c of its wage per case. The REGULATOR
grants a gate under one of three models: public interest (grants when visible error without the gate is at least
theta, the harm it can see), harm-weighted public interest (grants when visible error times the liability price of an
error, the expected visible harm per case, is at least a harm threshold), or capture (grants whatever is sought). The
emergent gate set is where the profession seeks and the regulator grants. The face-validity test compares the pre-tool emergent set with the registered one
(eight of eighteen occupations) and against the chance agreement of a random set of the same size.

The AI era. The same rules are then applied to the v3 wave: the tool changes what the gate is worth to the
profession (settlement revenue replaces gatekeeping revenue) and what the regulator can see (the tool's errors are
silent collisions). Which gates the two regulator models would now keep, lift or create is the experiment's
AI-related result, read with the confidence the pre-tool match has or has not earned.

    python3 run_bumping.py --corpus extended --n-eval 20 --gates none
    python3 run_bumping.py --corpus extended --n-eval 20 --gates all
    python3 run_closure_formation.py            # Table 15; data_outputs/closure_formation.csv
"""

import argparse
import csv
import itertools
import math
import os
import juri_paths
import statistics as st
from typing import Dict, List, Tuple

from domain_cases import FULL_EXTENDED_CORPUS
from model_config import CFG

BASE = juri_paths.ROOT
OUT = os.path.join(BASE, "data_outputs")
WAVES = ("v0", "v3")
LOBBY_COSTS = (0.02, 0.05, 0.10)        # lobbying cost as a share of the incumbent's wage per case
HARM_THRESHOLDS = (0.01, 0.02, 0.05)    # visible error without the gate at which a public-interest regulator grants
EXPECTED_HARM_THRESHOLDS = (0.3, 0.5, 0.7)   # visible error x liability price (challenger-wage units per case) for the harm-weighted regulator


def load(name: str) -> List[Dict[str, str]]:
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


_TASK_SPEC = {s.task_id: s for s in FULL_EXTENDED_CORPUS}


def task_liability(task_id: str) -> float:
    """Liability price of one task, from that task's own accountability score."""
    z = _TASK_SPEC[task_id].accountability_z
    return CFG.liability_scale * math.exp(CFG.liability_exp * z)


def occupation_specs() -> Dict[str, object]:
    """One representative non-periphery task spec per occupation (wage ratio and labels only)."""
    out: Dict[str, object] = {}
    for s in FULL_EXTENDED_CORPUS:
        if s.ring != "Outer_Periphery" and s.occ_title not in out:
            out[s.occ_title] = s
    for s in FULL_EXTENDED_CORPUS:
        out.setdefault(s.occ_title, s)
    return out


def per_occupation(rows: List[Dict[str, str]], specs: Dict[str, object]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Importance-weighted per-case quantities on core and buffer tasks, by occupation and wave."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for wave in WAVES:
        for occ in sorted({r["occ_title"] for r in rows}):
            rr = [r for r in rows if r["occ_title"] == occ and r["wave"] == wave and r["ring"] != "Outer_Periphery"]
            if not rr:
                continue
            w = {r["task_id"]: float(r["weight"]) for r in rr}
            tasks = sorted(w)
            wsum = sum(w[t] for t in tasks)
            def tmean(f) -> float:
                return sum(w[t] * st.mean(f(r) for r in rr if r["task_id"] == t) for t in tasks) / wsum
            w_inc = specs[occ].wage_ratio_inc_chall
            out[(occ, wave)] = {
                # harm is weighted task by task, with each task's own liability price, before aggregation.
                "visible_harm": tmean(lambda r: (1.0 - float(r["accurate"])) * (1 - int(r["silent"])) * task_liability(r["task_id"])),
                "revenue": tmean(lambda r: float(r["inc_hours"])) * w_inc,
                "sub_hours": tmean(lambda r: float(r["sub_hours"])),
                "client_cost": tmean(lambda r: float(r["labor_cost"])),
                "error": tmean(lambda r: 1.0 - float(r["accurate"])),
                "visible_error": tmean(lambda r: (1.0 - float(r["accurate"])) * (1 - int(r["silent"]))),
                "silent_error": tmean(lambda r: (1.0 - float(r["accurate"])) * int(r["silent"])),
                "gate_share": tmean(lambda r: 1.0 if r["arrangement"] == "GATE" else 0.0),
                "settle_share": tmean(lambda r: 1.0 if r["arrangement"] in ("SETTLE_INC", "SUB_SUPERVISED") else 0.0),
                "poach_share": tmean(lambda r: 1.0 if r["arrangement"] in ("POACH", "SUB_ALONE", "SETTLE_SUB") else 0.0),
            }
    return out


def agreement(emergent: Dict[str, int], registered: Dict[str, int]) -> Tuple[float, float, float]:
    occs = sorted(registered)
    agree = sum(1 for o in occs if emergent[o] == registered[o]) / len(occs)
    tp = sum(1 for o in occs if emergent[o] == 1 and registered[o] == 1)
    sens = tp / max(1, sum(registered.values()))
    spec = sum(1 for o in occs if emergent[o] == 0 and registered[o] == 0) / max(1, sum(1 for o in occs if registered[o] == 0))
    return agree, sens, spec


def chance_agreement(n: int, k_true: int, k_emergent: int) -> float:
    """Expected agreement with the registered set when a set of size k_emergent is drawn at random from n."""
    # expected true positives = k_true * k_emergent / n; agreement = (2 TP + n - k_true - k_emergent) / n
    tp = k_true * k_emergent / n
    return (2 * tp + n - k_true - k_emergent) / n


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    none = load("bumping_extended_gates_none.csv")
    all_ = load("bumping_extended_gates_all.csv")
    specs = occupation_specs()
    registered = {s.occ_title: int(any(x.statutory_gate for x in FULL_EXTENDED_CORPUS if x.occ_title == s.occ_title)) for s in FULL_EXTENDED_CORPUS}
    q0, q1 = per_occupation(none, specs), per_occupation(all_, specs)
    occs = sorted(registered)
    table: List[Dict[str, object]] = []
    for occ in occs:
        w_inc = specs[occ].wage_ratio_inc_chall
        L = CFG.liability_scale * math.exp(CFG.liability_exp * specs[occ].accountability_z)
        row: Dict[str, object] = {"occ_title": occ, "registered_gate": registered[occ], "accountability_z": specs[occ].accountability_z,
                                  "licensing_rate": specs[occ].cps_licensing_rate, "wage_ratio": w_inc, "liability_price": L}  # display: representative core task; harm uses per-task prices
        for wave in WAVES:
            a, b = q0[(occ, wave)], q1[(occ, wave)]
            row[f"{wave}_gate_gain"] = b["revenue"] - a["revenue"]
            row[f"{wave}_gate_gain_share"] = (b["revenue"] - a["revenue"]) / w_inc
            row[f"{wave}_visible_error_nogate"] = a["visible_error"]
            row[f"{wave}_visible_harm_nogate"] = a["visible_harm"]
            row[f"{wave}_silent_error_nogate"] = a["silent_error"]
            row[f"{wave}_error_nogate"] = a["error"]
            row[f"{wave}_error_gate"] = b["error"]
            row[f"{wave}_client_price"] = b["client_cost"] - a["client_cost"]
            row[f"{wave}_sub_hours_nogate"] = a["sub_hours"]
            row[f"{wave}_sub_hours_gate"] = b["sub_hours"]
            row[f"{wave}_poach_nogate"] = a["poach_share"]
        table.append(row)

    # the closure game over its constants
    results = []
    for wave, c in itertools.product(WAVES, LOBBY_COSTS):
        seek = {r["occ_title"]: int(r[f"{wave}_gate_gain_share"] > c) for r in table}
        variants = [("capture", None, dict(seek))]
        for theta in HARM_THRESHOLDS:
            grant = {r["occ_title"]: int(r[f"{wave}_visible_error_nogate"] >= theta) for r in table}
            variants.append(("public interest", theta, {o: seek[o] * grant[o] for o in occs}))
        for h in EXPECTED_HARM_THRESHOLDS:
            grant = {r["occ_title"]: int(r[f"{wave}_visible_harm_nogate"] >= h) for r in table}
            variants.append(("harm-weighted", h, {o: seek[o] * grant[o] for o in occs}))
            # the regulator alone (no demand condition): what a harm criterion would gate regardless of who asks
            variants.append(("harm-weighted, no demand test", h, dict(grant)))
        for model, thr, em in variants:
            ag, se, sp = agreement(em, registered)
            results.append({"wave": wave, "lobby_cost": c, "threshold": thr if thr is not None else "", "regulator": model, "n_gates": sum(em.values()),
                            "agreement": ag, "sensitivity": se, "specificity": sp,
                            "chance": chance_agreement(len(occs), sum(registered.values()), sum(em.values())),
                            "gates": ";".join(o for o in occs if em[o])})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "closure_formation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    with open(os.path.join(OUT, "closure_formation_game.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)

    print("=" * 150)
    print("TABLE 15: CLOSURE FORMATION — what a statute is worth to the profession and what a regulator can see, before the tool (v0) and with it (v3); core and buffer work")
    print("=" * 150)
    print(f"{'Occupation':<34} | {'Reg.':>4} | {'z':>5} | {'L':>5} | {'v0 gain':>7} | {'v0 vis err':>10} | {'v0 harm':>7} | {'v0 sil err':>10} | {'v0 price':>8} | {'v3 gain':>7} | {'v3 vis err':>10} | {'v3 harm':>7} | {'v3 sil err':>10} | {'v3 price':>8} | {'v3 poach':>8}")
    print("-" * 150)
    for r in sorted(table, key=lambda r: (-r["registered_gate"], -r["v0_gate_gain_share"])):
        print(f"{r['occ_title']:<34} | {r['registered_gate']:>4} | {r['accountability_z']:5.2f} | {r['liability_price']:5.2f} | {100*r['v0_gate_gain_share']:6.1f}% | {100*r['v0_visible_error_nogate']:9.1f}% | {r['v0_visible_harm_nogate']:7.2f} | {100*r['v0_silent_error_nogate']:9.1f}% | {r['v0_client_price']:+8.2f} | "
              f"{100*r['v3_gate_gain_share']:6.1f}% | {100*r['v3_visible_error_nogate']:9.1f}% | {r['v3_visible_harm_nogate']:7.2f} | {100*r['v3_silent_error_nogate']:9.1f}% | {r['v3_client_price']:+8.2f} | {100*r['v3_poach_nogate']:7.1f}%")
    print("-" * 150)
    print("Reg. = the registered (hand-coded) statutory gate; z = accountability score; gain = incumbent revenue per case with the gate minus without, as a share of the incumbent's wage;")
    print("vis / sil err = error without the gate on overt / silent cases; price = client cost per case with the gate minus without (challenger-wage units); v3 poach = share of cases done without any incumbent when ungated.")

    print("\n" + "=" * 150)
    print("TABLE 15b: THE CLOSURE GAME — emergent gate sets against the registered set (18 occupations, 8 registered gates)")
    print("=" * 150)
    print(f"{'Wave':<4} | {'Lobby cost':>10} | {'Threshold':>9} | {'Regulator':<30} | {'Gates':>5} | {'Agreement':>9} | {'Chance':>6} | {'Sensitivity':>11} | {'Specificity':>11} | Emergent gates")
    print("-" * 150)
    for r in results:
        if r["lobby_cost"] != 0.05 and "no demand" not in r["regulator"]:
            continue   # the lobbying cost never changes the demand set between 2% and 10%; print the 5% rows
        if r["lobby_cost"] != 0.05:
            continue
        thr = f"{r['threshold']:.2f}" if r["threshold"] != "" else ""
        print(f"{r['wave']:<4} | {100*r['lobby_cost']:9.0f}% | {thr:>9} | {r['regulator']:<30} | {r['n_gates']:>5} | {100*r['agreement']:8.1f}% | {100*r['chance']:5.1f}% | {100*r['sensitivity']:10.1f}% | {100*r['specificity']:10.1f}% | {r['gates'][:60]}")
    print("-" * 150)
    print("Agreement = occupations whose emergent and registered gate status coincide; Chance = expected agreement of a random gate set of the same size.")

    # ---- Table 15c: externalised harm. The client bears half the liability (liability_scale x 0.5 in the battle);
    # the regulator weighs visible error by the full liability price. Needs the *_phi05 bumping runs.
    if os.path.exists(os.path.join(OUT, "bumping_extended_gates_none_phi05.csv")) and os.path.exists(os.path.join(OUT, "bumping_extended_gates_all_phi05.csv")):
        n5, a5 = load("bumping_extended_gates_none_phi05.csv"), load("bumping_extended_gates_all_phi05.csv")
        p0, p1 = per_occupation(n5, specs), per_occupation(a5, specs)
        table5 = []
        for occ in occs:
            w_inc = specs[occ].wage_ratio_inc_chall
            L = CFG.liability_scale * math.exp(CFG.liability_exp * specs[occ].accountability_z)
            row = {"occ_title": occ, "registered_gate": registered[occ], "liability_price": L}
            for wave in WAVES:
                a, b = p0[(occ, wave)], p1[(occ, wave)]
                row[f"{wave}_gate_gain_share"] = (b["revenue"] - a["revenue"]) / w_inc
                row[f"{wave}_visible_error_nogate"] = a["visible_error"]
                row[f"{wave}_visible_harm_nogate"] = a["visible_harm"]
                row[f"{wave}_silent_error_nogate"] = a["silent_error"]
                row[f"{wave}_error_nogate"] = a["error"]
                row[f"{wave}_error_gate"] = b["error"]
            table5.append(row)
        with open(os.path.join(OUT, "closure_formation_phi05.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table5[0].keys())); w.writeheader(); w.writerows(table5)
        res5 = []
        for wave in WAVES:
            seek = {r["occ_title"]: int(r[f"{wave}_gate_gain_share"] > 0.05) for r in table5}
            variants = [("capture", None, dict(seek))]
            for theta in HARM_THRESHOLDS:
                grant = {r["occ_title"]: int(r[f"{wave}_visible_error_nogate"] >= theta) for r in table5}
                variants.append(("public interest", theta, {o: seek[o] * grant[o] for o in occs}))
            for h in EXPECTED_HARM_THRESHOLDS:
                grant = {r["occ_title"]: int(r[f"{wave}_visible_harm_nogate"] >= h) for r in table5}
                variants.append(("harm-weighted", h, {o: seek[o] * grant[o] for o in occs}))
                variants.append(("harm-weighted, no demand test", h, dict(grant)))
            for model, thr, em in variants:
                ag, se, sp = agreement(em, registered)
                res5.append({"wave": wave, "threshold": thr if thr is not None else "", "regulator": model, "n_gates": sum(em.values()), "agreement": ag,
                             "sensitivity": se, "specificity": sp, "chance": chance_agreement(len(occs), sum(registered.values()), sum(em.values())),
                             "gates": ";".join(o for o in occs if em[o])})
        with open(os.path.join(OUT, "closure_formation_game_phi05.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res5[0].keys())); w.writeheader(); w.writerows(res5)
        print("\n" + "=" * 150)
        print("TABLE 15c: THE CLOSURE GAME WHEN HARM IS EXTERNAL — the client bears half the liability; the regulator weighs visible error by the full liability price")
        print("=" * 150)
        print(f"{'Occupation':<34} | {'Reg.':>4} | {'L':>5} | {'v0 gain':>7} | {'v0 vis err':>10} | {'v0 harm':>7} | {'v0 err gate':>11} | {'v3 gain':>7} | {'v3 vis err':>10} | {'v3 harm':>7} | {'v3 sil err':>10}")
        print("-" * 150)
        for r in sorted(table5, key=lambda r: (-r["registered_gate"], -r["v0_gate_gain_share"])):
            print(f"{r['occ_title']:<34} | {r['registered_gate']:>4} | {r['liability_price']:5.2f} | {100*r['v0_gate_gain_share']:6.1f}% | {100*r['v0_visible_error_nogate']:9.1f}% | {r['v0_visible_harm_nogate']:7.2f} | {100*r['v0_error_gate']:10.1f}% | "
                  f"{100*r['v3_gate_gain_share']:6.1f}% | {100*r['v3_visible_error_nogate']:9.1f}% | {r['v3_visible_harm_nogate']:7.2f} | {100*r['v3_silent_error_nogate']:9.1f}%")
        print("-" * 150)
        print(f"{'Wave':<4} | {'Threshold':>9} | {'Regulator':<30} | {'Gates':>5} | {'Agreement':>9} | {'Chance':>6} | {'Sensitivity':>11} | {'Specificity':>11} | Emergent gates")
        for r in res5:
            thr = f"{r['threshold']:.2f}" if r["threshold"] != "" else ""
            print(f"{r['wave']:<4} | {thr:>9} | {r['regulator']:<30} | {r['n_gates']:>5} | {100*r['agreement']:8.1f}% | {100*r['chance']:5.1f}% | {100*r['sensitivity']:10.1f}% | {100*r['specificity']:10.1f}% | {r['gates'][:60]}")
        print("=" * 150)


if __name__ == "__main__":
    main()
