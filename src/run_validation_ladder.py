#!/usr/bin/env python3
"""
Consistency inventory (run_validation_ladder.py) — JURI Lab, Table 16

This table is an inventory of consistency, mismatch and untested prediction. It is NOT a validation count: most rows
are AI-era consistency checks or direct consequences of specified costs, labels and inputs, several compare different
constructs, and none was registered before the reference configuration was set. Each row therefore records its
evidence class (pre-AI institutional allocation, AI-era consistency, or untested prediction), whether the laboratory
and the source measure the same construct, the unit and contrast on each side, and the criterion that would count as
failure. The prospective protocol for validation proper is docs/VALIDATION_PROTOCOL.md.

Regularities from occupations theory and from the empirical record that the laboratory was never fitted to, each
set beside the laboratory's counterpart, with two honest labels: how the counterpart arises (by construction,
computed from the constants, or emergent from an interaction) and whether it matches (match, partial, mismatch).
The point of a ladder is that the mismatches are reported with the matches: they say where the laboratory's
assumptions break and therefore how far its AI-era results should be trusted.

Every number is recomputed from data_outputs/ when this script runs; the empirical and theoretical sides are
quoted from the sources named in the appendix (the author's programme notes are in data_inputs/empirical/).

    python3 run_validation_ladder.py     # prints Table 16; writes data_outputs/validation_ladder.csv
"""

import argparse
import csv
import json
import os
import juri_paths
import statistics as st
from collections import Counter
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


def main() -> None:
    argparse.ArgumentParser(description="Table 16: the validation ladder").parse_args()
    tr = json.load(open(os.path.join(OUT, "results_packet_traces_extended_1080.json")))
    occ = load("results_occupation_summary_18.csv")
    lab = load("dynamics_labor_mandatory.csv")
    bump = load("bumping_extended.csv")
    fb = load("dynamics_mandatory_fb_both.csv")
    rows: List[Dict[str, object]] = []

    def add(regularity, source, counterpart, value, grade, verdict, note="", *, evidence_class, construct,
            unit_contrast, failure_criterion):
        rows.append({"regularity": regularity, "source": source, "laboratory_counterpart": counterpart, "value": value,
                     "how_it_arises": grade, "verdict": verdict, "note": note, "evidence_class": evidence_class,
                     "construct": construct, "unit_contrast": unit_contrast, "failure_criterion": failure_criterion})

    # 1 subordination is the modal settlement in contested work
    cb = [t for t in tr if t["ring"] != "Outer_Periphery"]
    g = Counter(t["negotiated_governance"] for t in cb)
    settle, gate, poach = (100 * g[k] / len(cb) for k in ("ACCOUNTABILITY_SETTLEMENT", "INCUMBENT_FULL_GATEKEEPING", "DIRECT_CHALLENGER_POACHING"))
    add("Subordination is the modal settlement when a profession's work is contested",
        "Abbott (1988); the programme's census of 470 regulatory bodies: 80 subordination rows against 9 exclusions",
        "governance of buffer and core work under the status quo", f"settled {settle:.0f}%, gated {gate:.0f}%, poached {poach:.0f}%",
        "computed (liability and penalty constants)", "consistent in rank order; constructs differ",
        evidence_class="pre-AI institutional allocation (status quo, no tool)",
        construct="DIFFERENT: written subordination rules of regulatory bodies against the share of synthetic work packets settled",
        unit_contrast="source: rule rows per body; laboratory: packets in buffer and core rings",
        failure_criterion="gate or poach share above settle share in contested rings")

    # 2 sign-off tasks use AI least
    def tool_use(ts):
        return 100 * sum(1 for t in ts if t["negotiated_governance"] in ("DIRECT_CHALLENGER_POACHING", "ACCOUNTABILITY_SETTLEMENT")) / len(ts)
    t3 = [t for t in tr if t["complexity_tier"] == 3]
    add("Sign-off tasks show far less AI use than their exposure predicts",
        "the programme: 53 to 73% less use on two platforms (s547)",
        "share of precedent cases whose final answer came from the program, against all cases",
        f"{tool_use(t3):.0f}% against {tool_use(tr):.0f}% ({100 * (1 - tool_use(t3) / tool_use(tr)):.0f}% below)",
        "by construction (precedent is never compiled)", "consistent in sign by construction; the magnitude is typed",
        evidence_class="AI-era consistency (follows from the typed tier)",
        construct="DIFFERENT: reserved sign-off tasks in platform usage data against Tier 3 knowledge cases",
        unit_contrast="source: task usage within occupation; laboratory: packets by tier",
        failure_criterion="not falsifiable in this design (Tier 3 cases are never compiled by construction)")

    # 3 licensed occupations adopt less
    lic = [o for o in occ if int(o["statutory_gate"]) == 1]
    unl = [o for o in occ if int(o["statutory_gate"]) == 0]
    add("Licensed occupations adopt generative AI less than their exposure predicts",
        "the programme (s631): -0.25 to -0.29 SD per SD of licensing on two releases; README Table 4b",
        "share of work in which the program is involved, licensed against unlicensed occupations",
        f"{st.mean(100 - fl(o['Incumbent_Gatekeeping_Pct']) for o in lic):.0f}% against {st.mean(100 - fl(o['Incumbent_Gatekeeping_Pct']) for o in unl):.0f}%",
        "computed (typed task content and characteristics)", "consistent in sign; not attributable to the gate",
        "the project's own counterfactual (CF1, no sign-off penalty) leaves within-occupation uptake unchanged, so the contrast comes from the typed content of licensed occupations' work, not from the gate penalty",
        evidence_class="AI-era consistency",
        construct="related: adoption net of exposure against program involvement by occupation",
        unit_contrast="source: occupations, SD of licensing; laboratory: 18 occupations, gate dummy",
        failure_criterion="licensed involvement at or above unlicensed involvement")

    # 4 deregulating sign-off where errors are visible lowers price without quality loss
    err = {}
    for ring in ("Middle_Buffer", "Inner_Core"):
        rr = [t for t in tr if t["ring"] == ring]
        err[ring] = (100 * sum(1 for t in rr if not t["sq_accurate"]) / len(rr), 100 * sum(1 for t in rr if not t["cf1_accurate"]) / len(rr))
    add("Relaxing a sign-off requirement lowers prices without lowering quality where the work's errors are observable",
        "Kleiner, Marier, Park, and Wing (2016), nurse-practitioner scope of practice: prices of a routine medical exam are 3 to 16% HIGHER where rules are more restrictive (not a 3 to 16% fall from the restrictive baseline); no change in the measured health-outcome and malpractice proxies, specific to that setting",
        "error under the status quo and under CF1 (no sign-off penalty), buffer against core",
        f"buffer {err['Middle_Buffer'][0]:.1f}% to {err['Middle_Buffer'][1]:.1f}%; core {err['Inner_Core'][0]:.1f}% to {err['Inner_Core'][1]:.1f}%",
        "computed", "consistent for the buffer in direction; prices not modelled; the core is an untested prediction",
        evidence_class="AI-era consistency (buffer); untested prediction (core)",
        construct="DIFFERENT: market prices and measured health outcomes in one service against synthetic packet error rates",
        unit_contrast="source: US states, restrictive versus permissive; laboratory: SQ against CF1 by ring",
        failure_criterion="buffer error rises materially when the sign-off penalty is removed")

    # 5 licensing raises employment, not wages (Redbird 2017)
    def idx(rows_, key, gval):
        rr = [r for r in rows_ if int(r["statutory_gate"]) == gval]
        a = st.mean(fl(r[key]) for r in rr if r["period"] == "1"); b = st.mean(fl(r[key]) for r in rr if r["period"] == "16")
        return 100 * b / a
    istock = f"{idx(lab, 'I_stock', 1):.0f} against {idx(lab, 'I_stock', 0):.0f} (index, period 1 = 100)"
    add("Entry standardization: new licensing standardizes entry and raises participation without a net wage benefit",
        "Redbird (2017), American Sociological Review 82(3): licensing as entry standardization",
        "incumbent stock at period 16 relative to period 1, licensed against unlicensed (mandatory disclosure)",
        istock, "computed", "not tested: the laboratory has no entry-standardization mechanism",
        "post-shock stock growth in different occupations is not the licensing-employment effect; headcount here follows demand for signatures",
        evidence_class="untested (mechanism absent)",
        construct="DIFFERENT: effect of adopting licensing on participation and wages against post-AI stock growth",
        unit_contrast="source: occupation-state panels over licensing adoption; laboratory: licensed vs unlicensed synthetic tasks after the tool",
        failure_criterion="requires an entry-standardization mechanism; none is modelled")
    add("Supply restriction: licensing raises wages and hours and reduces employment",
        "Kleiner and Soltas (2023), Review of Economic Studies 90(5): occupations licensed differently across US states",
        "incumbent stock at period 16 relative to period 1, licensed against unlicensed (mandatory disclosure)",
        istock, "computed", "not tested: the laboratory has no entry barrier or occupational wage",
        "Redbird's entry-standardization and Kleiner-Soltas's supply-restriction findings concern different mechanisms, samples and outcomes and are not graded as one directional target",
        evidence_class="untested (mechanism absent)",
        construct="DIFFERENT: cross-state licensing effect on wages, hours and employment against post-AI stock growth",
        unit_contrast="source: occupations licensed in some states only; laboratory: synthetic tasks",
        failure_criterion="requires a supply-restriction mechanism; none is modelled")

    # 6 no closure mechanism protects pay against AI exposure
    add("No closure mechanism protects occupational pay against AI exposure",
        "the programme (s605): bounded nulls for union, licensing, certification, credentials, associations",
        "implied change in the log wage bill, licensed against unlicensed occupations",
        f"{st.mean(fl(o['Implied_Delta_Ln_WageBill']) for o in lic):+.2f} against {st.mean(fl(o['Implied_Delta_Ln_WageBill']) for o in unl):+.2f}",
        "computed", "not comparable: the laboratory's unit-cost measure leans toward protection", "the measures differ: a task's unit cost against an occupation's median wage",
        evidence_class="AI-era consistency",
        construct="DIFFERENT: occupational median wage against unit task cost",
        unit_contrast="source: occupations by closure type; laboratory: licensed vs unlicensed synthetic occupations",
        failure_criterion="requires an occupational wage; none is modelled")

    # 7 apprentices lose the work they learned on
    pe = {g: st.mean(fl(r["periods_to_eligibility"]) for r in lab if int(r["statutory_gate"]) == g and r["period"] == "16" and fl(r["periods_to_eligibility"]) < 1e6) for g in (1, 0)}
    add("When technology takes the routine share of the work, trainees lose the practice that made them experts",
        "Beane (2019) on surgical residents; Anthony (2021) on junior bankers",
        "periods to eligibility at period 16 against the reference 8 (mandatory disclosure)",
        f"licensed {pe[1]:.1f}, unlicensed {pe[0]:.1f}", "emergent (labour block on governance outcomes)", "consistent in direction",
        evidence_class="AI-era consistency",
        construct="related: qualitative accounts of lost practice against simulated periods to eligibility",
        unit_contrast="source: ethnographies of single settings; laboratory: synthetic cohorts",
        failure_criterion="periods to eligibility at or below the reference 8")

    # 8 the adjacent, less protected occupation loses first
    sub = {}
    for ring in ("Middle_Buffer", "Inner_Core"):
        r0 = [r for r in bump if r["ring"] == ring and r["wave"] == "v0"]; r3 = [r for r in bump if r["ring"] == ring and r["wave"] == "v3"]
        sub[ring] = (100 * st.mean(fl(r["sub_hours"]) for r in r3) / st.mean(fl(r["sub_hours"]) for r in r0),
                     100 * st.mean(fl(r["inc_hours"]) for r in r3) / st.mean(fl(r["inc_hours"]) for r in r0))
    add("A jurisdictional disturbance reaches the adjacent, less protected occupation first",
        "Abbott (1988) on vacancy and bumping chains; the programme's lawyer/paralegal postings test (null so far)",
        "subordinate and incumbent hours at the third compiler generation relative to the pre-tool division of labour",
        f"buffer: subordinate {sub['Middle_Buffer'][0]:.0f}, incumbent {sub['Middle_Buffer'][1]:.0f}; core: {sub['Inner_Core'][0]:.0f} and {sub['Inner_Core'][1]:.0f}",
        "computed", "consistent with Abbott's account; not observed in the programme's postings data", "the programme finds no break in the lawyer/paralegal ratio after guidance",
        evidence_class="AI-era consistency; empirical test so far null",
        construct="related: hours by occupation against postings ratios",
        unit_contrast="source: lawyer/paralegal postings; laboratory: bumping chains by ring",
        failure_criterion="incumbent hours fall before subordinate hours")

    # 9 regulators act on visible error
    g1 = 100 * st.mean(fl(r["gate_state"]) for r in fb if r["period"] == "1"); g16 = 100 * st.mean(fl(r["gate_state"]) for r in fb if r["period"] == "16")
    add("Professions' own bodies subordinate AI rather than exclude it; exclusion comes from legislatures where harm is visible",
        "the programme's census: own bodies 80 subordination rows, 9 exclusions; 13 of 15 binding exclusions are statutes",
        "share of tasks gated under a regulator that sees only visible error, period 1 against 16",
        f"{g1:.0f}% to {g16:.0f}%", "computed (regulator rule on visible error)", "consistent in direction by the specified rule",
        evidence_class="AI-era consistency (follows from the specified regulator rule)",
        construct="DIFFERENT: rule-making bodies' choices against a regulator that gates on visible error",
        unit_contrast="source: rule rows by body type; laboratory: share of tasks gated by period",
        failure_criterion="gate share falls as visible error rises")

    # 10 early-career workers lose employment first
    jd = {g: idx(lab, "J_demand", g) for g in (1, 0)}
    alt_path = os.path.join(OUT, "dynamics_labor_mandatory_junior_replaced.csv")
    if os.path.exists(alt_path):
        alt = load("dynamics_labor_mandatory_junior_replaced.csv")
        ja = {g: idx(alt, "J_demand", g) for g in (1, 0)}
        add("Early-career workers in exposed occupations lose employment first",
            "Brynjolfsson, Chandar, and Chen (2025): a 13 to 16% relative decline for ages 22 to 25",
            "junior demand at period 16 relative to period 1 (mandatory disclosure), under the reference assumption that juniors operate the program and under the switch in which the program replaces junior hours",
            f"reference: licensed {jd[1]:.0f}, unlicensed {jd[0]:.0f}; program replaces juniors: licensed {ja[1]:.0f}, unlicensed {ja[0]:.0f} (index, period 1 = 100)",
            "computed", "junior DEMAND falls under the replacement switch and rises under the operator assumption; employment is not tested",
            "demand is not headcount: the scenarios initialise stocks after the tool is present and normalise each to its own period 1, so neither is a causal effect relative to a common pre-AI population; stocks, hires, promotions and exits are reported separately in Table 9",
            evidence_class="AI-era scenario (post-adoption)",
            construct="DIFFERENT: payroll employment by age against modelled junior labour demand",
            unit_contrast="source: payroll records, ages 22-25; laboratory: junior hours demanded per period",
            failure_criterion="not a test; one assignment decides the sign of demand")
    else:
        add("Early-career workers in exposed occupations lose employment first",
            "Brynjolfsson, Chandar, and Chen (2025): a 13 to 16% relative decline for ages 22 to 25",
            "junior demand at period 16 relative to period 1 (mandatory disclosure)",
            f"licensed {jd[1]:.0f}, unlicensed {jd[0]:.0f} (index, period 1 = 100)",
            "computed", "junior demand rises; employment not tested", "the labour block assigns one tool-equipped junior-hour per settled or poached case, so junior demand rises; a block in which the program replaces junior hours would reverse the sign",
            evidence_class="AI-era scenario (post-adoption)", construct="DIFFERENT: payroll employment against junior demand",
            unit_contrast="as above", failure_criterion="not a test")

    # 11 closure forms where harm is high (closure formation)
    if os.path.exists(os.path.join(OUT, "closure_formation_game.csv")):
        game = load("closure_formation_game.csv")
        v0 = [r for r in game if r["wave"] == "v0" and fl(r["lobby_cost"]) == 0.05]
        best = max(v0, key=lambda r: fl(r["agreement"]))
        add("Statutory closure sits where errors harm third parties, in health, law, engineering and accounting",
            "the public-interest account of licensing; the reference gate set (8 of 18 occupations)",
            "best agreement of any pre-tool closure game (rent-seeking, visible-error or harm-weighted regulator) with the coded statutory gate set",
            f"{100 * fl(best['agreement']):.0f}% against {100 * fl(best['chance']):.0f}% by chance ({best['regulator']})",
            "emergent (closure game on the pre-tool battle)", "mismatch at full client liability",
            "the failure has several candidate explanations (target coding, subordinate error prior, client cost rule, lobbying demand and cost, error visibility, task aggregation); it does not identify third-party harm as the missing theory; the externalised-liability variant is Table 15c",
            evidence_class="pre-AI institutional allocation (the one direct pre-tool test)",
            construct="related: a profession-level hand coding of sign-off (benchmark coding, not statute) against modelled gates",
            unit_contrast="source: 18 occupations, 8 coded gated; laboratory: closure game per occupation",
            failure_criterion="agreement not above chance")
        v3 = [r for r in game if r["wave"] == "v3" and fl(r["lobby_cost"]) == 0.05 and r["regulator"] == "capture"]
        cf = load("closure_formation.csv")
        gains = [fl(r["v3_gate_gain_share"]) for r in cf]
        add("After the AI shock, demand for statutory sign-off rises across occupations",
            "the programme: licensing language in state AI bills rose from 3 mentions (2019 to 2022) to 52 (2025)",
            "the gate's value to the profession in the AI era, every occupation; and visible harm without the gate",
            f"gain {100 * min(gains):.0f} to {100 * max(gains):.0f}% of the wage per case in all 18 occupations; a captured regulator would gate {v3[0]['n_gates'] if v3 else 'all'} of 18",
            "emergent", "consistent in direction", "the visible harm that would justify the gate has vanished (Table 15)",
            evidence_class="AI-era prediction",
            construct="DIFFERENT: mentions of licensing in bills against modelled gate value",
            unit_contrast="source: state AI bills by year; laboratory: occupations",
            failure_criterion="gate value falls after the shock")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "validation_ladder.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("=" * 150)
    print("TABLE 16: CONSISTENCY INVENTORY — regularities beside laboratory counterparts (not a validation count)")
    print("=" * 150)
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r['regularity']}")
        print(f"    source: {r['source']}")
        print(f"    laboratory: {r['laboratory_counterpart']}: {r['value']}")
        print(f"    arises: {r['how_it_arises']} | verdict: {r['verdict']}" + (f" | {r['note']}" if r["note"] else ""))
        print(f"    class: {r['evidence_class']} | construct: {r['construct']} | unit: {r['unit_contrast']} | fails if: {r['failure_criterion']}")
    pre = [r for r in rows if r["evidence_class"].startswith("pre-AI")]
    print("-" * 150)
    print(f"Rows: {len(rows)}; pre-AI institutional rows: {len(pre)}; the only direct pre-tool allocation test (closure) fails.")
    print("No count of consistent rows is reported as validation; see docs/VALIDATION_PROTOCOL.md.")
    print("=" * 150)


if __name__ == "__main__":
    main()
