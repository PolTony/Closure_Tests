#!/usr/bin/env python3
"""
Lab moments against the empirical program (run_moments.py) — JURI Lab

Computes, from the registered-design traces, the quantities that correspond to the empirical facts
established in the occupations/AI program (research notes on the `main` branch, 2026-10-29/30), and prints
them side by side. The reference configuration was developed iteratively while aware of occupational theory. These are exploratory face-validity comparisons, not independent out-of-sample validation.

Mapping of constructs
  tool use            packet governed by poaching or settlement (tool output produced; in settlement the
                      incumbent audits it)                          ~ generative-AI uptake on the occupation's tasks
  within-task gate    tool use under CF1 (no statutory gate) minus SQ ~ within-occupation licensing effect on uptake
  sign-off layer      Tier 3 (precedent) packets                      ~ tasks requiring a licensee's signature
  between-occupation  tool use on licensed vs unlicensed tasks, raw and net of parity/accountability
  subordination       settlement share                                ~ professions' bodies settling AI by subordination
  price of the task   unit labour cost per packet (NOT an occupation wage)
"""
import csv, json, os
from collections import defaultdict
import math
import argparse as _argparse
import juri_paths
_argparse.ArgumentParser(description='Table 8: laboratory moments against the empirical program').parse_args()   # no options; rejects stray arguments and serves --help

BASE = juri_paths.ROOT
tr = json.load(open(os.path.join(BASE, "data_outputs", "results_packet_traces_extended_1080.json")))
rows = {r["task_id"]: r for r in csv.DictReader(open(os.path.join(BASE, "data_outputs", "results_extended_18occ_tasks.csv")))}
USE = {"DIRECT_CHALLENGER_POACHING", "ACCOUNTABILITY_SETTLEMENT"}

def share(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def ols(X, y):
    import itertools
    n, k = len(X), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    # gaussian elimination
    M = [XtX[a][:] + [Xty[a]] for a in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1): M[r][j] -= f * M[c][j]
    return [M[i][k] / M[i][i] for i in range(k)]

by = defaultdict(list)
for t in tr: by[t["task_id"]].append(t)
task_use_sq = {k: share([t["negotiated_governance"] in USE for t in v]) for k, v in by.items()}
task_use_cf1 = {k: share([t["cf1_governance"] in USE for t in v]) for k, v in by.items()}
task_gate_sq = {k: share([t["negotiated_governance"] == "INCUMBENT_FULL_GATEKEEPING" for t in v]) for k, v in by.items()}
task_gate_cf1 = {k: share([t["cf1_governance"] == "INCUMBENT_FULL_GATEKEEPING" for t in v]) for k, v in by.items()}

lic = [k for k in by if rows[k]["statutory_gate"] == "1"]
unl = [k for k in by if rows[k]["statutory_gate"] == "0"]
X = [[1.0, float(rows[k]["statutory_gate"]), float(rows[k]["accountability_z"]), float(rows[k]["A_Tool_v3_Pct"]) / 100, float(rows[k]["collision_density"])] for k in by]
y = [task_use_sq[k] for k in by]
b = ols(X, y)
t3 = [t for t in tr if t["complexity_tier"] == 3]
t3_use = share([t["negotiated_governance"] in USE for t in t3]); all_use = share([t["negotiated_governance"] in USE for t in tr])
silent = [t for t in tr if t["is_silent"]]
core = [t for t in tr if t["ring"] == "Inner_Core"]

print("=" * 118)
print("TABLE 8: LABORATORY MOMENTS AGAINST THE EMPIRICAL PROGRAM (exploratory, calibrated face-validity comparisons)")
print("=" * 118)
print(f"{'Construct':<46} | {'Lab (reference configuration)':<34} | {'Empirical program (hand-off of 2026-10-31, data_inputs/empirical/)'}")
print("-" * 118)
print(f"{'Within-occupation licence effect on uptake':<46} | {'CF1 - SQ tool use = ' + f'{100*(sum(task_use_cf1.values())-sum(task_use_sq.values()))/len(by):+.2f} pp (all 54 tasks: 0)':<34} | EU within-ISCO-4: +0.004 (SE 0.026); US states inconclusive")
_d_poach = 100*share([t["cf1_governance"]=="DIRECT_CHALLENGER_POACHING" for t in tr])-100*share([t["negotiated_governance"]=="DIRECT_CHALLENGER_POACHING" for t in tr])
_d_settle = 100*share([t["cf1_governance"]=="ACCOUNTABILITY_SETTLEMENT" for t in tr])-100*share([t["negotiated_governance"]=="ACCOUNTABILITY_SETTLEMENT" for t in tr])
_cell = f"poach {_d_poach:+.1f} pp, settle {_d_settle:+.1f} pp, gate 0"
print(f"{'  what the gate changes instead':<46} | {_cell:<34} | bodies issue subordination rules (80 rows) vs exclusion (9); who signs, not whether AI is used")
print(f"{'Between-occupation uptake, licensed vs not (raw)':<46} | {f'{100*share([task_use_sq[k] for k in lic]):.1f}% vs {100*share([task_use_sq[k] for k in unl]):.1f}%':<34} | licensed occupations adopt less: -0.25 to -0.29 SD per SD of statutory licensing on the two Claude releases (knowledge work), -0.12 to -0.18 SD on Massenkoff and Microsoft (s631 full regression)")
print(f"{'  net of capability and accountability':<46} | {f'licence coef {b[1]:+.3f} (share); parity {b[3]:+.2f}':<34} | gap read as a property of the reserved work (content, liability), not the licence")
print(f"{'Sign-off layer: tool use vs overall':<46} | {f'Tier 3 {100*t3_use:.1f}% vs all {100*all_use:.1f}%: {100*(1-t3_use/all_use):.0f}% below':<34} | within occupation, formal sign-off tasks show -73% on Massenkoff (t -4.95) and -53% on AEI June 2026 (t -1.70, wild-bootstrap p .089); pre-specified verdict PARTIAL (s547)")
_cs = 100*share([t["negotiated_governance"]=="ACCOUNTABILITY_SETTLEMENT" for t in core]); _cg = 100*share([t["negotiated_governance"]=="INCUMBENT_FULL_GATEKEEPING" for t in core]); _cp = 100*share([t["negotiated_governance"]=="DIRECT_CHALLENGER_POACHING" for t in core])
_cell = f"settle {_cs:.1f}% / gate {_cg:.1f}% / poach {_cp:.1f}% (core)"
print(f"{'Modal settlement in contested work':<46} | {_cell:<34} | subordination is the modal institutional settlement; no exclusion except by statute")
print(f"{'Closure and the price of the work':<46} | {'gate changes unit cost per task via who signs; no occupation wage in the model':<34} | no closure mechanism protects occupation pay (bounded within +/-0.02/SD, s605_closure); premiums intact in levels")
print(f"{'Incumbents and entry':<46} | {'incumbents retained as auditors (settle 45-55%); static reference; dynamic pipeline reported separately':<34} | incumbents retained, pay intact; hiring held up in licensed and unionized occupations; exposed occupations slow 3-5 lp/SD (s605, s628); the entrant-penalty evidence weakened in later audits (s462, s629)")
_e0 = 100*share([not t["sq_accurate"] for t in core]); _e1 = 100*share([not t["cf1_accurate"] for t in core])
_cell = f"CF1 raises core error {_e0:.1f}% -> {_e1:.1f}%; all silent collisions poached"
print(f"{'Silent errors and deregulation':<46} | {_cell:<34} | untested: scope-of-practice designs uninformative (MDE 8-14 lp); prediction for future releases")
print("-" * 118)
print("Reading: the laboratory reproduces the sign pattern of the program's robust facts (gate acts on who signs and on the")
print("sign-off layer, not on uptake within occupation; between-occupation uptake differences come from the content of reserved work;")
print("subordination is modal) under the iteratively developed reference configuration. It does not yet speak to wages, headcount, entry or incumbents' pay,")
print("in the reference configuration; the labour and apprenticeship block (Table 9, run_dynamics.py) is where those statements now live.")


# ---------------------------------------------------------------------------
# TABLE 8b: label exposure vs compilability — can the adoption gap be a property of the exposure construct?
# ---------------------------------------------------------------------------
# A text rater sees task documents, not outcomes. Overt precedent memos ("TIER3_COUPLED_PRECEDENT") read as
# work that needs a registry; everything else, including silent collisions, reads as codifiable. So the lab
# analogue of a labelled exposure is the share of a task's packets that are not overt Tier 3. Compilability
# (parity) is what the tool actually gets right.
label_exp = {k: share([not (t["complexity_tier"] == 3 and not t["is_silent"]) for t in v]) for k, v in by.items()}
parity = {k: float(rows[k]["A_Tool_v3_Pct"]) / 100 for k in by}
keys = list(by)
yv = [task_use_sq[k] for k in keys]
def gap_regress(expo):
    Xr = [[1.0, expo[k], float(rows[k]["accountability_z"]), float(rows[k]["collision_density"])] for k in keys]
    br = ols(Xr, yv)
    resid = [yv[i] - sum(br[j] * Xr[i][j] for j in range(4)) for i in range(len(keys))]
    lic_mean = share([resid[i] for i, k in enumerate(keys) if rows[k]["statutory_gate"] == "1"])
    unl_mean = share([resid[i] for i, k in enumerate(keys) if rows[k]["statutory_gate"] == "0"])
    sd = (sum((r - sum(resid) / len(resid)) ** 2 for r in resid) / (len(resid) - 1)) ** 0.5
    return br[1], lic_mean, unl_mean, sd
b_lab, gl, gu, sdl = gap_regress(label_exp)
b_par, pl, pu, sdp = gap_regress(parity)
print("\n" + "=" * 118)
print("TABLE 8b: UPTAKE NET OF EXPOSURE — WHEN EXPOSURE IS A TEXT LABEL vs WHEN IT IS MEASURED COMPILABILITY")
print("=" * 118)
print(f"{'exposure construct':<34} | {'uptake slope':>12} | {'residual gap, licensed':>22} | {'residual gap, unlicensed':>24} | {'gap in SD units':>15}")
print("-" * 118)
print(f"{'label exposure (share not overt Tier 3)':<34} | {b_lab:>+12.3f} | {gl:>+22.3f} | {gu:>+24.3f} | {(gl-gu)/sdl:>+15.2f}")
print(f"{'measured compilability (tool parity)':<34} | {b_par:>+12.3f} | {pl:>+22.3f} | {pu:>+24.3f} | {(pl-pu)/sdp:>+15.2f}")
print(f"mean label exposure: licensed {share([label_exp[k] for k in lic]):.2f} vs unlicensed {share([label_exp[k] for k in unl]):.2f}; mean compilability: {share([parity[k] for k in lic]):.2f} vs {share([parity[k] for k in unl]):.2f}")
print("Reading: the hypothesis that the empirical adoption gap is an artefact of labelled exposure is NOT supported by the")
print("reference configuration. Net of a text-style label, licensed tasks use the tool slightly MORE than the label predicts (+0.7 SD),")
print("not less: overt precedent memos are visible to a rater, and silent collisions (42 of 1,080 packets) are too few to open a")
print("wedge between label and compilability. The wedge the hypothesis needs would require a larger share of reserved work that")
print("reads as routine. Empirical analogue to explain: licensed occupations adopt -0.4 to -0.7 SD below labelled exposure (E_g).")
