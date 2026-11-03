#!/usr/bin/env python3
"""
Figures for the technical appendix (make_figures.py) — JURI Lab

Reads only files in data_outputs/ (plus the task specifications for one worked example) and writes PNG figures
to submission/figures/. Requires matplotlib (the laboratory itself needs only the standard library):

    pip install matplotlib && python3 make_figures.py

The figures use color together with markers and line styles; labels and printed values identify the series. Figure numbers and captions live in the
document, not in the image. Every figure is drawn from data the runners produced, including Figure 10, which
reads data_outputs/archive_boundary.csv written by run_archive_boundary.py.
"""

import collections
import csv
import json
import math
import hashlib
import os
import juri_paths
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator

BASE = juri_paths.ROOT
D = os.path.join(BASE, "data_outputs")
OUT = os.path.join(BASE, "submission", "figures")
os.makedirs(OUT, exist_ok=True)

# Categorical palette (validated order: blue, orange, aqua, yellow, magenta, green), with marker and line style
SHADE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
LINE = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]
HATCH = ["", "", "", "", "", ""]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
FILL = {"dark": "#2a78d6", "mid": "#eb6834", "light": "#9cc4ee", "pale": "#eef3fb", "aqua": "#1baf7a", "red": "#e34948"}
RINGS = ["Outer_Periphery", "Middle_Buffer", "Inner_Core"]
RING_LABEL = {"Outer_Periphery": "Routine tasks", "Middle_Buffer": "Contested tasks", "Inner_Core": "Core tasks"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8,
    "font.family": "DejaVu Sans", 
})


SUBMISSION_FIGURES = {'F01_architecture.png': 'A01_study_design.png', 'F06_governance_regimes.png': 'A02_work_allocation.png', 'F11_dynamics_parity.png': 'A03_case_disclosure.png', 'F18_hollow_signature.png': 'A04_professional_knowledge.png', 'F13_bumping.png': 'A05_occupational_competition.png', 'F20_fee_setting.png': 'A06_review_fees.png', 'F19_closure_formation.png': 'A07_licensing_formation.png'}


def save(fig, name, note=""):
    if name not in SUBMISSION_FIGURES:
        plt.close(fig)
        return
    name = SUBMISSION_FIGURES[name]
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {name}" + (f"  ({note})" if note else ""))


def rows(name):
    with open(os.path.join(D, name)) as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def pct(xs):
    return 100.0 * mean(xs)


def box(ax, x, y, w, h, text, fc, ec=INK2, fs=8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight, wrap=True)


def arrow(ax, p, q, text="", color=INK2, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.015, text, ha="center", va="bottom", fontsize=7, color=INK2)


def line(ax, x, y, i, label=None, lw=1.8, ms=4.5):
    ax.plot(x, y, color=SHADE[i % 6], ls=LINE[i % 6], marker=MARK[i % 6], ms=ms, lw=lw, label=label)


# ---------------------------------------------------------------------------------------------------------------
# F01 architecture
# ---------------------------------------------------------------------------------------------------------------
def f01_architecture():
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    blue, orange, green = "#e6eef9", "#fdeee6", "#e4f5ee"
    box(ax, .03, .68, .25, .24, "Workplace information\nWritten instructions\nResolved cases\nPrivate professional precedents", blue, fs=9)
    box(ax, .38, .74, .25, .16, "Automation agent\nReads public instructions\nand permitted cases", orange, fs=9)
    box(ax, .38, .43, .25, .20, "Professional\nWorks alone or reviews\nHas access to private precedents\nKnowledge can depend on training", orange, fs=8.5)
    box(ax, .73, .64, .23, .26, "Client allocates the work\nAutomation alone\nProfessional supervision\nProfessional execution", green, fs=8.5)
    box(ax, .73, .31, .23, .20, "Measured outcomes\nCorrect answers and errors\nWho works and signs\nHours, charges, and learning", green, fs=8.5)
    box(ax, .03, .31, .25, .20, "Researcher’s answer key\nExact answers for grading\nExcluded from automation", blue, fs=9)
    box(ax, .20, .06, .63, .13, "Following periods\nPublished outcomes change what automation can learn\nPractice, hiring, promotion, and retirement change professional knowledge", blue, fs=8.5)
    arrow(ax, (.28,.83),(.38,.82)); arrow(ax, (.28,.71),(.38,.55))
    arrow(ax, (.63,.82),(.73,.80)); arrow(ax, (.63,.54),(.73,.68))
    arrow(ax, (.85,.64),(.85,.51)); arrow(ax, (.28,.41),(.73,.41),"grade the delivered answer")
    arrow(ax, (.84,.31),(.74,.19))
    ax.plot([.20, .002, .002], [.12, .12, .80], color=INK2, lw=.9)
    arrow(ax, (.002,.80),(.03,.80))
    ax.text(.10, .14, "published cases", fontsize=7, color=INK2, ha="center")
    arrow(ax, (.53,.19),(.53,.43),"updated knowledge")
    save(fig, "F01_architecture.png")


# ---------------------------------------------------------------------------------------------------------------
# F02 who sees what -> accuracy grid; F03 accuracy by tier
# ---------------------------------------------------------------------------------------------------------------
def f02_f03_ceilings(traces):
    agents = [("Novice", "A_Base_e2e"), ("Tool v1", "A_Tool_v1_e2e"), ("Tool v2", "A_Tool_v2_e2e"), ("Tool v3", "A_Tool_v3_e2e"),
              ("Incumbent alone", "A_Inc_Solo_e2e"), ("Settlement (audited)", "A_Collab_e2e")]
    tiers = [0, 1, 2, 3, "silent"]
    tier_labels = ["Tier 0\nroutine\n(615)", "Tier 1\nexplicit\n(98)", "Tier 2\nconvention\n(177)", "Tier 3 overt\nprecedent\n(148)", "Tier 3 silent\nprecedent\n(42)"]
    grid = []
    for lab, key in agents:
        row = []
        for t in tiers:
            sel = [x for x in traces if (x["is_silent"] if t == "silent" else (x["complexity_tier"] == t and not x["is_silent"]))]
            row.append(pct([float(x[key]) for x in sel]))
        grid.append(row)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.grid(False)
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", ["#eef3fb", "#2a78d6"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(tiers))); ax.set_xticklabels(tier_labels)
    ax.set_yticks(range(len(agents))); ax.set_yticklabels([a for a, _ in agents])
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color=INK if v < 60 else "white")
    info = ["handbook", "handbook only (parsed)", "handbook and archive", "handbook and archive", "registry added", "registry added; reviews tool"]
    for i, s in enumerate(info):
        ax.text(len(tiers) - 0.5 + 0.08, i, s, va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(-0.5, len(tiers) + 1.5)
    ax.set_xlabel("case kind (number of packets); the agent's information set is printed to the right of each row")
    save(fig, "F02_ceilings_grid.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    labs = ["Novice", "Tool v1", "Tool v2", "Tool v3", "Incumbent", "Settlement"]
    n = len(labs); w = 0.14
    names = ["Tier 0 routine", "Tier 1 explicit", "Tier 2 convention", "Tier 3 overt", "Tier 3 silent"]
    for j, t in enumerate(tiers):
        vals = [grid[i][j] for i in range(n)]
        xs = [i + (j - 2) * w for i in range(n)]
        ax.bar(xs, vals, width=w - 0.015, color=SHADE[j % 6], edgecolor=SURF, linewidth=0.8, label=names[j], zorder=3)
    ax.set_xticks(range(n)); ax.set_xticklabels(labs)
    ax.set_ylim(0, 105); ax.set_ylabel("end-to-end accuracy, %")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "F03_accuracy_by_tier.png")


# ---------------------------------------------------------------------------------------------------------------
# F04 compiler learning curve + complexity; F05 learned thresholds
# ---------------------------------------------------------------------------------------------------------------
def f04_f05_compiler(rob, tasks):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    pts = sorted([(int(re.sub(r"\D", "", r["run"])), float(r["Tier2_Tool_v2"]), float(r["Tier2_Tool_v3"]), float(r["E2E_Tool_v3"])) for r in rob if r["run"].startswith("archive_")])
    xs = [p[0] for p in pts]
    line(ax, xs, [p[2] for p in pts], 0, "Tier 2 parity, v3")
    line(ax, xs, [p[1] for p in pts], 1, "Tier 2 parity, v2")
    line(ax, xs, [p[3] for p in pts], 2, "end to end, v3")
    ax.axvline(80, color=GRID, lw=1); ax.text(84, 4, "registered\narchive = 80", fontsize=7.5, color=INK2)
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("public archive size (resolved cases per task)"); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.legend(loc="lower right"); ax.set_title("A. Convention parity against the size of the public record")
    ax = axes[1]
    for k, ring in enumerate(RINGS):
        rr = [t for t in tasks if t["ring"] == ring]
        v1 = mean([float(t["halstead_vol_v1"]) for t in rr]); v3 = mean([float(t["halstead_vol_v3"]) for t in rr])
        line(ax, [0, 1], [v1, v3], k, RING_LABEL[ring])
        ax.text(1.04, v3, f"{v3:,.0f}", va="center", fontsize=8, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["v1 (handbook only)", "v3 (with induced handler)"]); ax.set_xlim(-0.2, 1.4)
    ax.set_ylabel("Halstead volume of the emitted program"); ax.legend(loc="upper left"); ax.set_title("B. Program volume by generation and ring")
    save(fig, "F04_compiler_learning.png")

    splits = []
    for t in tasks:
        m = re.search(r"V2>=([0-9.]+)", t["v3_split"] or "")
        if m:
            splits.append(float(m.group(1)))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(splits, bins=[0.100 + 0.0025 * i for i in range(16)], color=FILL["dark"], edgecolor=SURF, linewidth=0.8, zorder=3)
    ax.axvline(0.12, color=FILL["red"], lw=1.5); ax.text(0.1208, ax.get_ylim()[1] * 0.9, "true breakpoint 0.12", color=FILL["red"], fontsize=8)
    ax.set_xlabel("learned split on V2 in the v3 handler (34 tasks)"); ax.set_ylabel("tasks")
    save(fig, "F05_learned_thresholds.png", f"{len(splits)} splits, range {min(splits):.4f} to {max(splits):.4f}")


# ---------------------------------------------------------------------------------------------------------------
# F06 governance by ring and regime; F06b error decomposition; F07 cost curves
# ---------------------------------------------------------------------------------------------------------------
def f06_governance(traces):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    regimes = [("Status quo", "negotiated_governance", "sq_accurate"), ("CF1", "cf1_governance", "cf1_accurate"), ("CF2", "cf2_governance", "cf2_accurate")]
    govs = [("DIRECT_CHALLENGER_POACHING", "automation alone", 0), ("ACCOUNTABILITY_SETTLEMENT", "professional supervision", 1), ("INCUMBENT_FULL_GATEKEEPING", "professional execution", 2)]
    xpos, labels = [], []
    x = 0
    for ring in RINGS:
        for rl, gk, ak in regimes:
            sel = [t for t in traces if t["ring"] == ring]
            bottom = 0
            for g, gl, i in govs:
                v = 100.0 * sum(1 for t in sel if t[gk] == g) / len(sel)
                ax.bar(x, v, bottom=bottom, color=SHADE[i], edgecolor=SURF, linewidth=1.5, width=0.8, zorder=3, label=gl if x == 0 else None)
                bottom += v
            xpos.append(x); labels.append({"Status quo": "Current\nrules", "CF1": "No\nsign-off", "CF2": "No sign-off\nor liability"}[rl]); x += 1
        x += 0.8
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7.5)
    for k, ring in enumerate(RINGS):
        ax.text(k * 3.8 + 1, 103, RING_LABEL[ring], ha="center", fontsize=9, weight="bold")
    ax.set_ylim(0, 110); ax.set_ylabel("share of packets, %"); ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.24))
    ax.set_title("A. Allocation of work")
    ax = axes[1]
    for k, ring in enumerate(RINGS):
        sel = [t for t in traces if t["ring"] == ring]
        errs = [100.0 * sum(1 for t in sel if not t[ak]) / len(sel) for _, _, ak in regimes]
        line(ax, range(3), errs, k, RING_LABEL[ring])
        ax.text(2.1, errs[-1], f"{errs[-1]:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Current\nrules", "No\nsign-off", "No sign-off\nor liability"]); ax.set_xlim(-0.3, 2.5)
    ax.set_ylabel("error, %"); ax.legend(loc="upper left"); ax.set_title("B. Error under each regime")
    save(fig, "F06_governance_regimes.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    core = [t for t in traces if t["ring"] == "Inner_Core"]
    cats = [("Status quo", "sq_accurate"), ("CF1", "cf1_accurate"), ("CF2", "cf2_accurate")]
    sil = [100.0 * sum(1 for t in core if not t[a] and t["is_silent"]) / len(core) for _, a in cats]
    ove = [100.0 * sum(1 for t in core if not t[a] and not t["is_silent"]) / len(core) for _, a in cats]
    ax.bar(range(3), ove, color=FILL["dark"], edgecolor=SURF, linewidth=1.5, width=0.55, label="overt (visible) errors", zorder=3)
    ax.bar(range(3), sil, bottom=ove, color=FILL["red"], edgecolor=SURF, linewidth=1.5, width=0.55, label="silent-collision errors", zorder=3)
    for i in range(3):
        ax.text(i, ove[i] + sil[i] + 1, f"{ove[i] + sil[i]:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([c for c, _ in cats]); ax.set_ylabel("inner-core error, % of packets")
    ax.legend(loc="upper left")
    save(fig, "F06b_error_decomposition.png")


def f07_cost_curves():
    from domain_cases import FULL_EXTENDED_CORPUS
    from model_config import CFG
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, tid in zip(axes, ["CPA-B2", "FIN-B2"]):
        s = next(x for x in FULL_EXTENDED_CORPUS if x.task_id == tid)
        L = CFG.liability_scale * math.exp(CFG.liability_exp * s.accountability_z)
        pen = (CFG.penalty_base + CFG.penalty_per_lic * s.cps_licensing_rate) if s.statutory_gate else 0.0
        w_settle = CFG.w_chall + (CFG.review_fraction_base + CFG.review_fraction_per_lic * s.cps_licensing_rate) * s.wage_ratio_inc_chall
        us = [i / 200 for i in range(0, 201)]
        poach = [CFG.w_chall + pen + u * L for u in us]
        settle = [w_settle + ((CFG.residual_settle_low * u) if u <= CFG.residual_settle_cut else (CFG.residual_settle_high * u)) * L for u in us]
        gate = [s.wage_ratio_inc_chall + CFG.residual_incumbent * L] * len(us)
        ax.plot(us, poach, color=SHADE[0], lw=2, label="poach (tool alone)")
        ax.plot(us, settle, color=SHADE[1], lw=2, label="settle (tool, incumbent signs)")
        ax.plot(us, gate, color=SHADE[2], lw=2, label="gate (incumbent alone)")
        for u, lab in [(0.03, "routine"), (0.12, "override"), (0.19, "handler"), (0.86, "unparsed")]:
            ax.axvline(u, color=GRID, lw=0.8); ax.text(u, max(poach) * 0.97, lab, rotation=90, fontsize=7, color=INK2, va="top", ha="right")
        ax.set_yscale("log")
        ax.set_xlabel("critic's uncertainty u_p"); ax.set_ylabel("expected cost (challenger wage = 1), log scale")
        gated = "statutorily gated" if s.statutory_gate else "not gated"
        ax.set_title(f"{tid} ({gated}): w_inc = {s.wage_ratio_inc_chall:.2f}, L = {L:.1f}")
        if tid == "CPA-B2":
            ax.legend(loc="center right")
    save(fig, "F07_cost_curves.png")


# ---------------------------------------------------------------------------------------------------------------
# F08 occupation scatter; F09 robustness; F16 Monte Carlo
# ---------------------------------------------------------------------------------------------------------------
def f08_occupations(tasks):
    occ = collections.defaultdict(list)
    for t in tasks:
        occ[t["occ_title"]].append(t)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for name, tt in occ.items():
        w = [float(t["task_importance_wt"]) for t in tt]
        par = sum(float(t["A_Tool_v3_Pct"]) * x for t, x in zip(tt, w)) / sum(w)
        gate = sum(float(t["Incumbent_Gatekeeping_Pct"]) * x for t, x in zip(tt, w)) / sum(w)
        lic = any(int(t["statutory_gate"]) for t in tt)
        hv = mean([float(t["halstead_vol_v3"]) for t in tt])
        ax.scatter(par, gate, s=(hv - 1800) * 0.9 + 40, color=SHADE[1] if lic else SHADE[0], alpha=0.85, edgecolor=SURF, linewidth=1.2, zorder=3)
        # hand-placed offsets for the crowded corners so that no label overlaps another
        offsets = {"Lawyers": (8, 6), "Clinical_Counseling_Psychologists": (-60, 22), "Nurse_Practitioners": (8, -14),
                   "Civil_Engineers": (8, 6), "Architects": (-70, 14), "Accountants_Auditors": (10, 6), "Registered_Nurses": (10, -14),
                   "Business_Teachers_Postsecondary": (-90, 22), "Financial_Analysts": (-40, -26), "Systems_Analysts": (10, 6),
                   "Software_QA_Analysts": (-70, -26), "HR_Specialists": (8, 6), "Management_Analysts": (-80, 20), "Paralegals": (8, -12)}
        off = offsets.get(name, (6, 4))
        leader = dict(arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, alpha=0.7, shrinkA=0, shrinkB=4)) if (abs(off[0]) > 30 or abs(off[1]) > 18) else {}
        ax.annotate(name.replace("_", " "), (par, gate), textcoords="offset points", xytext=off, fontsize=7, color=INK2,
                    ha="right" if off[0] < -20 else "left", **leader)
    ax.scatter([], [], color=SHADE[1], label="statutory sign-off on some task"); ax.scatter([], [], color=SHADE[0], label="no statutory sign-off")
    for hv_ref, lab in ((1900, "program volume 1,900"), (2400, "2,400")):
        ax.scatter([], [], s=(hv_ref - 1800) * 0.9 + 40, color="none", edgecolor=INK2, linewidth=1.0, label=lab)
    ax.legend(loc="upper right", fontsize=7.5, labelspacing=1.1, borderpad=0.8); ax.set_xlabel("compiled tool parity (v3, importance-weighted), %"); ax.set_ylabel("work kept by the incumbent alone (gatekeeping), %")
    save(fig, "F08_occupations.png")


def f09_f16_robustness(rob):
    cells = [("E2E_Tool_v3", "Tool v3, end to end"), ("E2E_Collab", "Settlement, end to end"), ("Tier2_Tool_v3", "Tier 2 parity, v3"),
             ("Core_Gate", "Core gatekeeping"), ("Core_Settle", "Core settlement"), ("Core_SQErr", "Core error, status quo")]
    blocks = sorted(set(r["block"] for r in rob if r["block"] != "base"))
    reg = next(r for r in rob if r["run"] == "registered_design")
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (key, lab) in zip(axes.flat, cells):
        for i, b in enumerate(blocks):
            vals = [float(r[key]) for r in rob if r["block"] == b]
            ax.scatter(vals, [i] * len(vals), color=SHADE[i % 6], marker=MARK[i % 6], s=24, alpha=0.85, zorder=3, edgecolor=SURF, linewidth=0.8)
        ax.axvline(float(reg[key]), color=FILL["red"], lw=1.2)
        ax.text(float(reg[key]), -0.55, "registered", color=FILL["red"], fontsize=7, ha="center", va="top")
        ax.set_yticks(range(len(blocks))); ax.set_yticklabels([b.replace("_", " ") for b in blocks], fontsize=7.5); ax.set_ylim(-0.9, len(blocks) - 0.4)
        ax.set_title(lab); ax.set_xlabel("%")
    save(fig, "F09_robustness_dots.png")

    A = [r for r in rob if r["block"].startswith("A")]
    cells2 = [("E2E_Novice", "Novice"), ("E2E_Tool_v3", "Tool v3"), ("E2E_Inc_Solo", "Incumbent"), ("E2E_Collab", "Settlement"), ("Tier2_Tool_v3", "Tier 2, v3"), ("Tier3_Inc_Solo", "Tier 3, incumbent"), ("Core_SQErr", "Core error")]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (key, lab) in enumerate(cells2):
        vals = [float(r[key]) for r in A]
        ax.scatter(vals, [i] * len(vals), color=SHADE[0], s=28, alpha=0.7, zorder=3, edgecolor=SURF, label="alternative universes" if i == 0 else None)
        ax.scatter([float(reg[key])], [i], color=FILL["red"], s=55, marker="D", zorder=4, label="registered universe" if i == 0 else None)
        sd = math.sqrt(sum((v - mean(vals)) ** 2 for v in vals) / max(1, len(vals) - 1))
        ax.text(max(vals) + 1.2, i, f"SD {sd:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(cells2))); ax.set_yticklabels([l for _, l in cells2]); ax.set_xlabel("%"); ax.legend(loc="lower right")
    save(fig, "F16_monte_carlo.png")


# ---------------------------------------------------------------------------------------------------------------
# F10 archive boundary (data_outputs/archive_boundary.csv, written by run_archive_boundary.py)
# ---------------------------------------------------------------------------------------------------------------
def f10_archive_boundary():
    conds = ["Registered:\nprecedent withheld", "Precedent archived,\nsame grammar", "Interaction\ntemplate added", "Interaction template\nand 4-decimal grid", "Same, archive\nof 320 cases"]
    ab = rows("archive_boundary.csv")
    if len(ab) != len(conds):
        raise RuntimeError(f"archive_boundary.csv has {len(ab)} conditions; Figure 10 expects {len(conds)}")
    t3 = [float(r["tier3_v3"]) for r in ab]; t2 = [float(r["tier2_v3"]) for r in ab]; e2e = [float(r["e2e_v3"]) for r in ab]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = range(len(conds)); w = 0.26
    ax.bar([i - w for i in x], t2, width=w - 0.02, color=SHADE[1], label="Tier 2 convention, v3", zorder=3)
    ax.bar(list(x), t3, width=w - 0.02, color=SHADE[0], label="Tier 3 precedent, v3", zorder=3)
    ax.bar([i + w for i in x], e2e, width=w - 0.02, color=SHADE[2], label="end to end, v3", zorder=3)
    for i, v in enumerate(t3):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(conds, fontsize=8); ax.set_ylim(0, 108); ax.set_ylabel("%"); ax.legend(loc="upper left", ncol=3)
    save(fig, "F10_archive_boundary.png")


# ---------------------------------------------------------------------------------------------------------------
# F11 dynamics; F12 labour; F14 feedback
# ---------------------------------------------------------------------------------------------------------------
def f11_dynamics():
    regs = ["mandatory", "discretion", "strategic"]
    data = {r: rows(f"dynamics_{r}.csv") for r in regs}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, ring in zip(axes, RINGS):
        for i, r in enumerate(regs):
            T = max(int(x["period"]) for x in data[r])
            ys = [mean([float(x["tier2_parity"]) for x in data[r] if x["ring"] == ring and int(x["period"]) == t and x["tier2_parity"] not in ("", "nan")]) for t in range(1, T + 1)]
            line(ax, range(1, T + 1), ys, i, {"mandatory": "Required publication", "discretion": "Voluntary publication", "strategic": "Strategic withholding"}[r], ms=3.5)
        ax.set_title(RING_LABEL[ring]); ax.set_xlabel("period"); ax.set_ylim(0, 105); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("accuracy on unwritten conventions, %"); axes[0].legend(loc="lower right")
    save(fig, "F11_dynamics_parity.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    d = data["strategic"]; T = max(int(x["period"]) for x in d)
    for i, ring in enumerate(RINGS):
        ys = [100.0 * mean([float(x["closed"]) for x in d if x["ring"] == ring and int(x["period"]) == t]) for t in range(1, T + 1)]
        line(ax, range(1, T + 1), ys, i, RING_LABEL[ring], ms=3.5)
    ax.set_ylim(0, 100); ax.set_xlabel("period"); ax.set_ylabel("tasks whose incumbents withhold the record, %"); ax.legend(); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "F11b_strategic_closure.png")


def f12_labour():
    regs = ["mandatory", "discretion", "strategic"]
    data = {r: rows(f"dynamics_labor_{r}.csv") for r in regs}
    metrics = [("learning_index", "A. Learning per apprentice (period 1 = 1)", 1.0), ("periods_to_eligibility", "B. Periods to eligibility", None), ("eligible_backlog", "C. Eligible apprentices waiting, %", 100.0)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (key, lab, scale) in zip(axes, metrics):
        for i, r in enumerate(regs):
            d = data[r]; T = max(int(x["period"]) for x in d)
            ys = [mean([float(x[key]) for x in d if int(x["period"]) == t and float(x[key]) != float("inf")]) for t in range(1, T + 1)]
            if scale == 100.0:
                ys = [100 * y for y in ys]
            line(ax, range(1, T + 1), ys, i, {"mandatory": "Required publication", "discretion": "Voluntary publication", "strategic": "Strategic withholding"}[r], ms=3.5)
        ax.set_title(lab); ax.set_xlabel("period"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key == "learning_index":
            ax.axhline(1.0, color=GRID, lw=0.8)
    axes[0].legend(loc="lower left")
    save(fig, "F12_labour.png")


def f14_feedback():
    regs = ["mandatory", "strategic"]
    data = {r: rows(f"dynamics_{r}_fb_both.csv") for r in regs}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for i, r in enumerate(regs):
        d = data[r]; T = max(int(x["period"]) for x in d)
        gates = [100.0 * mean([float(x["gate_state"]) for x in d if int(x["period"]) == t]) for t in range(1, T + 1)]
        sil = [mean([float(x["silent_error"]) for x in d if x["ring"] == "Inner_Core" and int(x["period"]) == t]) for t in range(1, T + 1)]
        vis = [mean([float(x["visible_error"]) for x in d if x["ring"] == "Inner_Core" and int(x["period"]) == t]) for t in range(1, T + 1)]
        wm = [mean([float(x["wage_mult"]) for x in d if int(x["period"]) == t]) for t in range(1, T + 1)]
        line(axes[0], range(1, T + 1), gates, i, r, ms=3.5)
        axes[1].plot(range(1, T + 1), sil, color=SHADE[i], ls="-", marker=MARK[i], ms=3.5, lw=1.8, label=f"{r}: silent")
        axes[1].plot(range(1, T + 1), vis, color=SHADE[i], ls=":", lw=1.4, label=f"{r}: visible")
        line(axes[2], range(1, T + 1), wm, i, r, ms=3.5)
    axes[0].set_title("A. Tasks with a statutory gate, % of 54"); axes[0].legend(); axes[0].set_ylim(0, 50)
    axes[1].set_title("B. Core error, %: silent (solid), visible (dotted)"); axes[1].legend(fontsize=7)
    axes[2].set_title("C. Incumbent wage multiplier"); axes[2].axhline(1.0, color=GRID, lw=0.8)
    for ax in axes:
        ax.set_xlabel("period"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "F14_feedback.png")


# ---------------------------------------------------------------------------------------------------------------
# F13 bumping chains
# ---------------------------------------------------------------------------------------------------------------
def f13_bumping():
    d = rows("bumping_extended.csv")
    waves = ["v0", "v1", "v2", "v3"]
    fig = plt.figure(figsize=(10.5, 4.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[4, .9])
    axes = [fig.add_subplot(grid[0, j]) for j in range(2)]
    legends = [fig.add_subplot(grid[1, j]) for j in range(2)]
    for panel in legends:
        panel.axis("off")
    ax = axes[0]
    for i, ring in enumerate(RINGS):
        base_i = mean([float(x["inc_hours"]) for x in d if x["ring"] == ring and x["wave"] == "v0"])
        base_s = mean([float(x["sub_hours"]) for x in d if x["ring"] == ring and x["wave"] == "v0"])
        hi = [100 * mean([float(x["inc_hours"]) for x in d if x["ring"] == ring and x["wave"] == w]) / base_i for w in waves]
        hs = [100 * mean([float(x["sub_hours"]) for x in d if x["ring"] == ring and x["wave"] == w]) / base_s for w in waves]
        ax.plot(range(4), hi, color=SHADE[i], lw=1.8, ls="-", marker=MARK[i], ms=4.5, label=RING_LABEL[ring])
        ax.plot(range(4), hs, color=SHADE[i], lw=1.8, ls="--", marker=MARK[i], ms=4.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Before\nautomation", "Written\nrules", "Learn from\ncases", "Refine\nexceptions"], fontsize=8)
    ax.set_ylabel("Hours per case (before automation = 100)"); ax.set_ylim(0, 110)
    ax.set_title("A. Professional and subordinate hours")
    h, labels = ax.get_legend_handles_labels()
    legends[0].legend(h, labels, loc="center", ncol=3, fontsize=7,
                      title="Solid: professional; dashed: subordinate", title_fontsize=8)
    ax = axes[1]
    arrs = [("POACH", SHADE[0], "Automation alone"),
            ("SETTLE_INC", SHADE[1], "Professional review of tool"),
            ("GATE", SHADE[2], "Professional execution"),
            ("SUB_ALONE", SHADE[3], "Subordinate alone"),
            ("SUB_SUPERVISED", SHADE[4], "Supervised subordinate"),
            ("SETTLE_SUB", SHADE[5], "Subordinate review of tool")]
    x = 0; ticks = []; labels = []
    for ring in RINGS:
        mid = x + .5
        for wave in ("v0", "v3"):
            sel = [r for r in d if r["ring"] == ring and r["wave"] == wave]
            bottom = 0
            for arrangement, col, label in arrs:
                value = 100.0 * sum(r["arrangement"] == arrangement for r in sel) / len(sel)
                ax.bar(x, value, bottom=bottom, color=col, edgecolor=SURF, linewidth=1.5,
                       width=.8, zorder=3, label=label if x == 0 else None)
                bottom += value
            ticks.append(x); labels.append("Before\nAI" if wave == "v0" else "With\nAI"); x += 1
        ax.text(mid, 105, RING_LABEL[ring].replace(" tasks", ""), ha="center", fontsize=8, weight="bold")
        x += .6
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cases, %"); ax.set_ylim(0, 112); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("B. Allocation of work")
    h, labels = ax.get_legend_handles_labels()
    legends[1].legend(h, labels, loc="center", ncol=2, fontsize=7)
    save(fig, "F13_bumping.png")


# ---------------------------------------------------------------------------------------------------------------
# F15 AI arm
# ---------------------------------------------------------------------------------------------------------------
def f15_ai_arm(traces):
    arm = json.load(open(os.path.join(D, "results_llm_arm_replay_primary_legacy_replay_traces.json")))
    roles = [("Novice", "base_novice", "arm_novice"), ("Tool v3", "base_tool_v3", "arm_tool_v3"), ("Incumbent alone", "base_inc_solo", "arm_inc_solo"), ("Settlement", "base_collab", "arm_collab")]
    tiers = [(0, "Tier 0"), (1, "Tier 1"), (2, "Tier 2"), (3, "Tier 3")]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), sharey=True)
    for ax, (lab, bk, ak) in zip(axes, roles):
        for j, (t, tl) in enumerate(tiers):
            sel = [x for x in arm if x["complexity_tier"] == t]
            b = pct([float(x[bk]) for x in sel]); a = pct([float(x[ak]) for x in sel])
            ax.bar(j - 0.18, b, width=0.34, color=SHADE[0], label="rule-based baseline" if j == 0 else None, zorder=3)
            ax.bar(j + 0.18, a, width=0.34, color=SHADE[1], label="model-backed" if j == 0 else None, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([tl for _, tl in tiers], fontsize=8); ax.set_title(lab); ax.set_ylim(0, 105)
    axes[0].set_ylabel("accuracy, %"); axes[0].legend(loc="lower left", fontsize=7)
    save(fig, "F15_ai_arm.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    kinds = [("Tier 0", lambda x: x["complexity_tier"] == 0), ("Tier 1", lambda x: x["complexity_tier"] == 1), ("Tier 2", lambda x: x["complexity_tier"] == 2),
             ("Tier 3 overt", lambda x: x["complexity_tier"] == 3 and not x["is_silent"]), ("Tier 3 silent", lambda x: x["is_silent"])]
    for ax, (title, data, key) in zip(axes, [("A. Rule-based coverage critic (reference configuration)", traces, "epistemic_uncertainty"), ("B. Model critic (compact live run)", arm, "arm_u_p")]):
        for j, (kl, f) in enumerate(kinds):
            vals = [float(x[key]) for x in data if f(x)]
            jitter = [(j + ((int(hashlib.md5((str(v) + str(i)).encode()).hexdigest(), 16) % 100) / 100 - 0.5) * 0.5) for i, v in enumerate(vals)]
            ax.scatter(jitter, vals, s=10, color=SHADE[0], alpha=0.35, zorder=3)
            ax.plot([j - 0.3, j + 0.3], [mean(vals)] * 2, color=FILL["red"], lw=2.2, zorder=4)
        ax.set_xticks(range(5)); ax.set_xticklabels([k for k, _ in kinds], fontsize=8); ax.set_title(title); ax.set_ylim(-0.03, 1.0)
    axes[0].set_ylabel("critic's probability that the program is wrong")
    save(fig, "F15b_critic.png")


def f17_f18_knowledge():
    """The knowledge race (F17) and competence at promotion (F18), from the learning-profession dynamics."""
    regs = [r for r in ("mandatory", "discretion", "strategic") if os.path.exists(os.path.join(D, f"dynamics_{r}_learning.csv"))]
    if not regs:
        return
    data = {r: rows(f"dynamics_{r}_learning.csv") for r in regs}
    know = {r: rows(f"dynamics_knowledge_{r}_learning.csv") for r in regs}
    gated = {x["occ_title"]: int(x["statutory_gate"]) for r in regs for x in know[r]}
    fig, axes = plt.subplots(1, len(regs), figsize=(3.7 * len(regs) + 0.6, 3.6), sharey=True)
    axes = list(axes) if len(regs) > 1 else [axes]
    for ax, r in zip(axes, regs):
        d = data[r]; T = max(int(x["period"]) for x in d)
        for j, (grp, pred) in enumerate((("licensed", lambda x: gated.get(x["occ_title"], 0) == 1), ("unlicensed", lambda x: gated.get(x["occ_title"], 0) == 0))):
            sub = [x for x in d if pred(x)]
            par = [mean([float(x["tier2_parity"]) for x in sub if int(x["period"]) == t and x["tier2_parity"] not in ("", "nan")]) for t in range(1, T + 1)]
            conv = [100.0 * mean([float(x["s_conv"]) for x in sub if int(x["period"]) == t]) for t in range(1, T + 1)]
            reg = [100.0 * mean([float(x["s_reg"]) for x in sub if int(x["period"]) == t]) for t in range(1, T + 1)]
            ls = "-" if grp == "licensed" else "--"
            ax.plot(range(1, T + 1), par, color=SHADE[0], ls=ls, lw=1.8, marker=MARK[0], ms=3.2, label=f"program's convention parity, {grp}")
            ax.plot(range(1, T + 1), conv, color=SHADE[1], ls=ls, lw=1.8, marker=MARK[1], ms=3.2, label=f"seniors holding the convention, {grp}")
            ax.plot(range(1, T + 1), reg, color=SHADE[2], ls=ls, lw=1.4, marker=MARK[2], ms=3.0, label=f"seniors holding the registry, {grp}")
        ax.set_title(r); ax.set_xlabel("period"); ax.set_ylim(0, 105); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("share, %")
    axes[0].legend(loc="lower right", fontsize=6.3, frameon=True, framealpha=0.9)
    save(fig, "F17_knowledge_race.png")

    # F18: where reproduction breaks, licensed occupations: the registered calibration against the runs that outrun it
    variants = [("reference assumptions, 32 periods", "mandatory_learning_t32"), ("learning 1.5x slower", "mandatory_learning_k15"),
                ("retirement 10% a period", "mandatory_learning_delta10"), ("demand growth 5% a period", "mandatory_learning_growth05"),
                ("learning 1.5x slower, no sign-off rule", "mandatory_learning_k15_gates_none")]
    variants = [(lab, tag) for lab, tag in variants if os.path.exists(os.path.join(D, f"dynamics_{tag}.csv"))]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for i, (lab, tag) in enumerate(variants):
        d = [x for x in rows(f"dynamics_{tag}.csv") if gated.get(x["occ_title"], 0) == 1]
        T = max(int(x["period"]) for x in d)
        conv = [100.0 * mean([float(x["s_conv"]) for x in d if int(x["period"]) == t]) for t in range(1, T + 1)]
        hard = [sum(int(x["hard_n"]) for x in d if int(x["period"]) == t) or 1 for t in range(1, T + 1)]
        unk = [100.0 * sum(int(x["unknowing_handled"]) for x in d if int(x["period"]) == t) / hard[t - 1] for t in range(1, T + 1)]
        line(axes[0], range(1, T + 1), conv, i, lab, ms=3.5)
        line(axes[1], range(1, T + 1), unk, i, lab, ms=3.5)
    axes[0].set_title("A. Seniors with convention knowledge"); axes[0].set_ylim(50, 102); axes[0].set_ylabel("share of seniors, %")
    axes[0].axvline(16, color=GRID, lw=1.0); axes[1].axvline(16, color=GRID, lw=1.0)
    axes[1].set_title("B. Cases signed without required knowledge"); axes[1].set_ylim(0, 18); axes[1].set_ylabel("share of hard cases, %")
    for ax in axes:
        ax.set_xlabel("period"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(loc="lower left", fontsize=7.5)
    save(fig, "F18_hollow_signature.png")


def f19_closure_formation():
    """Compare predicted pre-automation licensing with the supplied status and chance."""
    data = rows("closure_formation_game.csv")
    conditions = [("capture", "", "Lobbying only"),
                  ("public interest", "0.02", "Visible error threshold: 2%"),
                  ("public interest", "0.05", "Visible error threshold: 5%"),
                  ("harm-weighted", "0.5", "Expected harm threshold: 0.50")]
    selected = [next(r for r in data if r["wave"] == "v0" and float(r["lobby_cost"]) == .05
                     and r["regulator"] == regulator and r["threshold"] == threshold)
                for regulator, threshold, _ in conditions]
    fig, ax = plt.subplots(figsize=(9, 4.1))
    ys = list(range(len(conditions)))
    agreement = [100 * float(r["agreement"]) for r in selected]
    chance = [100 * float(r["chance"]) for r in selected]
    ax.barh([y-.16 for y in ys], agreement, height=.28, color=SHADE[0], label="Model agreement")
    ax.barh([y+.16 for y in ys], chance, height=.28, color=SHADE[1], label="Chance agreement")
    for y, a, c in zip(ys, agreement, chance):
        ax.text(a+1, y-.16, f"{a:.1f}%", va="center", fontsize=8)
        ax.text(c+1, y+.16, f"{c:.1f}%", va="center", fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels([x[2] for x in conditions]); ax.invert_yaxis()
    ax.set_xlim(0, 65); ax.set_xlabel("Agreement with supplied licensing status, % of 18 occupations")
    ax.set_title("Licensing formation before automation")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, fontsize=8)
    save(fig, "F19_closure_formation.png")


def f20_f21_levers():
    """F20: the fee-setting incumbent (fee paths by ring and regime); F21: the labour block when the program replaces junior hours."""
    regs = [r for r in ("mandatory", "discretion", "strategic") if os.path.exists(os.path.join(D, f"dynamics_{r}_fee.csv"))]
    if regs:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
        for ax, ring in zip(axes, RINGS):
            for i, r in enumerate(regs):
                d = rows(f"dynamics_{r}_fee.csv"); T = max(int(x["period"]) for x in d)
                ys = [mean([float(x["fee"]) for x in d if x["ring"] == ring and int(x["period"]) == t]) for t in range(1, T + 1)]
                line(ax, range(1, T + 1), ys, i, {"mandatory":"Required publication", "discretion":"Voluntary publication", "strategic":"Strategic withholding"}[r], ms=3.5)
            reg = mean([float(x["fee_registered"]) for x in rows(f"dynamics_{regs[0]}_fee.csv") if x["ring"] == ring])
            ax.axhline(reg, color=INK2, lw=0.9, ls=":"); ax.text(1, reg + 0.005, "reference review charge", fontsize=7, color=INK2)
            ax.set_title(RING_LABEL[ring]); ax.set_xlabel("period"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylabel("review fee (share of the incumbent's wage)"); axes[0].legend(loc="upper right", fontsize=7.5)
        save(fig, "F20_fee_setting.png")
    if os.path.exists(os.path.join(D, "dynamics_labor_mandatory_junior_replaced.csv")):
        base, alt = rows("dynamics_labor_mandatory.csv"), rows("dynamics_labor_mandatory_junior_replaced.csv")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        metrics = [("J_demand", "A. Junior demand (period 1 = 100)", True), ("periods_to_eligibility", "B. Periods to eligibility", False), ("I_demand", "C. Incumbent demand (period 1 = 100)", True)]
        for ax, (key, lab, index) in zip(axes, metrics):
            for i, (name, d) in enumerate((("juniors operate the program (registered)", base), ("the program replaces junior hours", alt))):
                T = max(int(x["period"]) for x in d)
                ys = [mean([float(x[key]) for x in d if int(x["period"]) == t and float(x[key]) < 1e6]) for t in range(1, T + 1)]
                if index:
                    ys = [100 * y / ys[0] for y in ys]
                line(ax, range(1, T + 1), ys, i, name, ms=3.5)
            ax.set_title(lab); ax.set_xlabel("period"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].legend(loc="upper left", fontsize=7.5)
        save(fig, "F21_junior_replacement.png")


def main():
    traces = json.load(open(os.path.join(D, "results_packet_traces_extended_1080.json")))
    tasks = rows("results_extended_18occ_tasks.csv")
    rob = rows("robustness_summary.csv")
    f01_architecture()
    f02_f03_ceilings(traces)
    f04_f05_compiler(rob, tasks)
    f06_governance(traces)
    f07_cost_curves()
    f08_occupations(tasks)
    f09_f16_robustness(rob)
    f10_archive_boundary()
    f11_dynamics()
    f12_labour()
    f13_bumping()
    f14_feedback()
    f15_ai_arm(traces)
    f17_f18_knowledge()
    f19_closure_formation()
    f20_f21_levers()


if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(description='Draws the appendix figures into submission/figures/').parse_args()   # no options; rejects stray arguments and serves --help
    main()
