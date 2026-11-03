#!/usr/bin/env python3
"""
Module 4: End-to-End Benchmark Runner & Econometric Suite (run_live_benchmark.py)
JURI Lab — executed architecture

Phase A  Specification-driven compilation with closed-loop repair (v1 handbook compile, v2 linear handler,
         v3 piecewise handler) for 54 tasks; 162 scripts with AST / cyclomatic / Halstead metrics.
Phase B  Six architectures on the evaluation packets (default 20 per task = 1,080), then the three-agent
         governance game under the status quo and two counterfactual regimes. The game is also solved for
         each compiler generation as the challenger's available tool (Waves 1-3).
Phase C  Importance-weighted occupation trajectories and implied wage-bill compression.
Phase D  Micro-to-macro OLS bridge with HC1 and occupation-clustered (CR1) inference, an occupation-level
         (N = 18) specification and a permutation p-value. Macro outcomes come from data_inputs/macro_outcomes.csv; the
         run prints their provenance and refuses to call Table 4 a validation while any row is a placeholder.

`run_full()` returns a summary dictionary and is what run_robustness_suite.py calls repeatedly.
"""

import csv
import json
import math
import os
import juri_paths
from collections import Counter
from typing import Dict, List, Optional, Tuple

from model_config import CFG, describe_config, config_diff
from domain_cases import (
    ONETTaskSpec,
    PRIMARY_CORPUS,
    FULL_EXTENDED_CORPUS,
    MACRO_SOURCE_COUNTS,
    build_multi_doc_packets_for_task,
    build_public_sandbox_archive,
)
from compiler_agent import synthesize_compiled_tool_from_handbook, load_compiled_callable
from negotiated_order_lab import (
    execute_unassisted_novice,
    execute_hybrid_tool_plus_critic,
    execute_incumbent_react_agent,
    grade_agent_output,
    negotiate_workplace_governance,
)


BASE_DIR = juri_paths.ROOT
TOOLS_DIR = os.path.join(BASE_DIR, "compiled_tools")
OUT_DIR = os.path.join(BASE_DIR, "data_outputs")
RINGS = ("Outer_Periphery", "Middle_Buffer", "Inner_Core")
WAVES = ("v1", "v2", "v3")


# ---------------------------------------------------------------------------
# Statistics (pure standard library)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    max_it, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= fpmin else fpmin)
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= fpmin else fpmin)
        c = 1.0 + aa / (c if abs(c) >= fpmin else fpmin)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= fpmin else fpmin)
        c = 1.0 + aa / (c if abs(c) >= fpmin else fpmin)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _reg_inc_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    front = math.exp(ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t_stat: float, dof: float) -> float:
    if dof <= 0:
        return 1.0
    t_abs = abs(t_stat)
    return _reg_inc_beta(dof / 2.0, 0.5, dof / (dof + t_abs * t_abs))


def _matmul(A, B):
    return [[sum(A[i][m] * B[m][j] for m in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _invert(M):
    size = len(M)
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for i in range(size):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            for r in range(i + 1, size):
                if abs(aug[r][i]) > 1e-12:
                    aug[i], aug[r] = aug[r], aug[i]
                    pivot = aug[i][i]
                    break
        for j in range(2 * size):
            aug[i][j] /= pivot
        for r in range(size):
            if r != i:
                factor = aug[r][i]
                for j in range(2 * size):
                    aug[r][j] -= factor * aug[i][j]
    return [row[size:] for row in aug]


def _ols_core(X_rows, y):
    n = len(y)
    X = [[1.0] + list(row) for row in X_rows]
    k = len(X[0])
    XT = _transpose(X)
    XTX_inv = _invert(_matmul(XT, X))
    XTy = [[sum(XT[j][i] * y[i] for i in range(n))] for j in range(k)]
    betas = [b[0] for b in _matmul(XTX_inv, XTy)]
    resids = [y[i] - sum(X[i][j] * betas[j] for j in range(k)) for i in range(n)]
    y_mean = sum(y) / n
    tss = sum((yi - y_mean) ** 2 for yi in y)
    r2 = 1.0 - sum(e * e for e in resids) / tss if tss > 0 else 0.0
    return X, n, k, XTX_inv, betas, resids, round(r2, 4)


def ols_hc1(X_rows, y):
    """OLS with White HC1 robust SEs and t(N-K) p-values."""
    X, n, k, XTX_inv, betas, resids, r2 = _ols_core(X_rows, y)
    dof_scale = n / float(n - k)
    meat = [[0.0] * k for _ in range(k)]
    for i in range(n):
        e2 = (resids[i] ** 2) * dof_scale
        for r_idx in range(k):
            for c_idx in range(k):
                meat[r_idx][c_idx] += e2 * X[i][r_idx] * X[i][c_idx]
    cov = _matmul(_matmul(XTX_inv, meat), XTX_inv)
    ses = [math.sqrt(max(cov[j][j], 1e-16)) for j in range(k)]
    return betas, ses, [t_two_sided_p(abs(betas[j]) / ses[j], n - k) for j in range(k)], r2


def ols_cluster_robust(X_rows, y, cluster_ids):
    """OLS with Liang-Zeger CR1 SEs, G/(G-1)*(N-1)/(N-K) correction and t(G-1) p-values."""
    X, n, k, XTX_inv, betas, resids, _ = _ols_core(X_rows, y)
    clusters = sorted(set(cluster_ids))
    G = len(clusters)
    meat = [[0.0] * k for _ in range(k)]
    for g in clusters:
        score = [0.0] * k
        for i in range(n):
            if cluster_ids[i] == g:
                for j in range(k):
                    score[j] += X[i][j] * resids[i]
        for r_idx in range(k):
            for c_idx in range(k):
                meat[r_idx][c_idx] += score[r_idx] * score[c_idx]
    correction = (G / float(G - 1)) * ((n - 1) / float(n - k))
    cov = _matmul(_matmul(XTX_inv, meat), XTX_inv)
    ses = [math.sqrt(max(cov[j][j] * correction, 1e-16)) for j in range(k)]
    return betas, ses, [t_two_sided_p(abs(betas[j]) / ses[j], G - 1) for j in range(k)], G


def permutation_p(X_rows, y, col: int, n_perm: int = 2000, seed: int = 7, cluster_ids=None) -> float:
    """Two-sided permutation p-value for coefficient `col` (1-based after intercept) by permuting y.
    With `cluster_ids`, whole clusters are permuted (each cluster's block of y values moves together), which
    is the right null when y is an occupation-level value repeated across the occupation's tasks."""
    import random
    rng = random.Random(seed)
    obs = abs(_ols_core(X_rows, y)[4][col])
    hits = 0
    if cluster_ids is None:
        yy = list(y)
        for _ in range(n_perm):
            rng.shuffle(yy)
            if abs(_ols_core(X_rows, yy)[4][col]) >= obs:
                hits += 1
        return (hits + 1) / (n_perm + 1)
    clusters = sorted(set(cluster_ids))
    members = {g: [i for i, c in enumerate(cluster_ids) if c == g] for g in clusters}
    blocks = [[y[i] for i in members[g]] for g in clusters]
    for _ in range(n_perm):
        order = list(range(len(clusters)))
        rng.shuffle(order)
        yy = list(y)
        for g, src in zip(clusters, order):
            blk = blocks[src]
            for pos, i in enumerate(members[g]):
                yy[i] = blk[pos % len(blk)]
        if abs(_ols_core(X_rows, yy)[4][col]) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Phase A + B
# ---------------------------------------------------------------------------
def run_corpus_evaluation(corpus: List[ONETTaskSpec], label: str = "EXTENDED_18OCC", tools_dir: str = TOOLS_DIR):
    os.makedirs(tools_dir, exist_ok=True)
    task_rows: List[Dict[str, object]] = []
    packet_traces: List[Dict[str, object]] = []
    compile_logs: Dict[str, List[str]] = {}
    template_winners: Counter = Counter()

    for spec in corpus:
        packets = build_multi_doc_packets_for_task(spec, split="eval")
        archive = build_public_sandbox_archive(spec)
        handbook = packets[0].doc_handbook
        art = {}
        art["v1"] = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, tools_dir, "v1")
        art["v2"] = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, tools_dir, "v2", sandbox=archive, previous=art["v1"])
        art["v3"] = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, tools_dir, "v3", sandbox=archive, previous=art["v2"])
        compile_logs[spec.task_id] = art["v3"].repair_log
        fn = {w: load_compiled_callable(art[w]) for w in WAVES}
        learned = art["v3"].ir.get("learned", [])
        h = learned[0] if learned else None
        if h:
            template_winners[h["template"]] += 1

        scores = {k: [] for k in ("A_Base", "A_Tool_v1", "A_Tool_v2", "A_Tool_v3", "A_Hybrid_v3", "A_Inc_Solo", "A_Collab")}
        settlements = {w: [] for w in WAVES}
        hyb_e2e = {w: [] for w in WAVES}
        n_flag_v1 = 0
        for pkt in packets:
            d_in, d_sc, d_an = pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex
            out_base = execute_unassisted_novice(spec, d_in, d_sc, d_an)
            out_tool = {w: fn[w](d_in, d_sc, d_an) for w in WAVES}
            n_flag_v1 += bool(out_tool["v1"].get("validation_flags"))
            out_hyb = {w: execute_hybrid_tool_plus_critic(spec, fn[w], d_in, d_sc, d_an) for w in WAVES}
            out_inc_solo = execute_incumbent_react_agent(spec, pkt, mode="SOLO")
            out_collab = {w: execute_incumbent_react_agent(spec, pkt, mode="COLLAB_AUDIT", challenger_out=out_tool[w]) for w in WAVES}
            g = {"A_Base": grade_agent_output(out_base, pkt), "A_Inc_Solo": grade_agent_output(out_inc_solo, pkt),
                 "A_Collab": grade_agent_output(out_collab["v3"], pkt), "A_Hybrid_v3": grade_agent_output(out_hyb["v3"], pkt)}
            for w in WAVES:
                g[f"A_Tool_{w}"] = grade_agent_output(out_tool[w], pkt)
                hyb_e2e[w].append(grade_agent_output(out_hyb[w], pkt)["e2e_ok"])
                settlements[w].append(negotiate_workplace_governance(spec, pkt, out_hyb[w], out_collab[w], out_inc_solo))
            for k in scores:
                scores[k].append(g[k])
            settle = settlements["v3"][-1]
            packet_traces.append({
                "corpus_label": label, "packet_id": pkt.packet_id, "task_id": spec.task_id, "occ_soc": spec.occ_soc,
                "occ_title": spec.occ_title, "closure_tier": spec.closure_tier, "ring": spec.ring,
                "domain_family": spec.domain_family, "complexity_tier": pkt.complexity_tier,
                "is_collision": pkt.is_collision_case, "is_silent": pkt.is_silent_collision,
                "gt_diagnosis": pkt.gt_diagnosis, "gt_inference": pkt.gt_inference, "gt_treatment": pkt.gt_treatment,
                "A_Base_d": g["A_Base"]["d_ok"], "A_Base_t": g["A_Base"]["t_ok"], "A_Base_e2e": g["A_Base"]["e2e_ok"],
                "A_Tool_v1_d": g["A_Tool_v1"]["d_ok"], "A_Tool_v1_e2e": g["A_Tool_v1"]["e2e_ok"],
                "A_Tool_v2_d": g["A_Tool_v2"]["d_ok"], "A_Tool_v2_e2e": g["A_Tool_v2"]["e2e_ok"],
                "A_Tool_v3_d": g["A_Tool_v3"]["d_ok"], "A_Tool_v3_t": g["A_Tool_v3"]["t_ok"], "A_Tool_v3_e2e": g["A_Tool_v3"]["e2e_ok"],
                "A_Hybrid_v3_e2e": g["A_Hybrid_v3"]["e2e_ok"],
                "A_Inc_Solo_d": g["A_Inc_Solo"]["d_ok"], "A_Inc_Solo_t": g["A_Inc_Solo"]["t_ok"], "A_Inc_Solo_e2e": g["A_Inc_Solo"]["e2e_ok"],
                "A_Collab_d": g["A_Collab"]["d_ok"], "A_Collab_e2e": g["A_Collab"]["e2e_ok"],
                "inc_solo_mode": out_inc_solo.get("mode", ""), "inc_solo_hop": out_inc_solo.get("hop", ""),
                "collab_mode": out_collab["v3"].get("mode", ""), "tool_v3_fired": out_tool["v3"].get("fired", ""),
                "tool_v3_unhandled": ";".join(out_tool["v3"].get("unhandled_fields", [])),
                "epistemic_uncertainty": settle.epistemic_uncertainty, "critic_status": out_hyb["v3"].get("critic_status", ""),
                "negotiated_governance": settle.selected_governance, "sq_accurate": settle.final_output_accurate,
                "labor_cost_units": settle.labor_cost_units, "incumbent_hours_saved_pct": settle.incumbent_hours_saved_pct,
                "cf1_governance": settle.cf1_governance, "cf1_accurate": settle.cf1_accurate,
                "cf2_governance": settle.cf2_governance, "cf2_accurate": settle.cf2_accurate,
                "wave1_governance": settlements["v1"][-1].selected_governance, "wave2_governance": settlements["v2"][-1].selected_governance,
            })

        n_pkts = len(packets)

        def _pct(agent_k: str, fld: str = "e2e_ok") -> float:
            return round(100.0 * sum(x[fld] for x in scores[agent_k]) / n_pkts, 1)

        def _share(sts, pred) -> float:
            return round(100.0 * sum(1 for s in sts if pred(s)) / n_pkts, 1)

        wave_cols: Dict[str, object] = {}
        for w in WAVES:
            sts = settlements[w]
            avg_cost = round(math.fsum(s.labor_cost_units for s in sts) / n_pkts, 3)
            wave_cols[f"W_{w}_Poach_Pct"] = _share(sts, lambda s: s.selected_governance == "DIRECT_CHALLENGER_POACHING")
            wave_cols[f"W_{w}_Settle_Pct"] = _share(sts, lambda s: s.selected_governance == "ACCOUNTABILITY_SETTLEMENT")
            wave_cols[f"W_{w}_Gate_Pct"] = _share(sts, lambda s: s.selected_governance == "INCUMBENT_FULL_GATEKEEPING")
            wave_cols[f"W_{w}_SQ_Error_Pct"] = _share(sts, lambda s: s.final_output_accurate == 0)
            wave_cols[f"W_{w}_Delta_Ln_WageBill"] = round(math.log(avg_cost / spec.wage_ratio_inc_chall), 4)
            wave_cols[f"W_{w}_Hybrid_Pct"] = round(100.0 * sum(hyb_e2e[w]) / n_pkts, 1)

        sts3 = settlements["v3"]
        poach_rate = wave_cols["W_v3_Poach_Pct"]
        tech_parity_v3 = _pct("A_Hybrid_v3")
        row = {
            "occ_soc": spec.occ_soc, "occ_title": spec.occ_title, "job_zone": spec.job_zone, "closure_tier": spec.closure_tier,
            "domain_family": spec.domain_family, "task_id": spec.task_id, "ring": spec.ring, "overlap_j": spec.overlap_j,
            "task_importance_wt": spec.task_importance_wt, "statutory_gate": spec.statutory_gate,
            "cps_licensing_rate": spec.cps_licensing_rate, "wage_ratio_inc_chall": spec.wage_ratio_inc_chall,
            "accountability_z": spec.accountability_z, "collision_density": spec.collision_density,
            "archive_n_cases": len(archive), "archive_n_tier2": sum(1 for p in archive if p.complexity_tier == 2),
            "v2_handler_key": h["key"] if h else "", "v2_handler_template": h["template"] if h else "",
            "v2_handler_coefs": str(tuple(h["coefs"])) if h else "",
            "v3_split": (f'{h["split"]["feature"]}>={h["split"]["thr"]}' if h and h.get("split") else ""),
            "ast_nodes_v1": art["v1"].ast_node_count, "ast_nodes_v2": art["v2"].ast_node_count, "ast_nodes_v3": art["v3"].ast_node_count,
            "ast_cyclo_v3": art["v3"].cyclomatic_complexity, "halstead_vol_v1": art["v1"].halstead_volume,
            "halstead_vol_v3": art["v3"].halstead_volume,
            # structural metrics without the inactive validation-flag scaffold, and how often it fires
            "ast_nodes_v3_active": art["v3"].ast_node_count_active, "halstead_vol_v3_active": art["v3"].halstead_volume_active,
            "n_validation_checks": art["v3"].n_validation_checks, "validation_flag_rate_v1": round(100.0 * n_flag_v1 / len(packets), 1),
            "A_Base_Pct": _pct("A_Base"), "A_Tool_v1_Pct": _pct("A_Tool_v1"), "A_Tool_v2_Pct": _pct("A_Tool_v2"),
            "A_Tool_v3_Pct": _pct("A_Tool_v3"), "A_Hybrid_v3_Pct": tech_parity_v3, "A_Inc_Solo_Pct": _pct("A_Inc_Solo"),
            "A_Collab_Pct": _pct("A_Collab"),
            "Direct_Poaching_Pct": poach_rate, "Accountability_Settlement_Pct": wave_cols["W_v3_Settle_Pct"],
            "Incumbent_Gatekeeping_Pct": wave_cols["W_v3_Gate_Pct"],
            "Institutional_Shield_Gap_pp": round(tech_parity_v3 - poach_rate, 1),
            "Incumbent_Hours_Saved_Pct": round(sum(s.incumbent_hours_saved_pct for s in sts3) / n_pkts, 1),
            "Implied_Delta_Ln_WageBill": wave_cols["W_v3_Delta_Ln_WageBill"],
            "SQ_Error_Rate_Pct": wave_cols["W_v3_SQ_Error_Pct"],
            "CF1_Dereg_Poach_Pct": _share(sts3, lambda s: s.cf1_governance == "DIRECT_CHALLENGER_POACHING"),
            "CF1_Dereg_Error_Pct": _share(sts3, lambda s: s.cf1_accurate == 0),
            "CF2_NoLiab_Error_Pct": _share(sts3, lambda s: s.cf2_accurate == 0),
            "macro_auto_delta_19_26": spec.macro_auto_delta_19_26, "macro_telemetry_gap_z": spec.macro_telemetry_gap_z,
            "macro_semantic_shift_z": spec.macro_semantic_shift_z, "macro_source": spec.macro_source,
        }
        row.update(wave_cols)
        task_rows.append(row)
    return task_rows, packet_traces, compile_logs, template_winners


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _rate(subset, col) -> float:
    return round(100.0 * sum(int(x[col]) for x in subset) / len(subset), 1) if subset else 0.0


def _ring_mean(rows, ring, col) -> float:
    rr = [r for r in rows if r["ring"] == ring]
    return round(math.fsum(float(r[col]) for r in rr) / len(rr), 1)


# ---------------------------------------------------------------------------
# Full run: tables, files, summary
# ---------------------------------------------------------------------------
def run_full(tools_dir: str = TOOLS_DIR, out_dir: Optional[str] = OUT_DIR, quiet: bool = False) -> Dict[str, object]:
    say = (lambda *a, **k: None) if quiet else print
    ext_tasks, ext_traces, compile_logs, winners = run_corpus_evaluation(FULL_EXTENDED_CORPUS, "EXTENDED_18OCC", tools_dir)
    primary_ids = {s.task_id for s in PRIMARY_CORPUS}
    summary: Dict[str, object] = {"config_diff": config_diff(), "n_packets": len(ext_traces)}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(os.path.join(out_dir, "results_task_level_primary.csv"), [r for r in ext_tasks if r["task_id"] in primary_ids])
        _write_csv(os.path.join(out_dir, "results_extended_18occ_tasks.csv"), ext_tasks)
        with open(os.path.join(out_dir, "results_packet_traces_primary_360.json"), "w") as f:
            json.dump([dict(t, corpus_label="PRIMARY_6OCC") for t in ext_traces if t["task_id"] in primary_ids], f, indent=2)
        with open(os.path.join(out_dir, "results_packet_traces_extended_1080.json"), "w") as f:
            json.dump(ext_traces, f, indent=2)
        with open(os.path.join(out_dir, "compiler_repair_logs.json"), "w") as f:
            json.dump(compile_logs, f, indent=2)

    # ---- model card ----
    say("=" * 120)
    say("MODEL CARD: INFORMATION SETS & BEHAVIOURAL PARAMETERS (model_config.py; every hand-set constant)")
    say("=" * 120)
    say("Challenger compiler sees: public handbook + public resolved-case archive (Tiers 0-2). Incumbent sees: full rulebook + "
        "private registry. Novice sees: handbook only. Critic sees: the script's own coverage report.")
    for k, v in describe_config():
        say(f"  {k:<32} {v}")
    n_v2 = sum(1 for r in ext_tasks if r["v2_handler_key"])
    n_v3 = sum(1 for r in ext_tasks if r["v3_split"])
    n_t2 = sum(1 for r in ext_tasks if r["archive_n_tier2"] > 0)
    say(f"Compiler repair summary: {n_t2}/54 tasks have Tier 2 cases in the public archive; v2 induced a handler on {n_v2} "
        f"(template winners: {dict(winners)}); v3 induced a piecewise split on {n_v3}.")
    say("Macro outcome provenance (data_inputs/macro_outcomes.csv):", MACRO_SOURCE_COUNTS)
    summary["compiler"] = {"n_tier2_tasks": n_t2, "n_v2": n_v2, "n_v3": n_v3, "winners": dict(winners)}

    # ---- Table 1 ----
    tiers = {t: [x for x in ext_traces if x["complexity_tier"] == t] for t in range(4)}
    cols = ("A_Base", "A_Tool_v1", "A_Tool_v2", "A_Tool_v3", "A_Inc_Solo", "A_Collab")
    t1_rows = [("1. Diagnosis Accuracy (D)", ext_traces, "d"), ("2a. Tier 0: Routine Separable", tiers[0], "e2e"),
               ("2b. Tier 1: Explicit Annex Alert", tiers[1], "e2e"), ("2c. Tier 2: 1st-Order Piecewise Collision", tiers[2], "e2e"),
               ("2d. Tier 3: 2nd-Order Precedent Collision", tiers[3], "e2e"), ("3. All Multi-Doc Packets E2E", ext_traces, "e2e")]
    say("\n" + "=" * 120)
    say(f"TABLE 1: DISAGGREGATING ABBOTT'S TRIAD ACROSS 3 COMPILER GENERATIONS & BOUNDED RATIONALITY (N = {len(ext_traces)})")
    say("=" * 120)
    say(f"{'Cognitive / Epistemic Regime':<52} | {'Novice':>7} | {'Tool_v1':>7} | {'Tool_v2':>7} | {'Tool_v3':>7} | {'Inc_Solo':>8} | {'Collab_AS':>9}")
    say("-" * 120)
    summary["table1"] = {}
    for name, subset, fld in t1_rows:
        vals = {c: _rate(subset, c + "_" + fld) for c in cols}
        summary["table1"][name] = dict(vals, N=len(subset))
        say(f"{name + f' (N={len(subset)})':<52} | " + " | ".join(f"{vals[c]:>{w}.1f}%" for c, w in zip(cols, (6, 6, 6, 6, 7, 8))))
    silent = [x for x in ext_traces if x["is_silent"]]
    say(f"   Silent collisions (N={len(silent)}): Tool_v3 {_rate(silent, 'A_Tool_v3_e2e')}% | Inc_Solo {_rate(silent, 'A_Inc_Solo_e2e')}% | "
        f"Collab_AS {_rate(silent, 'A_Collab_e2e')}% | critic u_p on silent = {Counter(x['epistemic_uncertainty'] for x in silent)}")

    # ---- Table 2 ----
    say("\n" + "=" * 140)
    say("TABLE 2: CONCENTRIC RINGS — HALSTEAD VOLUME, COMPILER CO-EVOLUTION, WORKPLACE GOVERNANCE & COUNTERFACTUALS")
    say("=" * 140)
    say(f"{'Concentric Ring':<16} | {'Halst_V':>7} | {'Tool_v1':>7} | {'Tool_v2':>7} | {'Tool_v3':>7} | {'IncSolo':>7} | {'Poach%':>7} | {'Settle%':>7} | {'Gate%':>6} | {'CF1_Poach':>9} | {'SQ_Err%':>7} | {'CF1_Err':>7} | {'CF2_Err':>7}")
    say("-" * 140)
    summary["rings"] = {}
    for ring in RINGS:
        m = lambda col: _ring_mean(ext_tasks, ring, col)
        summary["rings"][ring] = {c: m(c) for c in ("halstead_vol_v3", "A_Tool_v1_Pct", "A_Tool_v2_Pct", "A_Tool_v3_Pct", "A_Inc_Solo_Pct",
                                                    "Direct_Poaching_Pct", "Accountability_Settlement_Pct", "Incumbent_Gatekeeping_Pct",
                                                    "CF1_Dereg_Poach_Pct", "SQ_Error_Rate_Pct", "CF1_Dereg_Error_Pct", "CF2_NoLiab_Error_Pct")}
        say(f"{ring:<16} | {m('halstead_vol_v3'):7.1f} | {m('A_Tool_v1_Pct'):6.1f}% | {m('A_Tool_v2_Pct'):6.1f}% | {m('A_Tool_v3_Pct'):6.1f}% | {m('A_Inc_Solo_Pct'):6.1f}% | "
            f"{m('Direct_Poaching_Pct'):6.1f}% | {m('Accountability_Settlement_Pct'):6.1f}% | {m('Incumbent_Gatekeeping_Pct'):5.1f}% | "
            f"{m('CF1_Dereg_Poach_Pct'):8.1f}% | {m('SQ_Error_Rate_Pct'):6.1f}% | {m('CF1_Dereg_Error_Pct'):6.1f}% | {m('CF2_NoLiab_Error_Pct'):6.1f}%")

    # ---- Table 3 ----
    say("\n" + "=" * 140)
    say("TABLE 3: IMPORTANCE-WEIGHTED OCCUPATION AUDIT — COMPILER PARITY, GOVERNANCE, FEE COMPRESSION & ERRORS (18 OCCS)")
    say("=" * 140)
    say(f"{'Occupation (SOC)':<32} | {'Domain':<11} | {'Halst_V':>7} | {'TechPar':>7} | {'Poach%':>7} | {'Settle%':>7} | {'Gate%':>6} | {'ShieldGap':>9} | {'dLnWage':>8} | {'SQ_Err':>6}")
    say("-" * 140)
    occ_rows = []
    seen = []
    for r in ext_tasks:
        if r["occ_title"] not in seen:
            seen.append(r["occ_title"])
    for occ in seen:
        o_rows = [r for r in ext_tasks if r["occ_title"] == occ]
        wavg = lambda col: math.fsum(float(r[col]) * float(r["task_importance_wt"]) for r in o_rows)
        hv, tp, dp = round(wavg("halstead_vol_v3"), 1), round(wavg("A_Tool_v3_Pct"), 1), round(wavg("Direct_Poaching_Pct"), 1)
        row = {"occ_soc": o_rows[0]["occ_soc"], "occ_title": occ, "closure_tier": o_rows[0]["closure_tier"], "domain_family": o_rows[0]["domain_family"],
               "job_zone": o_rows[0]["job_zone"], "statutory_gate": o_rows[0]["statutory_gate"],
               "Weighted_Overlap_J": round(wavg("overlap_j"), 3),
               "Weighted_Halstead_Vol_v3": hv, "Weighted_Halstead_Vol_v3_active": round(wavg("halstead_vol_v3_active"), 1),
               "Tech_Parity_v3_Pct": tp, "Direct_Poaching_Pct": dp,
               "Accountability_Settlement_Pct": round(wavg("Accountability_Settlement_Pct"), 1),
               "Incumbent_Gatekeeping_Pct": round(wavg("Incumbent_Gatekeeping_Pct"), 1),
               "Institutional_Shield_Gap_pp": round(tp - dp, 1), "Incumbent_Hours_Saved_Pct": round(wavg("Incumbent_Hours_Saved_Pct"), 1),
               "Implied_Delta_Ln_WageBill": round(wavg("Implied_Delta_Ln_WageBill"), 3), "SQ_Error_Rate_Pct": round(wavg("SQ_Error_Rate_Pct"), 1),
               "CF2_Unregulated_Error_Pct": round(wavg("CF2_NoLiab_Error_Pct"), 1),
               "Macro_ONET_Auto_Delta_19_26_z": round(wavg("macro_auto_delta_19_26"), 3),
               "Macro_Telemetry_Adoption_Gap_z": round(wavg("macro_telemetry_gap_z"), 3),
               "Macro_Semantic_Task_Shift_z": round(wavg("macro_semantic_shift_z"), 3)}
        occ_rows.append(row)
        say(f"{occ + ' (' + row['occ_soc'] + ')':<32} | {row['domain_family']:<11} | {hv:7.1f} | {tp:6.1f}% | {dp:6.1f}% | {row['Accountability_Settlement_Pct']:6.1f}% | "
            f"{row['Incumbent_Gatekeeping_Pct']:5.1f}% | {row['Institutional_Shield_Gap_pp']:+8.1f}pp | {row['Implied_Delta_Ln_WageBill']:+8.3f} | {row['SQ_Error_Rate_Pct']:5.1f}%")
    if out_dir:
        _write_csv(os.path.join(out_dir, "results_occupation_summary_18.csv"), occ_rows)
    summary["occ"] = occ_rows

    # ---- Table 4 ----
    def _fmt_p(val: float) -> str:
        return "< .001" if val < 0.001 else (f"{val:.3f}".lstrip("0") if val < 1.0 else "1.000")

    shield = [float(r["Institutional_Shield_Gap_pp"]) / 100.0 for r in ext_tasks]
    X_ols = [[shield[i], float(r["overlap_j"]), float(r["job_zone"])] for i, r in enumerate(ext_tasks)]
    occ_cluster = [str(r["occ_soc"]) for r in ext_tasks]
    dvs = [("Model 1 (DV: 2019->2026 O*NET Degree of Automation Growth", "macro_auto_delta_19_26", "Macro_ONET_Auto_Delta_19_26_z"),
           ("Model 2 (DV: Workplace AI Telemetry Adoption Gap", "macro_telemetry_gap_z", "Macro_Telemetry_Adoption_Gap_z"),
           ("Model 3 (DV: 2019->2026 O*NET Semantic Task-Space Shift on AI Axis", "macro_semantic_shift_z", "Macro_Semantic_Task_Shift_z")]
    placeholder = any(not str(r["macro_source"]).startswith(("ONET", "AEI", "ASQ", "OBS")) for r in ext_tasks)
    missing = any(any(math.isnan(float(r[c])) for c in ("macro_auto_delta_19_26", "macro_telemetry_gap_z", "macro_semantic_shift_z")) for r in ext_tasks)
    say("\n" + "=" * 120)
    say("TABLE 4: MULTIVARIATE OLS MICRO-TO-MACRO BRIDGE (task level N = 54: HC1 t(N-K), CR1 t(G-1) in brackets; occupation level N = 18)")
    say("=" * 120)
    summary["table4"] = {}
    if missing:
        say("  Not estimated: data_inputs/macro_outcomes.csv has no sourced outcomes yet (source = PENDING_MERGE). The regression code is in")
        say("  place; merge the occupation-level adoption-gap and recomposition measures and the task-level semantic shift to run it.")
    if placeholder and not missing:
        say("  WARNING: data_inputs/macro_outcomes.csv still contains PLACEHOLDER rows. These coefficients describe placeholder values,")
        say("  not observed outcomes, and must not be reported as validation until the file carries sourced data.")
    X_occ = [[float(o["Institutional_Shield_Gap_pp"]) / 100.0, float(o["Weighted_Overlap_J"]), float(o["job_zone"])] for o in occ_rows]
    gate_occ = [float(o["statutory_gate"]) for o in occ_rows]
    X_occ_gate = [x + [g] for x, g in zip(X_occ, gate_occ)]
    # collinearity diagnostic: how much of the laboratory's shield gap is the typed statutory-gate dummy?
    _, _, _, r2_shield_on_gate = ols_hc1([[g] for g in gate_occ], [x[0] for x in X_occ])
    summary["table4_shield_on_gate_r2"] = r2_shield_on_gate
    for title, col, occ_col in ([] if missing else dvs):
        y = [float(r[col]) for r in ext_tasks]
        b, se, pv, r2 = ols_hc1(X_ols, y)
        _, cse, cpv, G = ols_cluster_robust(X_ols, y, occ_cluster)
        perm = permutation_p(X_ols, y, 1, n_perm=2000, cluster_ids=occ_cluster)
        yo = [float(o[occ_col]) for o in occ_rows]
        bo, seo, pvo, r2o = ols_hc1(X_occ, yo)
        bg, seg, pvg, r2g = ols_hc1(X_occ_gate, yo)
        summary["table4"][col] = {"isr_beta": round(b[1], 3), "isr_hc1_se": round(se[1], 3), "isr_cr1_se": round(cse[1], 3), "isr_cr1_p": round(cpv[1], 5),
                                  "isr_perm_p": round(perm, 4), "r2": r2, "occ_isr_beta": round(bo[1], 3), "occ_isr_p": round(pvo[1], 4), "occ_r2": r2o,
                                  "occ_isr_beta_gate_ctrl": round(bg[1], 3), "occ_isr_p_gate_ctrl": round(pvg[1], 4),
                                  "occ_gate_beta": round(bg[4], 3), "occ_gate_p": round(pvg[4], 4), "occ_r2_gate_ctrl": r2g}
        say(f"{title}, task-level R^2 = {r2:.3f}; occupation-level R^2 = {r2o:.3f}):")
        for j, nm in ((1, "Institutional Shield Gap (ISR)"), (2, "Concentric Task Overlap (J)   "), (3, "O*NET Job Zone (Skill Control)")):
            say(f"  - {nm}: beta = {b[j]:+.3f} (HC1 SE = {se[j]:.3f}, p = {_fmt_p(pv[j])})  [CR1 SE = {cse[j]:.3f}, p = {_fmt_p(cpv[j])}, G = {G}]")
        say(f"  - ISR block-permutation p (2,000 draws, occupations permuted) = {_fmt_p(perm)}; occupation-level ISR beta = {bo[1]:+.3f} (HC1 SE = {seo[1]:.3f}, p = {_fmt_p(pvo[1])}, N = 18)")
        say(f"  - occupation level with the statutory-gate dummy added: ISR beta = {bg[1]:+.3f} (p = {_fmt_p(pvg[1])}); gate beta = {bg[4]:+.3f} (p = {_fmt_p(pvg[4])}); R^2 = {r2g:.3f}")
    say("=" * 120)
    if not missing:
        say("  Reading: every outcome is an occupation-level value repeated across the occupation's three tasks, so the task-level")
        say("  rows are not independent; the occupation-level estimates (N = 18) and the cluster-robust bracketed ones are the ones to quote.")
        say(f"  The shield gap is largely the typed statutory gate: regressing it on the gate dummy alone gives R^2 = {r2_shield_on_gate:.2f}, so the")
        say("  gate-controlled line reports what the laboratory adds beyond the hand-typed institution; with 18 occupations it rarely separates.")

    # ---- Table 4b: the extended outcome file (occupation level, N = 18) ----
    ext_path = os.path.join(BASE_DIR, "data_inputs", "macro_outcomes_extended_lab54.csv")
    if not missing and os.path.exists(ext_path):
        with open(ext_path, encoding="utf-8-sig") as f:
            ext_rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
        # SOC codes in the hand-off file that the corpus writes differently (2018 SOC detail code for clinical psychologists)
        soc_alias = {"19-3031": "19-3033"}
        by_occ = {}
        outcome_cols = ("gap_int26_z", "gap_int25_z", "gap_massenkoff_z", "gap_microsoft_z", "auto_delta_19_26_rerated_z", "d_ln_emp_2019_2024", "d_ln_wage_2019_2024")
        for r in ext_rows:   # one row per task; the outcome columns are occupation-level and must agree across the occupation's tasks
            key = soc_alias.get(r["occ_soc"], r["occ_soc"])
            if key in by_occ:
                if any(by_occ[key].get(c) != r.get(c) for c in outcome_cols):
                    raise ValueError(f"macro_outcomes_extended_lab54.csv: occupation {key} has inconsistent outcome values across its tasks")
                continue
            by_occ[key] = r
        unmatched = [o["occ_soc"] for o in occ_rows if str(o["occ_soc"]) not in by_occ]
        if unmatched:
            raise ValueError(f"macro_outcomes_extended_lab54.csv has no row for corpus occupations {unmatched}")
        outcomes = [("gap_int26_z", "Adoption gap, Claude usage June 2026 (z)"), ("gap_int25_z", "Adoption gap, Claude usage Aug 2025 (z)"),
                    ("gap_massenkoff_z", "Adoption gap, Massenkoff-McCrory observed exposure (z)"), ("gap_microsoft_z", "Adoption gap, Microsoft applicability (z)"),
                    ("auto_delta_19_26_rerated_z", "O*NET Degree of Automation change 2019-26, re-rated occupations only (z)"),
                    ("d_ln_emp_2019_2024", "Employment change 2019-2024 (log points)"), ("d_ln_wage_2019_2024", "Median wage change 2019-2024 (log points)")]
        say("\n" + "=" * 120)
        say("TABLE 4b: OCCUPATION-LEVEL BRIDGE ON THE EXTENDED OUTCOME FILE (N = 18 occupations; OLS with HC1; shield gap, overlap, job zone)")
        say("=" * 120)
        say(f"{'Outcome':<72} | {'N':>2} | {'ISR beta':>8} | {'HC1 SE':>6} | {'p':>6} | {'perm p':>6} | {'R^2':>5} | {'ISR|gate':>8} | {'p':>6}")
        say("-" * 120)
        summary["table4b"] = {}
        for col, label in outcomes:
            pairs = []
            for o in occ_rows:
                r = by_occ.get(str(o["occ_soc"]))
                if r is None or r.get(col, "") in ("", "nan"):
                    continue
                pairs.append(([float(o["Institutional_Shield_Gap_pp"]) / 100.0, float(o["Weighted_Overlap_J"]), float(o["job_zone"])], float(r[col]), float(o["statutory_gate"])))
            if len(pairs) < 8:
                continue
            Xo = [x for x, _, _ in pairs]; yo = [y for _, y, _ in pairs]; Xg = [x + [g] for x, _, g in pairs]
            bo, seo, pvo, r2o = ols_hc1(Xo, yo)
            bg, seg, pvg, r2g = ols_hc1(Xg, yo)
            perm_o = permutation_p(Xo, yo, 1, n_perm=2000)
            summary["table4b"][col] = {"n": len(pairs), "isr_beta": round(bo[1], 3), "isr_se": round(seo[1], 3), "isr_p": round(pvo[1], 4), "perm_p": round(perm_o, 4), "r2": round(r2o, 3),
                                       "isr_beta_gate_ctrl": round(bg[1], 3), "isr_p_gate_ctrl": round(pvg[1], 4), "gate_beta": round(bg[4], 3), "gate_p": round(pvg[4], 4)}
            say(f"{label:<72} | {len(pairs):>2} | {bo[1]:+8.3f} | {seo[1]:6.3f} | {_fmt_p(pvo[1]):>6} | {_fmt_p(perm_o):>6} | {r2o:5.3f} | {bg[1]:+8.3f} | {_fmt_p(pvg[1]):>6}")
        say("-" * 120)
        say("  ISR = the laboratory's institutional shield gap (importance-weighted, as a share). Sources per column are in data_inputs/macro_outcomes_extended_lab54.csv.")
        say("  'ISR|gate' adds the typed statutory-gate dummy as a control. Two pairs of occupations share a source unit (nurse practitioners")
        say("  take the registered-nurse unit's values; QA analysts take the software-developer unit's), so distinct outcome values number 17 or fewer.")
        say("=" * 120)

    # ---- Table 5: three waves ----
    say("\n" + "=" * 120)
    say("TABLE 5: DYNAMIC JURISDICTIONAL EVOLUTION — GOVERNANCE EQUILIBRIUM AS THE CHALLENGER'S TOOL ADVANCES (WAVES 1-3)")
    say("=" * 120)
    say(f"{'Ring':<16} | {'Wave':<4} | {'Hybrid%':>7} | {'Poach%':>7} | {'Settle%':>7} | {'Gate%':>6} | {'SQ_Err%':>7} | {'dLnWage':>8}")
    say("-" * 120)
    summary["waves"] = {}
    for ring in RINGS:
        summary["waves"][ring] = {}
        for w in WAVES:
            m = lambda col: _ring_mean(ext_tasks, ring, col)
            vals = {c: m(f"W_{w}_{c}") for c in ("Hybrid_Pct", "Poach_Pct", "Settle_Pct", "Gate_Pct", "SQ_Error_Pct")}
            dln = round(sum(float(r[f"W_{w}_Delta_Ln_WageBill"]) for r in ext_tasks if r["ring"] == ring) / 18.0, 3)
            vals["Delta_Ln_WageBill"] = dln
            summary["waves"][ring][w] = vals
            say(f"{ring:<16} | {w:<4} | {vals['Hybrid_Pct']:6.1f}% | {vals['Poach_Pct']:6.1f}% | {vals['Settle_Pct']:6.1f}% | {vals['Gate_Pct']:5.1f}% | {vals['SQ_Error_Pct']:6.1f}% | {dln:+8.3f}")
    say("=" * 120)
    return summary


def print_and_save_all_results():
    return run_full()


if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(description='Reference configuration: model card, Tables 1-5, compiled_tools/ and data_outputs/').parse_args()   # no options; rejects stray arguments and serves --help
    print_and_save_all_results()
