#!/usr/bin/env python3
"""
Endogenous codification dynamics (run_dynamics.py) — JURI Lab

The static waves of Table 5 hold the public archive fixed and vary the compiler generation. Here the
archive is endogenous: what the challenger can learn next period is whatever this period's governance
let out into the public record.

State per task: the public archive A_t of resolved Tier 0-2 cases available to the challenger's compiler.
Each period t = 1..T:
  1. The compiler produces v1 -> v2 -> v3 from A_t (closed-loop repair against A_t).
  2. A fresh universe of evaluation packets is drawn (seed salt "_T{t}", randomized tier allocation).
  3. Novice, tools, coverage critic, incumbent and auditor act; the governance game resolves each packet.
  4. Publication: a resolved Tier 0-2 packet enters A_{t+1} with a probability set by who governed it and
     by the disclosure regime. Tier 3 resolutions are never archived (private precedent).

Disclosure regimes
  mandatory   every resolved Tier 0-2 case is published (p = 1.0 for poached, settled and gated work).
  discretion  challengers publish what they poach (0.9); settled work is published at 0.5; incumbents
              publish what they gate at 0.1.
  strategic   as discretion, but each period each task's incumbent chooses to publish gated work openly
              (0.9) or to withhold it (0.0) by a bounded one-step lookahead on next period's gross incumbent
              revenue (gatekeeping wage + settlement review fee) plus a reputation bonus per published case.
              The lookahead evaluates candidates on an independently seeded forecast sample with independent
              forecast publication draws (dyn_forecast_mode="independent"); knowledge, wages, gates and client
              beliefs are held at their current states. The realised archive then receives the realised draw
              under the chosen policy. dyn_forecast_mode="perfect_foresight" is a labelled benchmark only.

Fee versus time: the fee lever changes the PRICE of a settlement review. The physical review time in the
labour block stays at the registered fraction (review_time_mode="registered"); "excess_review_charge" is the
charge above the registered fee, not an economic rent.

Archive labels: archive_label_mode selects whether published cases carry independently adjudicated outcomes
(the upper benchmark used for the reference results), the recorded decision, a noisy mixture, or a delayed
adjudicated record. Client learning: with client_learning the client's belief about incumbent-signed error
follows overt errors observed one period earlier; the fixed belief is the reference.

Outputs: data_outputs/dynamics_<regime>.csv (period x task) and Table 7 (period x ring) per regime.
"""

import argparse
from dataclasses import replace
import csv
import math
import os
import juri_paths
import shutil
import sys
from typing import Dict, List, Optional, Tuple

from model_config import CFG, set_config, reset_config
from domain_cases import (ONETTaskSpec, PRIMARY_CORPUS, FULL_EXTENDED_CORPUS, MultiDocWorkPacket, _det_float,
                          build_multi_doc_packets_for_task, build_public_sandbox_archive)
from compiler_agent import synthesize_compiled_tool_from_handbook, load_compiled_callable
from negotiated_order_lab import (FULL_KNOWLEDGE, execute_hybrid_tool_plus_critic, execute_incumbent_react_agent, grade_agent_output,
                                  negotiate_workplace_governance, parse_doc_fields, registered_review_fraction,
                                  review_time_share)
from labor_market import OccupationLabor
from profession import OccupationKnowledge, pretool_exposure_per_head

BASE_DIR = juri_paths.ROOT
OUT_DIR = os.path.join(BASE_DIR, "data_outputs")
TOOLS_DIR = os.path.join(BASE_DIR, "dynamics_tools")
RINGS = ("Outer_Periphery", "Middle_Buffer", "Inner_Core")

def publication_table() -> Dict[str, Dict[str, float]]:
    """Publication probabilities by regime, read from the model card (model_config.CFG.dyn_*)."""
    return {
        "mandatory": {"DIRECT_CHALLENGER_POACHING": 1.0, "ACCOUNTABILITY_SETTLEMENT": 1.0, "INCUMBENT_FULL_GATEKEEPING": 1.0},
        "discretion": {"DIRECT_CHALLENGER_POACHING": CFG.dyn_pub_poach, "ACCOUNTABILITY_SETTLEMENT": CFG.dyn_pub_settle,
                       "INCUMBENT_FULL_GATEKEEPING": CFG.dyn_pub_gate},
    }


GATE_CLOSED = 0.0


def compile_v3(spec: ONETTaskSpec, handbook: str, archive: List[MultiDocWorkPacket], tools_dir: str):
    a1 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, tools_dir, "v1")
    a2 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, tools_dir, "v2", sandbox=archive, previous=a1)
    a3 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, handbook, tools_dir, "v3", sandbox=archive, previous=a2)
    return load_compiled_callable(a3), a3


def period_packets(spec: ONETTaskSpec, t: int, n: int, stream: str = "realised") -> List[MultiDocWorkPacket]:
    """Period-t packets. The run's universe (caller seed_salt + dyn_universe) is preserved in the salt and the prior
    configuration is restored afterwards. stream="forecast" draws the incumbent's forecast sample from an
    independent seed stream: it never coincides with the realised packets of period t."""
    prior = (CFG.seed_salt, CFG.random_tier_allocation)
    base = f"{prior[0]}{CFG.dyn_universe}"
    salt = f"{base}_T{t}" if stream == "realised" else f"{base}_FC{t}"
    try:
        set_config(seed_salt=salt, random_tier_allocation=True)
        pk = build_multi_doc_packets_for_task(spec, n, split="eval")
    finally:
        set_config(seed_salt=prior[0], random_tier_allocation=prior[1])
    tag = f"T{t}" if stream == "realised" else f"F{t}"
    for p in pk:
        p.packet_id = f"{p.packet_id}-{tag}"   # unique across periods and streams; the archive accumulates several periods
    return pk


def evaluate_period(spec: ONETTaskSpec, fn, packets: List[MultiDocWorkPacket], prof: Optional[OccupationKnowledge] = None, t: int = 0,
                    review_fraction: Optional[float] = None, belief: Optional[float] = None):
    """Runs agents and governance on one period's packets. Returns per-packet settlements and tool grades. With a
    learning profession, the senior who handles a gated case and the senior who reviews a settled case are drawn from
    the profession's current knowledge shares, and their masks are recorded in the settlement's notes."""
    rows = []
    for pkt in packets:
        d_in, d_sc, d_an = pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex
        out_tool = fn(d_in, d_sc, d_an)
        hyb = execute_hybrid_tool_plus_critic(spec, fn, d_in, d_sc, d_an)
        if prof is not None and prof.initialised:
            # an overt precedent memo is recognised from the annex text (what a professional can see), not from the hidden label
            f = parse_doc_fields(d_in, d_sc, d_an)
            overt = bool(f["TIER3_MEMO"]) or f["base_haircut"] is not None
            solo_mask, esc_solo = prof.draw_mask("seniors", f"{pkt.packet_id}_T{t}_solo", overt)
            review_mask, esc_rev = prof.draw_mask("seniors", f"{pkt.packet_id}_T{t}_review", overt)
        else:
            solo_mask, review_mask, esc_solo, esc_rev = dict(FULL_KNOWLEDGE), dict(FULL_KNOWLEDGE), False, False
        solo = execute_incumbent_react_agent(spec, pkt, "SOLO", knowledge=solo_mask)
        col = execute_incumbent_react_agent(spec, pkt, "COLLAB_AUDIT", challenger_out=out_tool, knowledge=review_mask)
        st = negotiate_workplace_governance(spec, pkt, hyb, col, solo, review_fraction=review_fraction, incumbent_error_belief=belief)
        tool_ok = grade_agent_output(out_tool, pkt)["e2e_ok"]
        st.notes = {"solo_mask": solo_mask, "review_mask": review_mask, "esc_solo": esc_solo, "esc_rev": esc_rev,
                    "solo_hop": solo.get("hop"), "collab_mode": col.get("mode"), "collab_hop": col.get("hop"), "tool_fired": str(out_tool.get("fired", "")),
                    "tool_ok": bool(tool_ok)}
        rows.append((pkt, st, tool_ok))
    return rows


def needed_component(pkt: MultiDocWorkPacket) -> Optional[str]:
    return {2: "convention", 3: "registry"}.get(pkt.complexity_tier)


def credit_exposure(prof: OccupationKnowledge, pkt: MultiDocWorkPacket, st, q: float) -> None:
    """Exposure credits for one resolved case, by the arrangement that handled it (weights on the model card)."""
    comp = needed_component(pkt)
    if comp is None:
        return
    n = st.notes
    if st.selected_governance == "INCUMBENT_FULL_GATEKEEPING":
        if n["solo_mask"][comp]:                      # a senior who held the component worked the case; juniors worked alongside
            if CFG.labor_junior_operates_tool or CFG.labor_junior_assist > 0:
                prof.credit("juniors", comp, CFG.learn_w_gate_junior * q)
            if n["esc_solo"]:                         # the senior who escalated watched a colleague resolve it
                prof.credit("seniors", comp, CFG.learn_w_escalate_senior * q)
    elif st.selected_governance == "ACCOUNTABILITY_SETTLEMENT":
        mode = str(n["collab_mode"])
        if comp == "convention":
            # the program's handler made the convention explicit to whoever read it; by default only a correct handler
            # teaches (a wrong one teaches a wrong rule, modelled as no learning; switch: learn_credit_requires_correct)
            if n["tool_fired"].startswith("H") and (n["tool_ok"] or not CFG.learn_credit_requires_correct):
                if CFG.labor_junior_operates_tool:        # a junior ran the program and saw its handler fire
                    prof.credit("juniors", comp, CFG.learn_w_settle_junior * q)
                prof.credit("seniors", comp, (CFG.learn_w_review_deep if "DEEP" in mode else CFG.learn_w_review_shallow) * q)
        elif n["review_mask"]["registry"] and "OVERRIDE" in mode:   # a reviewer who held the registry caught the case
            if CFG.labor_junior_operates_tool:
                prof.credit("juniors", comp, CFG.learn_w_settle_junior * q)
            if n["esc_rev"]:
                prof.credit("seniors", comp, CFG.learn_w_escalate_senior * q)


def lacked_needed_knowledge(pkt: MultiDocWorkPacket, st) -> bool:
    """True when the professional whose answer the client received did not hold the component the case required."""
    comp = needed_component(pkt)
    if comp is None or st.notes is None:
        return False
    if st.selected_governance == "INCUMBENT_FULL_GATEKEEPING":
        return not st.notes["solo_mask"][comp]
    if st.selected_governance == "ACCOUNTABILITY_SETTLEMENT":
        return not st.notes["review_mask"][comp]
    return False


def incumbent_revenue(spec: ONETTaskSpec, rows, review_fraction: Optional[float] = None) -> float:
    """Incumbent GROSS revenue over a period's cases: the full wage for gated work, the review fee for settled work.
    It is the decision objective of the strategic and fee-setting incumbent; it is not a rent (see excess_review_charge)."""
    w_inc = spec.wage_ratio_inc_chall
    r_inc = registered_review_fraction(spec) if review_fraction is None else review_fraction
    rev = 0.0
    for _, st, _ in rows:
        if st.selected_governance == "INCUMBENT_FULL_GATEKEEPING":
            rev += w_inc
        elif st.selected_governance == "ACCOUNTABILITY_SETTLEMENT":
            rev += r_inc * w_inc
    return rev


def excess_review_charge(spec: ONETTaskSpec, rows, fee: Optional[float]) -> float:
    """Charge for settled work above the registered fee. The registered fee prices the review time at the incumbent
    wage, so this is the fee revenue net of the review labour valued at that wage. It is NOT economic rent: no
    opportunity cost of professional time beyond the wage and no reservation price is modelled."""
    if fee is None:
        return 0.0
    n_settle = sum(1 for _, st, _ in rows if st.selected_governance == "ACCOUNTABILITY_SETTLEMENT")
    return n_settle * (fee - registered_review_fraction(spec)) * spec.wage_ratio_inc_chall


def publish(rows, probs: Dict[str, float], t: int, stream: str = "realised") -> List[MultiDocWorkPacket]:
    """Publication draws. stream="realised" is the archive's own draw (unchanged from the reference configuration);
    stream="forecast" is the incumbent's independent draw used only to evaluate candidate choices."""
    out = []
    suffix = "_pub" if stream == "realised" else f"_fcpub{CFG.dyn_universe}"
    for pkt, st, _ in rows:
        if pkt.complexity_tier >= 3:
            continue
        if _det_float(f"{pkt.packet_id}_T{t}{suffix}", 0.0, 1.0, 6) < probs[st.selected_governance]:
            out.append(pkt)
    return out


def archive_record(pkt: MultiDocWorkPacket, st, t: int) -> Tuple[Optional[MultiDocWorkPacket], int]:
    """The record a published case leaves in the public archive and the period it becomes available.

    adjudicated: an independent adjudicator verifies the outcome; the record carries the exact label (upper benchmark).
    recorded:    the record carries the decision actually delivered to the client, right or wrong.
    noisy:       adjudicated with probability archive_adjudication_rate (deterministic draw), else the recorded decision.
    delayed:     the adjudicated outcome enters only archive_adjudication_delay periods later; nothing before that."""
    mode = CFG.archive_label_mode
    adjudicated = replace(pkt, label_source="adjudicator", label_status="adjudicated")
    out = st.final_output or {}
    recorded = replace(pkt, gt_diagnosis=str(out.get("diagnosis", pkt.gt_diagnosis)), gt_inference=str(out.get("inference", pkt.gt_inference)),
                       gt_treatment=float(out.get("treatment", pkt.gt_treatment)), label_source="recorded_decision",
                       label_status="recorded_correct" if st.final_output_accurate else "recorded_wrong")
    if mode == "adjudicated":
        return adjudicated, t + 1
    if mode == "recorded":
        return recorded, t + 1
    if mode == "noisy":
        hit = _det_float(f"{pkt.packet_id}_T{t}_adj{CFG.dyn_universe}", 0.0, 1.0, 6) < CFG.archive_adjudication_rate
        return (adjudicated if hit else recorded), t + 1
    if mode == "delayed":
        return adjudicated, t + 1 + max(0, int(CFG.archive_adjudication_delay))
    raise ValueError(f"unknown archive_label_mode {mode!r}")


def run_regime(regime: str, corpus: List[ONETTaskSpec], T: int, n_eval: int, a0: int, overrides: Optional[Dict[str, object]] = None,
               feedback: str = "none", profession: str = "oracle", gates: str = "registered", incumbent_fee: bool = False):
    """feedback: 'none' (registered), 'wage' (apprentice backlog and debt move the incumbent wage), 'regulator' (the statutory
    gate follows VISIBLE error), or 'both'. With feedback on, each period's task specification is a copy of the registered one
    with the occupation's current wage multiplier and gate state substituted; the registered specs are never mutated.
    profession: 'oracle' (the registered incumbent knows the whole rulebook) or 'learning' (profession.py: knowledge is
    reproduced through practice and the incumbent's accuracy becomes a state variable). Returns (records, labor_records,
    knowledge_records)."""
    tools_dir = os.path.join(TOOLS_DIR, f"{regime}_{os.getpid()}")
    os.makedirs(tools_dir, exist_ok=True)
    reset_config()
    if overrides:
        set_config(**overrides)   # e.g. labour constants for sensitivity runs; applied after the registered defaults
    if gates != "registered":
        # counterfactual statutes: every core and buffer task gated ("all") or no task gated ("none"); the registered
        # specs are never mutated. Used to isolate what the statute does from what the typed occupation inputs do.
        corpus = [replace(s, statutory_gate=(1 if (gates == "all" and s.ring != "Outer_Periphery") else 0)) for s in corpus]
    archives: Dict[str, List[MultiDocWorkPacket]] = {s.task_id: build_public_sandbox_archive(s, a0) for s in corpus}
    labor: Dict[str, OccupationLabor] = {}
    for s in corpus:
        labor.setdefault(s.occ_title, OccupationLabor(s.occ_title, s.wage_ratio_inc_chall,
                                                       CFG.review_fraction_base + CFG.review_fraction_per_lic * s.cps_licensing_rate))
    labor_records: List[Dict[str, object]] = []
    knowledge_records: List[Dict[str, object]] = []
    prof: Dict[str, OccupationKnowledge] = {s.occ_title: OccupationKnowledge(s.occ_title) for s in corpus} if profession == "learning" else {}
    occ_tasks: Dict[str, List[ONETTaskSpec]] = {}
    for s in corpus:
        occ_tasks.setdefault(s.occ_title, []).append(s)

    def packet_volume(spec: ONETTaskSpec, t: int) -> float:
        """Case volume per evaluation packet, on the labour block's demand scale (labor_market.OccupationLabor.step)."""
        tasks = occ_tasks[spec.occ_title]
        wsum = sum(x.task_importance_wt for x in tasks) or 1.0
        scale = CFG.labor_Q0 * len(tasks) * (1.0 + CFG.labor_demand_growth) ** (t - 1)
        return scale * spec.task_importance_wt / wsum / n_eval

    pending: Dict[str, List[Tuple[int, MultiDocWorkPacket]]] = {s.task_id: [] for s in corpus}   # archive records not yet available
    # the client's belief about the error of incumbent-signed work; fixed at the model-card residual unless client_learning
    belief: Dict[str, float] = {s.occ_title: CFG.residual_incumbent for s in corpus}
    if CFG.dyn_forecast_mode not in ("independent", "perfect_foresight"):
        raise ValueError(f"unknown dyn_forecast_mode {CFG.dyn_forecast_mode!r}")
    n_fc = CFG.dyn_forecast_n or n_eval
    period_rows: Dict[str, list] = {}
    period_tiers: Dict[str, Dict[int, float]] = {}
    # the fee-setting incumbent: the review fraction charged on settled work in each task, chosen one period ahead
    # from the model card's grid by the same one-step revenue lookahead as the publication decision; period 1 charges
    # the registered fee
    fee_state: Dict[str, float] = {s.task_id: registered_review_fraction(s) for s in corpus}
    handbooks = {s.task_id: build_multi_doc_packets_for_task(s, 1, split="eval")[0].doc_handbook for s in corpus}
    records: List[Dict[str, object]] = []
    fb_wage = feedback in ("wage", "both")
    fb_reg = feedback in ("regulator", "both")
    wage_mult: Dict[str, float] = {s.occ_title: 1.0 for s in corpus}
    # the statute names TASKS (an occupation's core may be gated while its periphery is not); the regulator decides per
    # occupation and either lifts every gate the occupation holds or restores the registered pattern
    gate_state: Dict[str, int] = {s.task_id: s.statutory_gate for s in corpus}
    registered_gate: Dict[str, int] = {s.task_id: s.statutory_gate for s in corpus}
    occ_gated: Dict[str, bool] = {s.occ_title: any(x.statutory_gate for x in corpus if x.occ_title == s.occ_title) for s in corpus}
    visible_hist: Dict[str, List[float]] = {s.occ_title: [] for s in corpus}
    occ_err: Dict[str, Dict[str, int]] = {}
    for t in range(1, T + 1):
        occ_err = {s.occ_title: {"n": 0, "visible": 0, "silent": 0, "signed": 0, "signed_overt_err": 0} for s in corpus}
        for spec0 in corpus:
            spec = replace(spec0, wage_ratio_inc_chall=round(spec0.wage_ratio_inc_chall * wage_mult[spec0.occ_title], 4),
                           statutory_gate=gate_state[spec0.task_id]) if feedback != "none" else spec0
            # archive records whose availability date has arrived (only the delayed mode defers them)
            due = [q for (avail, q) in pending[spec.task_id] if avail <= t]
            pending[spec.task_id] = [(avail, q) for (avail, q) in pending[spec.task_id] if avail > t]
            archives[spec.task_id] = archives[spec.task_id] + due
            A = archives[spec.task_id]
            b_now = belief[spec.occ_title] if CFG.client_learning else None
            fn, art = compile_v3(spec, handbooks[spec.task_id], A, tools_dir)
            pk = period_packets(spec, t, n_eval)
            pr = prof.get(spec.occ_title)
            fee_now = fee_state[spec.task_id] if incumbent_fee else None
            rows = evaluate_period(spec, fn, pk, prof=pr, t=t, review_fraction=fee_now, belief=b_now)
            period_rows[spec.task_id] = rows
            period_tiers[spec.task_id] = {k: sum(1 for p in pk if p.complexity_tier == k) / len(pk) for k in (0, 1, 2, 3)}
            if pr is not None:
                q = packet_volume(spec, t)
                for p, st, _ in rows:
                    credit_exposure(pr, p, st, q)
            e = occ_err[spec.occ_title]
            e["n"] += len(rows)
            e["visible"] += sum(1 for p, st, _ in rows if not st.final_output_accurate and not p.is_silent_collision)
            e["silent"] += sum(1 for p, st, _ in rows if not st.final_output_accurate and p.is_silent_collision)
            signed = [(p, st) for p, st, _ in rows if st.selected_governance in ("INCUMBENT_FULL_GATEKEEPING", "ACCOUNTABILITY_SETTLEMENT")]
            e["signed"] += len(signed)
            e["signed_overt_err"] += sum(1 for p, st in signed if not st.final_output_accurate and not p.is_silent_collision)
            n = len(rows)
            PUBLICATION = publication_table()
            gov = {g: sum(1 for _, st, _ in rows if st.selected_governance == g) / n for g in PUBLICATION["mandatory"]}
            hard_gate = sum(1 for p, st, _ in rows if p.complexity_tier >= 2 and st.selected_governance == "INCUMBENT_FULL_GATEKEEPING") / n
            hard_settle = sum(1 for p, st, _ in rows if p.complexity_tier >= 2 and st.selected_governance == "ACCOUNTABILITY_SETTLEMENT") / n
            t2 = [ok for p, _, ok in rows if p.complexity_tier == 2]
            avg_cost = sum(st.labor_cost_units for _, st, _ in rows) / n
            closed = 0
            fee_next = fee_state[spec.task_id]
            st_of = {p.packet_id: st for p, st, _ in rows}
            if regime == "strategic" or incumbent_fee:
                # bounded one-step lookahead on next period's incumbent revenue: over open vs closed publication of gated
                # work (strategic regime) and, with the fee lever on, over the fee grid. Under the default "independent"
                # forecast mode the incumbent evaluates candidates on an independently seeded forecast sample with
                # independent forecast publication draws; it never sees the realised next-period packets or the realised
                # publication draw. Knowledge shares, wages, gates and client beliefs are held at their current state.
                # "perfect_foresight" is the labelled benchmark that evaluates on the realised packets and draws.
                pf = CFG.dyn_forecast_mode == "perfect_foresight"
                nxt = period_packets(spec, t + 1, n_eval) if pf else period_packets(spec, t + 1, n_fc, stream="forecast")
                pub_options = ((("open", CFG.dyn_gate_open), ("closed", GATE_CLOSED)) if regime == "strategic"
                               else ((regime, None),))
                fee_options = tuple(CFG.dyn_fee_grid) if incumbent_fee else (None,)
                best = None
                for label, p_gate in pub_options:
                    if regime == "strategic":
                        probs = dict(PUBLICATION["discretion"], INCUMBENT_FULL_GATEKEEPING=p_gate)
                    else:
                        probs = PUBLICATION[regime]
                    pub_fc = publish(rows, probs, t, stream="realised" if pf else "forecast")
                    # the forecast archive uses the record the incumbent expects under the archive label mode
                    fc_records = [r for r, avail in (archive_record(q, st_of[q.packet_id], t) for q in pub_fc) if avail <= t + 1]
                    fn2, _ = compile_v3(spec, handbooks[spec.task_id], A + fc_records, os.path.join(tools_dir, "lookahead"))
                    for fee in fee_options:
                        rev = incumbent_revenue(spec, evaluate_period(spec, fn2, nxt, prof=pr, t=t + 1, review_fraction=fee, belief=b_now), review_fraction=fee)
                        if regime == "strategic":
                            rev += CFG.dyn_reputation_bonus * len(pub_fc)
                        if best is None or rev > best[0] + 1e-9:
                            best = (rev, label, probs, fee)
                closed = 1 if best[1] == "closed" else 0
                new_cases = publish(rows, best[2], t)   # the realised publication draw under the chosen policy
                if incumbent_fee:
                    fee_next = best[3]
            else:
                new_cases = publish(rows, PUBLICATION[regime], t)
            for q in new_cases:
                rec_q, avail = archive_record(q, st_of[q.packet_id], t)
                pending[spec.task_id].append((avail, rec_q))
            fee_state[spec.task_id] = fee_next
            records.append({
                "regime": regime, "period": t, "task_id": spec.task_id, "occ_title": spec.occ_title, "ring": spec.ring,
                "archive_size": len(A), "archive_tier2": sum(1 for q in A if q.complexity_tier == 2),
                "v3_has_handler": int(bool(art.ir.get("learned"))), "v3_has_split": int(bool(art.ir.get("learned") and art.ir["learned"][0].get("split"))),
                "n_packets": n, "tier2_parity": round(100.0 * sum(t2) / len(t2), 1) if t2 else float("nan"),
                "tool_e2e": round(100.0 * sum(ok for _, _, ok in rows) / n, 1),
                "poach": round(100.0 * gov["DIRECT_CHALLENGER_POACHING"], 1), "settle": round(100.0 * gov["ACCOUNTABILITY_SETTLEMENT"], 1),
                "gate": round(100.0 * gov["INCUMBENT_FULL_GATEKEEPING"], 1),
                "sq_error": round(100.0 * sum(1 for _, st, _ in rows if not st.final_output_accurate) / n, 1),
                "dln_wagebill": round(math.log(avg_cost / spec.wage_ratio_inc_chall), 4),
                "incumbent_revenue": round(incumbent_revenue(spec, rows, review_fraction=fee_now), 2), "closed": closed, "published": len(new_cases),
                "fee": round(fee_now if fee_now is not None else registered_review_fraction(spec), 4),
                "fee_registered": round(registered_review_fraction(spec), 4),
                "review_time_share": round(review_time_share(spec, fee_now), 4),
                "excess_review_charge": round(excess_review_charge(spec, rows, fee_now), 4),
                "client_belief": round(belief[spec.occ_title], 5),
                "archive_wrong_labels": sum(1 for q in A if q.label_status == "recorded_wrong"),
                "archive_label_mode": CFG.archive_label_mode, "forecast_mode": CFG.dyn_forecast_mode,
                "run_universe": f"{CFG.seed_salt}{CFG.dyn_universe}", "forecast_universe": f"{CFG.seed_salt}{CFG.dyn_universe}_FC",
                "weight": spec.task_importance_wt, "unit_cost": avg_cost, "hard_gate": hard_gate, "hard_settle": hard_settle,
                "wage_mult": round(wage_mult[spec.occ_title], 4), "gate_state": gate_state[spec.task_id],
                "visible_error": round(100.0 * sum(1 for p, st, _ in rows if not st.final_output_accurate and not p.is_silent_collision) / n, 1),
                "silent_error": round(100.0 * sum(1 for p, st, _ in rows if not st.final_output_accurate and p.is_silent_collision) / n, 1),
                # the learning profession: knowledge shares at the start of the period and what the masks did to this period's work
                "s_conv": round(pr.share("seniors", "convention"), 4) if pr is not None else 1.0,
                "s_reg": round(pr.share("seniors", "registry"), 4) if pr is not None else 1.0,
                "j_conv": round(pr.share("juniors", "convention"), 4) if pr is not None else float("nan"),
                "j_reg": round(pr.share("juniors", "registry"), 4) if pr is not None else float("nan"),
                "hard_n": sum(1 for p in pk if p.complexity_tier >= 2),
                "unknowing_handled": sum(1 for p, st, _ in rows if lacked_needed_knowledge(p, st)),
                "unknowing_errors": sum(1 for p, st, _ in rows if lacked_needed_knowledge(p, st) and not st.final_output_accurate),
                "unknowing_conv": sum(1 for p, st, _ in rows if lacked_needed_knowledge(p, st) and p.complexity_tier == 2),
                "unknowing_conv_err": sum(1 for p, st, _ in rows if lacked_needed_knowledge(p, st) and p.complexity_tier == 2 and not st.final_output_accurate),
                "unknowing_reg": sum(1 for p, st, _ in rows if lacked_needed_knowledge(p, st) and p.complexity_tier == 3),
                "unknowing_reg_err": sum(1 for p, st, _ in rows if lacked_needed_knowledge(p, st) and p.complexity_tier == 3 and not st.final_output_accurate),
                "hard_conv_n": sum(1 for p in pk if p.complexity_tier == 2),
                "hard_reg_n": sum(1 for p in pk if p.complexity_tier == 3),
                "hard_settle_err": sum(1 for p, st, _ in rows if p.complexity_tier >= 2 and st.selected_governance == "ACCOUNTABILITY_SETTLEMENT" and not st.final_output_accurate),
                "hard_gate_err": sum(1 for p, st, _ in rows if p.complexity_tier >= 2 and st.selected_governance == "INCUMBENT_FULL_GATEKEEPING" and not st.final_output_accurate),
                "escalations": sum(1 for p, st, _ in rows if st.notes and (st.notes["esc_solo"] and st.selected_governance == "INCUMBENT_FULL_GATEKEEPING" or st.notes["esc_rev"] and st.selected_governance == "ACCOUNTABILITY_SETTLEMENT")),
                "deferrals": sum(1 for p, st, _ in rows if st.notes and st.selected_governance == "ACCOUNTABILITY_SETTLEMENT" and "DEFER" in str(st.notes["collab_mode"])),
            })
        # the client's belief about incumbent-signed error: updated at the end of the period from the overt errors of
        # incumbent-signed work (gated or settled) and used from the next period on, i.e. observed with a one-period delay.
        # Silent-collision errors are not observed. With client_learning off the belief stays at the fixed reference.
        if CFG.client_learning:
            for occ, e in occ_err.items():
                if e["signed"] > 0:
                    obs = e["signed_overt_err"] / e["signed"]
                    nb = (1.0 - CFG.client_learning_rate) * belief[occ] + CFG.client_learning_rate * obs
                    belief[occ] = min(CFG.client_belief_cap, max(0.0, nb))
        # occupation-level labour accounting for this period (tasks pooled: apprentices learn on the periphery, replace the core)
        for occ, st in labor.items():
            tks = [{"weight": r["weight"], "poach": r["poach"] / 100.0, "settle": r["settle"] / 100.0, "gate": r["gate"] / 100.0,
                    "hard_gate": r["hard_gate"], "hard_settle": r["hard_settle"], "unit_cost": r["unit_cost"], "review_time": r["review_time_share"]}
                   for r in records if r["period"] == t and r["occ_title"] == occ]
            if tks:
                rec = st.step(t, tks)
                spec_o = next(s for s in corpus if s.occ_title == occ)
                labor_records.append({"regime": regime, "period": t, "occ_title": occ,
                                      "statutory_gate": int(any(x.statutory_gate for x in corpus if x.occ_title == occ)),   # registered status: fixed grouping
                                      "gate_now": int(occ_gated[occ]),
                                      "closure_tier": spec_o.closure_tier, "wage_mult": round(wage_mult[occ], 4),
                                      **{k: round(v, 4) for k, v in rec.items() if k != "period"}})
                if occ in prof:
                    pr = prof[occ]
                    if not pr.initialised:
                        # the labour block's state at the start of period 1: a mature pipeline with one cohort per apprenticeship period
                        n_c = max(1, int(round(CFG.labor_apprenticeship_periods)))
                        cohorts0 = [[rec["A_stock"] / n_c, st.K * k / n_c] for k in range(n_c)]
                        vols = {x.task_id: packet_volume(x, 1) * n_eval for x in occ_tasks[occ]}
                        pre = pretool_exposure_per_head(vols, {x.task_id: period_tiers[x.task_id] for x in occ_tasks[occ]}, rec["A_stock"])
                        p1 = {comp: pr.credits["juniors"][comp] / max(rec["A_stock"], 1e-9) for comp in ("convention", "registry")}
                        pr.initialise(rec["I_stock"], cohorts0, st.K, pre, t, period1_exposure=p1)
                    krec = pr.end_period(t, rec)
                    knowledge_records.append({"regime": regime, "occ_title": occ, "statutory_gate": int(any(x.statutory_gate for x in corpus if x.occ_title == occ)),
                                              **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in krec.items()}})
                if fb_wage:
                    # scarcity raises the incumbent rate, a backlog of eligible apprentices lowers it, unfilled vacancies raise it
                    m = ((max(rec["I_demand"], 1e-9) / max(rec["I_stock"], 1e-9)) ** CFG.labor_scarcity_eta
                         * (1.0 - CFG.fb_wage_backlog_kappa * rec["eligible_backlog"]) * (1.0 + CFG.fb_wage_debt_kappa * rec["apprenticeship_debt"]))
                    wage_mult[occ] = min(CFG.fb_wage_cap, max(CFG.fb_wage_floor, m))
        if fb_reg:
            # the regulator sees only visible error: it relaxes a gate after `reg_window` quiet periods and imposes one after
            # `reg_window` loud ones. Silent error never enters its record.
            for occ, e in occ_err.items():
                visible_hist[occ].append(e["visible"] / max(1, e["n"]))
                h = visible_hist[occ][-CFG.reg_window:]
                if len(h) == CFG.reg_window:
                    tasks_of = [x.task_id for x in corpus if x.occ_title == occ]
                    if occ_gated[occ] and all(v < CFG.reg_relax_below for v in h):
                        occ_gated[occ] = False
                        for tid in tasks_of:
                            gate_state[tid] = 0
                        visible_hist[occ] = []
                    elif not occ_gated[occ] and all(v > CFG.reg_tighten_above for v in h):
                        occ_gated[occ] = True
                        for tid in tasks_of:
                            gate_state[tid] = max(registered_gate[tid], 1 if x_core(tid, corpus) else 0)
                        visible_hist[occ] = []
        print(f"  [{regime}] period {t}: mean archive {sum(r['archive_size'] for r in records if r['period']==t and r['regime']==regime)/len(corpus):.1f}", flush=True)
    shutil.rmtree(tools_dir, ignore_errors=True)
    return records, labor_records, knowledge_records


def print_table7(records: List[Dict[str, object]], regime: str, T: int) -> None:
    print("\n" + "=" * 128)
    print(f"TABLE 7: ENDOGENOUS CODIFICATION DYNAMICS — regime = {regime} (period x ring means; archive = public cases at start of period)")
    print("=" * 128)
    print(f"{'Ring':<16} | {'t':>2} | {'Archive':>7} | {'T2 arch':>7} | {'T2 parity':>9} | {'Tool E2E':>8} | {'Poach%':>6} | {'Settle%':>7} | {'Gate%':>5} | {'SQ_Err':>6} | {'dLnWage':>8} | {'Closed%':>7}")
    print("-" * 128)
    for ring in RINGS:
        for t in range(1, T + 1):
            rr = [r for r in records if r["ring"] == ring and r["period"] == t]
            m = lambda k: sum(float(r[k]) for r in rr if r[k] == r[k]) / max(1, sum(1 for r in rr if r[k] == r[k]))
            print(f"{ring:<16} | {t:>2} | {m('archive_size'):7.1f} | {m('archive_tier2'):7.1f} | {m('tier2_parity'):8.1f}% | {m('tool_e2e'):7.1f}% | "
                  f"{m('poach'):5.1f}% | {m('settle'):6.1f}% | {m('gate'):4.1f}% | {m('sq_error'):5.1f}% | {m('dln_wagebill'):+8.3f} | {100*m('closed'):6.1f}%")
    print("=" * 128)


def print_table9(labor_records: List[Dict[str, object]], regime: str, T: int) -> None:
    print("\n" + "=" * 134)
    print(f"TABLE 9: LABOUR DEMAND AND THE APPRENTICESHIP PIPELINE — regime = {regime} (occupation-level; group means; indices relative to period 1)")
    print("=" * 146)
    print(f"{'Group':<20} | {'t':>2} | {'Inc dem':>7} | {'Inc stock':>9} | {'Juniors':>7} | {'Learn idx':>9} | {'Apprent.':>8} | {'Promo/Vac':>9} | {'Debt%':>6} | {'Backlog%':>8} | {'Util':>5} | {'Emp in':>6} | {'UnitCost':>8} | {'w_inc sc':>8}")
    print("-" * 146)
    for label, pred in (("Licensed (gate=1)", lambda r: r["statutory_gate"] == 1), ("Unlicensed (gate=0)", lambda r: r["statutory_gate"] == 0)):
        base = {}
        for t in range(1, T + 1):
            rr = [r for r in labor_records if r["period"] == t and pred(r)]
            if not rr:
                continue
            m = lambda k: sum(float(r[k]) for r in rr if float(r[k]) != float("inf")) / max(1, sum(1 for r in rr if float(r[k]) != float("inf")))
            if t == 1:
                base = {k: m(k) for k in ("I_demand", "I_stock", "A_stock", "emp_inside")}
            idx = lambda k: 100.0 * m(k) / max(base[k], 1e-9)
            pv = 100.0 * sum(float(r["promotions"]) for r in rr) / max(1e-9, sum(float(r["vacancies"]) for r in rr))
            print(f"{label:<20} | {t:>2} | {idx('I_demand'):7.1f} | {idx('I_stock'):9.1f} | {idx('A_stock'):7.1f} | {m('learning_index'):9.2f} | {m('periods_to_eligibility'):8.1f} | "
                  f"{pv:8.1f}% | {100*m('apprenticeship_debt'):5.1f}% | {100*m('eligible_backlog'):7.1f}% | {m('utilisation'):5.2f} | {idx('emp_inside'):6.1f} | {m('unit_cost_rel'):8.3f} | {m('w_inc_scarcity'):8.3f}")
    print("=" * 146)
    print("Learn idx = hands-on hard-case learning per apprentice relative to period 1; Apprent. = periods to eligibility at the current pace;")
    print("Promo/Vac = promotions as a share of vacancies; Debt% = vacancies left unfilled for want of eligible apprentices, relative to incumbents demanded;")
    print("Backlog% = eligible apprentices waiting for a vacancy, as a share of apprentices.")


def print_table14(records: List[Dict[str, object]], regime: str, T: int, gate_of: Dict[str, int]) -> None:
    print("\n" + "=" * 150)
    print(f"TABLE 14: THE KNOWLEDGE RACE — regime = {regime} (licensed and unlicensed occupations; period means; shares of the senior and junior pools that hold the component)")
    print("=" * 150)
    print(f"{'Group':<20} | {'t':>2} | {'T2 parity':>9} | {'Sen conv':>8} | {'Jun conv':>8} | {'Sen reg':>7} | {'Jun reg':>7} | {'Unknowing%':>10} | {'Unk err%':>8} | {'Settle hard err%':>16} | {'SQ err%':>7} | {'Escal.':>6} | {'Defer':>5}")
    print("-" * 150)
    for label, pred in (("Licensed (gate=1)", lambda r: gate_of[r["occ_title"]] == 1), ("Unlicensed (gate=0)", lambda r: gate_of[r["occ_title"]] == 0)):
        for t in range(1, T + 1):
            rr = [r for r in records if r["period"] == t and pred(r)]
            if not rr:
                continue
            m = lambda k: sum(float(r[k]) for r in rr if r[k] == r[k]) / max(1, sum(1 for r in rr if r[k] == r[k]))
            hard = sum(int(r["hard_n"]) for r in rr) or 1
            settled_hard = sum(int(r["hard_settle_err"]) for r in rr)
            print(f"{label:<20} | {t:>2} | {m('tier2_parity'):8.1f}% | {100*m('s_conv'):7.1f}% | {100*m('j_conv'):7.1f}% | {100*m('s_reg'):6.1f}% | {100*m('j_reg'):6.1f}% | "
                  f"{100*sum(int(r['unknowing_handled']) for r in rr)/hard:9.1f}% | {100*sum(int(r['unknowing_errors']) for r in rr)/hard:7.1f}% | "
                  f"{100*settled_hard/hard:15.1f}% | {m('sq_error'):6.1f}% | {sum(int(r['escalations']) for r in rr):6d} | {sum(int(r['deferrals']) for r in rr):5d}")
    print("-" * 150)
    print("T2 parity = the compiled program's accuracy on convention cases; Sen/Jun conv = share of seniors / juniors who hold the convention; Sen/Jun reg = share who hold the registry;")
    print("Unknowing% = hard cases (Tiers 2 and 3) whose signing professional lacked the component the case required, as a share of hard cases; Unk err% = the errors among them;")
    print("Settle hard err% = errors on settled hard cases as a share of all hard cases; Escal. = overt precedent memos routed to a registry holder; Defer = reviewers deferring to a handler they did not know.")


def print_table17(records: List[Dict[str, object]], regime: str, T: int) -> None:
    print("\n" + "=" * 128)
    print(f"TABLE 17: THE FEE-SETTING INCUMBENT — regime = {regime} (period x ring means; fee = review fraction charged on settled work; Excess = fee minus the registered fee (an excess review charge, not economic rent; review hours stay at the registered time))")
    print("=" * 128)
    print(f"{'Ring':<16} | {'t':>2} | {'Fee':>6} | {'Excess':>6} | {'at max':>6} | {'at min':>6} | {'Poach%':>6} | {'Settle%':>7} | {'Gate%':>5} | {'Inc rev':>7} | {'T2 parity':>9} | {'Closed%':>7}")
    print("-" * 128)
    for ring in RINGS:
        for t in range(1, T + 1):
            rr = [r for r in records if r["ring"] == ring and r["period"] == t]
            if not rr:
                continue
            m = lambda k: sum(float(r[k]) for r in rr if r[k] == r[k]) / max(1, sum(1 for r in rr if r[k] == r[k]))
            hi = 100.0 * sum(1 for r in rr if abs(float(r["fee"]) - max(CFG.dyn_fee_grid)) < 1e-9) / len(rr)
            lo = 100.0 * sum(1 for r in rr if abs(float(r["fee"]) - min(CFG.dyn_fee_grid)) < 1e-9) / len(rr)
            print(f"{ring:<16} | {t:>2} | {m('fee'):6.3f} | {m('fee') - m('fee_registered'):+6.3f} | {hi:5.0f}% | {lo:5.0f}% | {m('poach'):5.1f}% | {m('settle'):6.1f}% | {m('gate'):4.1f}% | {m('incumbent_revenue'):7.2f} | {m('tier2_parity'):8.1f}% | {100*m('closed'):6.1f}%")
    print("=" * 128)
    print("at max / at min = share of tasks charging the highest / lowest fee on the grid; Inc rev = incumbent revenue per period over the ring's packets (challenger-wage units).")


def x_core(task_id: str, corpus: List[ONETTaskSpec]) -> bool:
    return any(x.task_id == task_id and x.ring == "Inner_Core" for x in corpus)


def print_table11(records: List[Dict[str, object]], regime: str, T: int, feedback: str) -> None:
    print("\n" + "=" * 128)
    print(f"TABLE 11: FEEDBACK DYNAMICS — regime = {regime}, feedback = {feedback} (period x ring means)")
    print("=" * 128)
    print(f"{'Ring':<16} | {'t':>2} | {'Gate on':>7} | {'w_inc x':>7} | {'Poach%':>6} | {'Settle%':>7} | {'Gate%':>5} | {'Vis err':>7} | {'Sil err':>7} | {'SQ_Err':>6} | {'T2 parity':>9} | {'dLnWage':>8}")
    print("-" * 128)
    for ring in RINGS:
        for t in range(1, T + 1):
            rr = [r for r in records if r["ring"] == ring and r["period"] == t]
            if not rr:
                continue
            m = lambda k: sum(float(r[k]) for r in rr if r[k] == r[k]) / max(1, sum(1 for r in rr if r[k] == r[k]))
            print(f"{ring:<16} | {t:>2} | {100*m('gate_state'):6.0f}% | {m('wage_mult'):7.3f} | {m('poach'):5.1f}% | {m('settle'):6.1f}% | {m('gate'):4.1f}% | "
                  f"{m('visible_error'):6.1f}% | {m('silent_error'):6.1f}% | {m('sq_error'):5.1f}% | {m('tier2_parity'):8.1f}% | {m('dln_wagebill'):+8.3f}")
    print("-" * 128)
    print("Gate on = share of tasks whose occupation currently has a statutory sign-off requirement; w_inc x = incumbent wage multiplier;")
    print("Vis err = errors on overt packets (what a regulator can observe); Sil err = errors on silent collisions (what it cannot).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="mandatory,discretion,strategic")
    ap.add_argument("--periods", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--archive0", type=int, default=None, help="archive size at period 1 (default: model card dyn_archive0 = 20)")
    ap.add_argument("--corpus", choices=["primary", "extended"], default="extended")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a model_config field for every regime, e.g. --set labor_delta=0.10")
    ap.add_argument("--feedback", choices=["none", "wage", "regulator", "both"], default="none",
                    help="let apprentice scarcity move the incumbent wage and/or let the regulator set the statutory gate from visible error")
    ap.add_argument("--profession", choices=["oracle", "learning"], default="oracle",
                    help="oracle: the registered incumbent knows the whole rulebook; learning: knowledge is reproduced through practice (profession.py)")
    ap.add_argument("--gates", choices=["registered", "all", "none"], default="registered",
                    help="statutory sign-off as registered, imposed on every core and buffer task, or removed everywhere (counterfactual statutes)")
    ap.add_argument("--incumbent-fee", action="store_true",
                    help="the incumbent chooses its review fee each period from the model card's grid by revenue lookahead (jointly with publication in the strategic regime)")
    ap.add_argument("--forecast-mode", choices=["independent", "perfect_foresight"], default=None,
                    help="strategic/fee lookahead sample: independent forecast draws (default) or the labelled perfect-foresight benchmark")
    ap.add_argument("--archive-labels", choices=["adjudicated", "recorded", "noisy", "delayed"], default=None,
                    help="label carried by published archive cases (default: independently adjudicated outcome)")
    ap.add_argument("--client-learning", action="store_true", help="clients learn incumbent-signed error from overt errors observed one period earlier")
    ap.add_argument("--universe", default=None, help="run universe seed (dyn_universe); distinct values give independent replications")
    ap.add_argument("--out-dir", default=None, help="output directory (default data_outputs/)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(CFG, k)
        if isinstance(cur, bool):
            overrides[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, (tuple, list)):
            import ast
            overrides[k] = type(cur)(ast.literal_eval(v))
        else:
            overrides[k] = type(cur)(v)
    if args.forecast_mode:
        overrides["dyn_forecast_mode"] = args.forecast_mode
    if args.archive_labels:
        overrides["archive_label_mode"] = args.archive_labels
    if args.client_learning:
        overrides["client_learning"] = True
    if args.universe is not None:
        overrides["dyn_universe"] = args.universe
    global OUT_DIR
    if args.out_dir:
        OUT_DIR = os.path.abspath(args.out_dir)
    corpus = PRIMARY_CORPUS if args.corpus == "primary" else FULL_EXTENDED_CORPUS
    os.makedirs(OUT_DIR, exist_ok=True)
    for regime in [r.strip() for r in args.regimes.split(",") if r.strip()]:
        a0 = args.archive0 if args.archive0 is not None else int((overrides or {}).get("dyn_archive0", CFG.dyn_archive0))
        recs, labs, knows = run_regime(regime, corpus, args.periods, args.n_eval, a0, overrides=overrides, feedback=args.feedback,
                                       profession=args.profession, gates=args.gates, incumbent_fee=args.incumbent_fee)
        tag = args.tag or ((f"_fb_{args.feedback}" if args.feedback != "none" else "") + ("_learning" if args.profession == "learning" else "")
                           + (f"_gates_{args.gates}" if args.gates != "registered" else "") + ("_fee" if args.incumbent_fee else "")
                           + (f"_labels_{overrides['archive_label_mode']}" if overrides.get("archive_label_mode", "adjudicated") != "adjudicated" else "")
                           + ("_clientlearn" if overrides.get("client_learning") else "")
                           + ("_perfectforesight" if overrides.get("dyn_forecast_mode") == "perfect_foresight" else "")
                           + (f"_u{overrides['dyn_universe']}" if overrides.get("dyn_universe") else ""))
        gate_of = {s.occ_title: int(any(x.statutory_gate for x in corpus if x.occ_title == s.occ_title)) for s in corpus}
        with open(os.path.join(OUT_DIR, f"dynamics_{regime}{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        with open(os.path.join(OUT_DIR, f"dynamics_labor_{regime}{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(labs[0].keys()))
            w.writeheader()
            w.writerows(labs)
        if knows:
            with open(os.path.join(OUT_DIR, f"dynamics_knowledge_{regime}{tag}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(knows[0].keys()))
                w.writeheader()
                w.writerows(knows)
        print_table7(recs, regime, args.periods)
        print_table9(labs, regime, args.periods)
        if args.feedback != "none":
            print_table11(recs, regime, args.periods, args.feedback)
        if args.profession == "learning":
            print_table14(recs, regime, args.periods, gate_of)
        if args.incumbent_fee:
            print_table17(recs, regime, args.periods)
    reset_config()
    try:
        os.rmdir(TOOLS_DIR)   # each run removes its own per-process directory; remove the parent only when it is empty (parallel runs)
    except OSError:
        pass


if __name__ == "__main__":
    main()
