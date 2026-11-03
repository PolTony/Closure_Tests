#!/usr/bin/env python3
"""
Module 3: Multi-Agent Workplace Turf, Bounded Rationality & Governance Game (negotiated_order_lab.py)
JURI Lab — executed architecture

Every agent reads the four workplace documents; none reads the packet's hidden parameters or its answer. The
challenger side (compiled tools, the critic, and in the AI arm the model agents) never imports the rule
interpreter or the registry. The rule-based novice and incumbent are stylised models of people who KNOW part
or all of the rulebook: the novice applies the handbook's routine rules with drifted cutoffs; the incumbent
classifies the case from the documents and then applies the full rulebook (handbook, convention, registry)
through the same interpreter that grades every agent. The incumbent is therefore an oracle up to three
assumptions that are declared, printed on the model card, and swept: tier classification from documents
(silent collisions require the registry's quantitative gate and a deep audit), arithmetic slips above a noise
threshold, and audit depth under time pressure. The live model run (run_llm_arm.py) anchors these assumptions
with an incumbent that reads the same documents and registry without any interpreter.

  A_Base   Unassisted Novice: applies the public handbook's routine rules with drifted cutoffs, cannot
           process annex exceptions, and mis-integrates the schedule when table noise is high.
  A_Tool   v1/v2/v3: the compiled scripts (compiler_agent.py).
  A_Hybrid Tool + coverage critic: the critic reads two structural signals the script reports about its
           own run, (i) whether an annex field in key=value form was left unconsumed by every rule, and
           (ii) which rule fired (handbook override, archive-induced handler, or routine), and maps them
           to an epistemic uncertainty u_p. It has no keyword list: a silent collision whose annex uses
           the routine format is, to the critic, routine.
  A_Inc_Solo: 3-hop incumbent. Hop 1 parses intake/schedule; Hop 2 classifies the annex, including the
           registry's quantitative silent-collision gate (archive metric above the registry floor AND
           V2 above the registry ratio gate); Hop 3 resolves Tier 3 through the registry and evaluates
           the full rulebook. Solo arithmetic slips occur above a noise threshold.
  A_Collab (Accountability Settlement): the challenger's script computes backstage; the incumbent audits
           frontstage with its own derivation and accepts the tool when they agree. Above the audit-depth
           noise limit the frontstage audit skips the quantitative gate, so a silent collision whose
           routine-looking output matches the tool is rubber-stamped.

All behavioural constants come from model_config.CFG and are printed by the runner as a model card.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from model_config import CFG, describe_config
from domain_cases import (
    ONETTaskSpec,
    MultiDocWorkPacket,
    PRECEDENT_REGISTRY,
    RULEBOOKS,
    eval_expr,
    evaluate_rulebook,
    parse_doc_fields,
)


def describe_behavioral_parameters() -> List[Tuple[str, object]]:
    return describe_config()


# ---------------------------------------------------------------------------
# A_Base: unassisted novice
# ---------------------------------------------------------------------------
def execute_unassisted_novice(spec: ONETTaskSpec, doc_intake: str, doc_sched: str, doc_annex: str) -> Dict[str, object]:
    f = parse_doc_fields(doc_intake, doc_sched, doc_annex)
    v1, v2, v3, v4, nse = f["V1"], f["V2"], f["V3"], f["V4"], f["NOISE"]
    drift_p = spec.primary_cutoff + CFG.novice_primary_drift
    drift_r = spec.ratio_cutoff + CFG.novice_ratio_drift
    diag_band = "HIGH_EXPOSURE" if v1 >= drift_p else "MODERATE_EXPOSURE"
    thresh_band = "THRESHOLD_BREACH" if v2 >= drift_r else "WITHIN_TOLERANCE"
    diagnosis = f"{diag_band}__{thresh_band}"
    if f["EXPLICIT_TIER1_OVERRIDE"]:
        return {"diagnosis": diagnosis, "inference": f"{spec.domain_family}_NOVICE_GUESS", "treatment": round(v1 * 0.60 - v3, 2)}
    noise_tol = CFG.novice_noise_tol_base + CFG.novice_noise_tol_slope * spec.overlap_j
    near_boundary = (spec.primary_cutoff <= v1 < drift_p) or (spec.ratio_cutoff <= v2 < drift_r)
    if nse < noise_tol and not near_boundary:
        rb = RULEBOOKS[spec.domain_family]
        env = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "PRIMARY_CUTOFF": drift_p, "RATIO_CUTOFF": drift_r}
        for name, ex in rb.lets:
            env[name] = eval_expr(ex, env)
        label, ex = rb.breach_rule if v2 >= drift_r else rb.base_rule
        return {"diagnosis": diagnosis, "inference": label, "treatment": round(eval_expr(ex, env), 2)}
    return {"diagnosis": diagnosis, "inference": "NOVICE_SCHEDULE_MISALIGNMENT", "treatment": round(v1 + 0.20 * v3 + 0.05 * nse, 2)}


# ---------------------------------------------------------------------------
# A_Hybrid: compiled script + coverage critic
# ---------------------------------------------------------------------------
def execute_hybrid_tool_plus_critic(spec: ONETTaskSpec, compiled_fn: Callable[[str, str, str], Dict[str, object]],
                                    doc_intake: str, doc_sched: str, doc_annex: str) -> Dict[str, object]:
    out = dict(compiled_fn(doc_intake, doc_sched, doc_annex))
    fired = str(out.get("fired", ""))
    if out.get("unhandled_fields"):
        u, status = CFG.u_p_unparsed, "UNPARSED_ANNEX_FIELD_ESCALATED"
    elif fired in ("", "NONE", "ERROR", "UNKNOWN") or str(out.get("diagnosis", "")).endswith("ERROR") or str(out.get("diagnosis", "")) in ("TIMEOUT", "PARSE_ERROR"):
        # a script that crashed, timed out or could not parse its inputs is escalated, never trusted as clean
        u, status = CFG.u_p_unparsed, "TOOL_FAILURE_ESCALATED"
    elif fired.startswith("H"):
        u, status = CFG.u_p_learned, "LEARNED_HANDLER_FIRED"
    elif fired == "R1":
        u, status = CFG.u_p_override, "HANDBOOK_OVERRIDE_FIRED"
    else:
        u, status = CFG.u_p_clean, "CLEAN_ROUTINE"
    out["epistemic_uncertainty"] = u
    out["critic_status"] = status
    return out


execute_hybrid_v3_plus_critic = execute_hybrid_tool_plus_critic  # backward-compatible name


# ---------------------------------------------------------------------------
# A_Inc: three-hop incumbent (solo) and frontstage auditor (collab)
# ---------------------------------------------------------------------------
FULL_KNOWLEDGE = {"convention": True, "registry": True}   # the reference configuration's incumbent knows the whole rulebook


def _incumbent_three_hops(spec: ONETTaskSpec, pkt: MultiDocWorkPacket, deep_audit: bool,
                          knowledge: Optional[Dict[str, bool]] = None) -> Tuple[str, str, float, int, str, float]:
    """`knowledge` is the professional's knowledge mask (profession.py): a professional who has not learned the
    domain's convention reads a convention case as the handbook does; one who does not hold the registry cannot
    resolve an overt precedent memo and cannot apply the silent-collision gate. With the default (full) mask this is
    the reference configuration's incumbent."""
    k = knowledge or FULL_KNOWLEDGE
    f = parse_doc_fields(pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
    v1, v2, v3, v4, noise = f["V1"], f["V2"], f["V3"], f["V4"], f["NOISE"]
    reg = PRECEDENT_REGISTRY[spec.domain_family]
    metric = f["ARCHIVE_METRIC"]
    if f["EXPLICIT_TIER1_OVERRIDE"]:
        tier, sev, hop = 1, f["ANNEX_SEVERITY"], "HOP2_EXPLICIT_OVERRIDE"
    elif f["severity_index"] is not None:
        if k.get("convention", True):
            tier, sev, hop = 2, f["severity_index"], "HOP2_PROTOCOL_EXCEPTION"
        else:
            tier, sev, hop = 0, 0.0, "HOP2_CONVENTION_UNKNOWN"
    elif f["TIER3_MEMO"] or f["base_haircut"] is not None:
        if k.get("registry", True):
            tier, sev, hop = 3, f["base_haircut"], "HOP3_REGISTRY_OVERT"
        else:
            tier, sev, hop = 0, 0.0, "HOP3_REGISTRY_UNKNOWN"
    elif (deep_audit and k.get("registry", True) and metric is not None and metric >= reg["silent_metric_floor"] and v2 > reg["silent_ratio_gate"]):
        tier, sev, hop = 3, metric, "HOP3_REGISTRY_SILENT_GATE"
    else:
        tier, sev, hop = 0, 0.0, "HOP2_ROUTINE"
    d, i, t = evaluate_rulebook(spec, v1, v2, v3, v4, sev, tier)
    return d, i, t, tier, hop, noise


def _slips(spec: ONETTaskSpec, noise: float) -> bool:
    return noise > (CFG.solo_slip_base + CFG.solo_slip_per_jz * spec.job_zone)


def execute_incumbent_react_agent(spec: ONETTaskSpec, pkt: MultiDocWorkPacket, mode: str = "SOLO",
                                  challenger_out: Optional[Dict[str, object]] = None,
                                  knowledge: Optional[Dict[str, bool]] = None) -> Dict[str, object]:
    if mode == "SOLO":
        d, i, t, tier, hop, noise = _incumbent_three_hops(spec, pkt, deep_audit=True, knowledge=knowledge)
        if _slips(spec, noise):
            return {"diagnosis": d, "inference": i, "treatment": round(t * CFG.solo_slip_multiplier, 2), "mode": "SOLO_ARITHMETIC_SLIP", "hop": hop}
        return {"diagnosis": d, "inference": i, "treatment": t, "mode": "SOLO_EXACT", "hop": hop}

    # COLLAB_AUDIT: backstage tool + frontstage incumbent audit. Audit depth is set by time pressure.
    audit_noise_limit = CFG.audit_depth_base + CFG.audit_depth_per_z * spec.accountability_z
    noise = parse_doc_fields(pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)["NOISE"]
    deep = noise <= audit_noise_limit
    d, i, t, tier, hop, _ = _incumbent_three_hops(spec, pkt, deep_audit=deep, knowledge=knowledge)
    depth = "DEEP" if deep else "SHALLOW"
    if (challenger_out is not None and challenger_out.get("diagnosis") == d and challenger_out.get("inference") == i
            and abs(float(challenger_out.get("treatment", 0.0)) - t) <= 0.01):
        return {"diagnosis": d, "inference": i, "treatment": float(challenger_out["treatment"]),
                "mode": f"COLLAB_ACCEPT_{depth}_AUDIT", "hop": hop}
    if (hop == "HOP2_CONVENTION_UNKNOWN" and challenger_out is not None and str(challenger_out.get("fired", "")).startswith("H")
            and CFG.learn_defer_when_unknowing):
        # a reviewer who has not learned the convention sees the program apply an explicit exception handler it does not
        # recognise; the declared behaviour is deference to the program (the alternative, overriding with the handbook
        # answer, is a model-card switch)
        return {"diagnosis": str(challenger_out.get("diagnosis")), "inference": str(challenger_out.get("inference")),
                "treatment": float(challenger_out.get("treatment", 0.0)), "mode": f"COLLAB_DEFER_UNKNOWING_{depth}", "hop": hop}
    if CFG.override_arithmetic == "manual" and _slips(spec, noise):
        return {"diagnosis": d, "inference": i, "treatment": round(t * CFG.solo_slip_multiplier, 2),
                "mode": f"COLLAB_OVERRIDE_{depth}_AUDIT_SLIP", "hop": hop}
    return {"diagnosis": d, "inference": i, "treatment": t, "mode": f"COLLAB_OVERRIDE_{depth}_AUDIT", "hop": hop}


def grade_agent_output(pred: Dict[str, object], pkt: MultiDocWorkPacket) -> Dict[str, int]:
    d_ok = int(pred["diagnosis"] == pkt.gt_diagnosis)
    i_ok = int(pred["inference"] == pkt.gt_inference)
    t_ok = int(abs(float(pred["treatment"]) - pkt.gt_treatment) <= 0.01)
    return {"d_ok": d_ok, "i_ok": i_ok, "t_ok": t_ok, "e2e_ok": int(d_ok == 1 and i_ok == 1 and t_ok == 1)}


# ---------------------------------------------------------------------------
# Governance game (status quo, CF1 deregulated sign-off, CF2 unregulated)
# ---------------------------------------------------------------------------
@dataclass
class NegotiatedSettlementResult:
    packet_id: str
    task_id: str
    occ_title: str
    closure_tier: str
    ring: str
    complexity_tier: int
    is_collision: bool
    is_silent: bool
    epistemic_uncertainty: float
    statutory_gate: int
    accountability_z: float
    selected_governance: str
    final_output_accurate: int
    labor_cost_units: float
    incumbent_hours_saved_pct: float
    cf1_governance: str
    cf1_accurate: int
    cf1_hours_saved_pct: float
    cf2_governance: str
    cf2_accurate: int
    notes: Dict[str, object] = None   # optional per-case annotations set by callers (run_dynamics: knowledge masks and modes)
    final_output: Dict[str, object] = None   # the answer the client received under the status quo (the recorded decision)


def _solve_governance_equilibrium(u_p, w_chall, w_settle, w_inc, statutory_penalty, error_liability_L,
                                  challenger_out, collab_out, solo_inc_out, pkt, incumbent_error_belief: Optional[float] = None,
                                  return_output: bool = False):
    """incumbent_error_belief: the client's belief that incumbent-signed work errs (default: the fixed model-card
    residual). A belief above the reference raises the expected error of both arrangements the incumbent signs."""
    b = CFG.residual_incumbent if incumbent_error_belief is None else float(incumbent_error_belief)
    excess = max(0.0, b - CFG.residual_incumbent)
    cost_poach = w_chall + statutory_penalty + (u_p * error_liability_L)
    residual_err_settle = (CFG.residual_settle_low * u_p) if u_p <= CFG.residual_settle_cut else (CFG.residual_settle_high * u_p)
    cost_settle = w_settle + ((residual_err_settle + excess) * error_liability_L)
    cost_inc = w_inc + (b * error_liability_L)
    if cost_poach <= cost_settle and cost_poach <= cost_inc:
        gov, final_pred, labor_c, hrs_saved = "DIRECT_CHALLENGER_POACHING", challenger_out, w_chall, 100.0
    elif cost_settle <= cost_inc:
        gov = "ACCOUNTABILITY_SETTLEMENT"   # Abbott's subordination settlement: tool backstage, licensee signs
        final_pred = collab_out               # the settlement always returns the audited output
        labor_c = w_settle
        hrs_saved = round(max(0.0, 100.0 * (w_inc - w_settle) / w_inc), 1)
    else:
        gov, final_pred, labor_c, hrs_saved = "INCUMBENT_FULL_GATEKEEPING", solo_inc_out, w_inc, 0.0
    if return_output:
        return gov, grade_agent_output(final_pred, pkt)["e2e_ok"], labor_c, hrs_saved, final_pred
    return gov, grade_agent_output(final_pred, pkt)["e2e_ok"], labor_c, hrs_saved


def registered_review_fraction(spec) -> float:
    """The registered review fraction: the share of an incumbent work unit that reviewing a settled case takes. At the
    registered fee the client pays exactly this time at the incumbent wage."""
    return CFG.review_fraction_base + CFG.review_fraction_per_lic * spec.cps_licensing_rate


def review_time_share(spec, fee: Optional[float] = None) -> float:
    """Physical incumbent review time per settled case. Fixed at the registered fraction whatever fee is charged,
    unless the legacy diagnostic `review_time_mode="fee"` is selected, in which the charged fee also sets the hours."""
    if CFG.review_time_mode == "fee" and fee is not None:
        return fee
    if CFG.review_time_mode not in ("registered", "fee"):
        raise ValueError(f"unknown review_time_mode {CFG.review_time_mode!r}")
    return registered_review_fraction(spec)


def negotiate_workplace_governance(spec, pkt, hybrid_out, collab_out, solo_inc_out, review_fraction: Optional[float] = None,
                                   incumbent_error_belief: Optional[float] = None) -> NegotiatedSettlementResult:
    """`review_fraction` overrides the registered settlement FEE charged to the client (the fee-setting incumbent of
    run_dynamics.py). It is a price: it does not change the physical review time, which the labour block reads from
    `review_time_share`. `incumbent_error_belief` replaces the client's fixed belief about incumbent-signed error."""
    u_p = float(hybrid_out.get("epistemic_uncertainty", 0.10))
    w_chall = CFG.w_chall
    w_inc = spec.wage_ratio_inc_chall
    r_inc = registered_review_fraction(spec) if review_fraction is None else review_fraction
    w_settle = round(w_chall + r_inc * w_inc, 3)
    error_liability_L = CFG.liability_scale * math.exp(CFG.liability_exp * spec.accountability_z)
    statutory_penalty = (CFG.penalty_base + CFG.penalty_per_lic * spec.cps_licensing_rate) if spec.statutory_gate == 1 else 0.0
    b = incumbent_error_belief
    gov_sq, acc_sq, cost_sq, saved_sq, out_sq = _solve_governance_equilibrium(u_p, w_chall, w_settle, w_inc, statutory_penalty, error_liability_L, hybrid_out, collab_out, solo_inc_out, pkt, b, True)
    gov_cf1, acc_cf1, _, saved_cf1 = _solve_governance_equilibrium(u_p, w_chall, w_settle, w_inc, 0.0, error_liability_L, hybrid_out, collab_out, solo_inc_out, pkt, b)
    gov_cf2, acc_cf2, _, _ = _solve_governance_equilibrium(u_p, w_chall, w_settle, w_inc, 0.0, 0.0, hybrid_out, collab_out, solo_inc_out, pkt, b)
    return NegotiatedSettlementResult(
        packet_id=pkt.packet_id, task_id=pkt.task_id, occ_title=spec.occ_title, closure_tier=spec.closure_tier, ring=spec.ring,
        complexity_tier=pkt.complexity_tier, is_collision=pkt.is_collision_case, is_silent=pkt.is_silent_collision,
        epistemic_uncertainty=u_p, statutory_gate=spec.statutory_gate, accountability_z=spec.accountability_z,
        selected_governance=gov_sq, final_output_accurate=acc_sq, labor_cost_units=cost_sq, incumbent_hours_saved_pct=saved_sq,
        cf1_governance=gov_cf1, cf1_accurate=acc_cf1, cf1_hours_saved_pct=saved_cf1, cf2_governance=gov_cf2, cf2_accurate=acc_cf2, final_output=dict(out_sq),
    )
