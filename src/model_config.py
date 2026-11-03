#!/usr/bin/env python3
"""
Central model configuration (model_config.py) — JURI Lab

Every hand-set constant of the laboratory lives here, so that (a) the runner can print a complete model
card, and (b) `run_robustness_suite.py` can vary any of them without touching the modules. Modules read
`CFG` attributes at call time; the suite mutates `CFG` in place and restores it afterwards.

The default values are the reference configuration. `seed_salt=""` reproduces the registered packet universe
bit for bit; any other string draws an alternative universe with the same structure.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Tuple


@dataclass
class ModelConfig:
    # --- packets and archive ---
    n_eval_packets: int = 20            # evaluation packets per task (54 tasks -> 1,080)
    seed_salt: str = ""                 # "" = reference configuration; any other string = alternative packet universe
    archive_size: int = 80              # public resolved-case draws per task (Tiers 0-2 kept)
    silent_gate_sufficient: bool = True # silent-collision V2 drawn above the registry ratio gate (+0.005)
    decoy_note_rate: float = 0.0        # share of Tier 0 packets whose annex carries a benign high-metric note
    random_tier_allocation: bool = False  # draw each packet's tier from the task's tier probabilities instead of a fixed proportional allocation
    # --- packet generator draw ranges (uniform, hash-seeded; the reference configuration never varies them) ---
    gen_v1_range: Tuple[float, float] = (160.0, 910.0)      # primary magnitude V1
    gen_v2_range: Tuple[float, float] = (0.040, 0.210)      # ratio V2
    gen_v3_range: Tuple[float, float] = (20.0, 86.0)        # auxiliary V3
    gen_v4_range: Tuple[float, float] = (90.0, 420.0)       # schedule V4
    gen_noise_range: Tuple[float, float] = (210.0, 790.0)   # table-noise metric (slip and audit-depth rules read it)
    gen_severity_range: Tuple[float, float] = (0.64, 0.91)  # annex severity / precedent coefficient for Tiers 1-3
    gen_routine_metric_range: Tuple[float, float] = (0.12, 0.44)  # routine archive metric (below the registry floor of 0.50)
    gen_decoy_metric_range: Tuple[float, float] = (0.50, 0.91)    # benign note metric when decoy_note_rate > 0 (overlaps severity)
    gen_silent_v2_margin: float = 0.005                     # silent-collision V2 is drawn from gate + margin ...
    gen_silent_v2_max: float = 0.215                        # ... up to this ceiling when silent_gate_sufficient is True
    # --- compiler ---
    coef_decimals: int = 2              # coefficient grid for exact-solve hypotheses (published-schedule prior)
    triple_sample: int = 8              # labelled failures enumerated for k-case exact solves
    min_consensus: int = 3              # archived outcomes a hypothesis must reproduce exactly
    grammar: str = "full"               # "full" = four handler templates with Occam tie-break; "G1" = single template; "no-G1-reparam" (legacy "no-G1") = drops G1 but keeps G4, which nests G1; "no-G1-family" = drops G1 and G4 (true family ablation); "full+G5" adds the interaction template
    selection_validation_fraction: float = 0.0   # 0 = additional-case consensus on the whole archive (registered); >0 = reserve every m-th archive packet from key/label discovery and fitting in both generations; validation-based selection
    archive_includes_tier3: bool = False  # experiment: precedent-tier resolutions are archived publicly (tests the archive boundary)
    # --- novice ---
    novice_primary_drift: float = 42.0
    novice_ratio_drift: float = 0.016
    novice_noise_tol_base: float = 410.0
    novice_noise_tol_slope: float = 330.0   # per unit of overlap_j
    # --- incumbent ---
    solo_slip_base: float = 635.0
    solo_slip_per_jz: float = 22.0
    solo_slip_multiplier: float = 1.035
    audit_depth_base: float = 470.0
    audit_depth_per_z: float = 155.0        # per unit of accountability_z
    override_arithmetic: str = "tool_assisted"  # "manual" applies the solo slip rule to frontstage overrides
    # --- critic (coverage monitor) ---
    u_p_clean: float = 0.03             # no exception handler fired, no unparsed annex field
    u_p_override: float = 0.12          # handbook override rule fired
    u_p_learned: float = 0.19           # archive-induced handler fired
    u_p_unparsed: float = 0.86          # annex carries a structured field no rule consumed
    # --- governance game ---
    w_chall: float = 1.0
    review_fraction_base: float = 0.25
    review_fraction_per_lic: float = 0.08
    liability_scale: float = 5.4
    liability_exp: float = 0.65
    penalty_base: float = 35.0
    penalty_per_lic: float = 15.0
    residual_settle_low: float = 0.035
    residual_settle_high: float = 0.55
    residual_settle_cut: float = 0.30
    residual_incumbent: float = 0.008
    # --- labour block variant: who operates the compiled program ---
    labor_junior_operates_tool: bool = True   # True (registered): one tool-equipped junior-hour per settled or poached case; False: the program replaces junior hours and juniors assist only on gated hard cases
    labor_junior_assist: float = 1.0          # junior-hours per gated case when juniors assist rather than operate (used when the switch is False)
    # --- the learning profession (profession.py; run_dynamics.py --profession learning) ---
    learn_w_gate_junior: float = 1.0      # exposure a junior gains per hard case worked alongside a knowing senior (gated work)
    learn_w_settle_junior: float = 0.5    # ... per settled hard case run through the program whose handler made the rule explicit
    learn_w_review_deep: float = 0.5      # exposure a senior gains per deep review of a program answer that applied an explicit handler
    learn_w_review_shallow: float = 0.1   # ... per shallow review
    learn_w_escalate_senior: float = 1.0  # exposure a senior gains per overt precedent case escalated to a colleague who holds the registry
    learn_kappa_scale: float = 1.0        # multiplies the calibrated knowledge thresholds (1.0 = a pre-compilation apprenticeship transmits both components exactly)
    learn_kappa_basis: str = "pretool"    # "pretool": kappa from the pre-compilation pace (every hard case worked alongside a holder); "period1": from the realised period-1 pace, the labour block's own basis for K
    learn_credit_requires_correct: bool = True  # a settled case teaches the convention only when the program's handler was right (a wrong handler teaches a wrong rule, modelled as no learning)
    learn_escalate_overt: bool = True     # an overt precedent memo is routed to a senior who holds the registry while one exists
    learn_defer_when_unknowing: bool = True  # a reviewer who has not learned the convention defers to a program that applied an explicit handler
    # --- dynamics (run_dynamics.py): disclosure regimes and the strategic incumbent ---
    dyn_archive0: int = 20                # resolved cases in each task's archive at period 1 (the registered 80 is the static design)
    dyn_pub_poach: float = 0.9            # discretion regime: share of poached work the challenger publishes
    dyn_pub_settle: float = 0.5           # discretion regime: share of settled work published
    dyn_pub_gate: float = 0.1             # discretion regime: share of gated work the incumbent publishes
    dyn_gate_open: float = 0.9            # strategic regime: publication of gated work when the incumbent chooses to open the record
    dyn_reputation_bonus: float = 0.02    # strategic regime: revenue credit per published case (the incumbent's reputational return to openness)
    dyn_fee_grid: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.50)   # review fees (fractions of the incumbent wage) the fee-setting incumbent may charge (run_dynamics.py --incumbent-fee)
    review_time_mode: str = "registered"  # physical review time per settled case: "registered" = the registered review fraction, fixed whatever fee is charged; "fee" = legacy confound in which the charged fee also sets review hours (diagnostic only)
    dyn_universe: str = ""                # run universe seed for the dynamics; prepended to every period salt ("" = the registered universe)
    dyn_forecast_mode: str = "independent"  # strategic/fee lookahead: "independent" = a separately seeded forecast sample and forecast publication draws; "perfect_foresight" = the realised next-period packets and realised draws (labelled benchmark)
    dyn_forecast_n: int = 0               # forecast packets per task per decision (0 = the number of evaluation packets)
    archive_label_mode: str = "adjudicated"  # label carried by a published case: "adjudicated" (independent adjudication, exact outcome; upper benchmark), "recorded" (the decision actually delivered), "noisy" (adjudicated with probability archive_adjudication_rate, else recorded), "delayed" (adjudicated outcome enters after archive_adjudication_delay periods)
    archive_adjudication_rate: float = 0.5   # noisy mode: share of published cases that an adjudicator verifies
    archive_adjudication_delay: int = 2      # delayed mode: periods between resolution and the adjudicated record
    client_learning: bool = False         # clients update their belief about the error of incumbent-signed work from overt errors observed one period earlier (False = fixed belief, the reference)
    client_learning_rate: float = 0.5     # weight on the latest observed overt error rate of incumbent-signed work
    client_belief_cap: float = 0.5        # upper bound of the belief
    # --- labour demand and apprenticeship pipeline (labor_market.py) ---
    labor_Q0: float = 1000.0            # task volume per occupation in period 1 (packets)
    labor_capacity: float = 100.0       # packets per worker-period
    labor_apprenticeship_periods: float = 8.0  # apprenticeship length at the pre-compilation learning pace (K = this x period-1 learning per apprentice)
    labor_lambda_settle: float = 0.5    # learning weight of a settled (tool + audit) case relative to a gated case
    labor_delta: float = 0.05           # incumbent retirement rate per period
    labor_demand_growth: float = 0.02   # exogenous growth of task volume per period
    labor_price_elasticity: float = 0.0 # demand response to the unit cost of the task (0 = none)
    labor_scarcity_eta: float = 0.5     # exponent of the scarcity-implied incumbent rate (reported, not fed back)
    # --- Adjacent subordinate occupation (run_bumping.py): a second incumbent with weaker closure ---
    sub_wage_share: float = 0.45          # w_sub = 1 + share * (w_inc - 1) when no named pair exists in the corpus
    sub_licensing_rate: float = 0.05      # licensing coverage of the generic subordinate
    sub_review_fraction: float = 0.25     # subordinate's review share when it supervises the tool
    sub_u_prior: float = 0.10             # client's prior that an unsupervised subordinate errs (it lacks the registry)
    sub_supervised_residual: float = 0.02 # residual error when the subordinate works and the incumbent signs
    # --- Feedback dynamics (run_dynamics.py --feedback): wages and regulation respond to outcomes ---
    fb_wage_backlog_kappa: float = 0.5    # incumbent wage falls with the eligible-apprentice backlog share
    fb_wage_debt_kappa: float = 1.0       # and rises with apprenticeship debt
    fb_wage_floor: float = 0.5            # wage multiplier bounds
    fb_wage_cap: float = 2.0
    reg_relax_below: float = 0.01         # regulator removes the statutory gate when VISIBLE error stays below this ...
    reg_tighten_above: float = 0.05       # ... and imposes it when visible error exceeds this
    reg_window: int = 2                   # consecutive periods required before the regulator acts


CFG = ModelConfig()
_REGISTERED = ModelConfig()


def set_config(**overrides) -> None:
    for k, v in overrides.items():
        if not hasattr(CFG, k):
            raise KeyError(f"unknown config field: {k}")
        setattr(CFG, k, v)


def reset_config() -> None:
    for f in fields(ModelConfig):
        setattr(CFG, f.name, getattr(_REGISTERED, f.name))


def describe_config(cfg: ModelConfig = None) -> List[Tuple[str, object]]:
    cfg = cfg or CFG
    return [(f.name, getattr(cfg, f.name)) for f in fields(ModelConfig)]


def config_diff(cfg: ModelConfig = None) -> Dict[str, object]:
    cfg = cfg or CFG
    return {f.name: getattr(cfg, f.name) for f in fields(ModelConfig) if getattr(cfg, f.name) != getattr(_REGISTERED, f.name)}
