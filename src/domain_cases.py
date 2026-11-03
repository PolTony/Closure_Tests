#!/usr/bin/env python3
"""
Module 1: Non-Isomorphic Multi-Domain Case Generator, Rule DSL & Exact Solver (domain_cases.py)
JURI Lab — executed architecture

Information structure (who can read what):
  * PUBLIC HANDBOOK (`doc_handbook`): the document field map, diagnostic cutoffs, the domain's routine
    rules (LET definitions, R1 explicit-override rule, R2 threshold rule, R3 otherwise rule) and the
    validation sub-clauses, written in a small rule language. Read by the Challenger's compiler and by
    the Novice.
  * PUBLIC RESOLVED-CASE ARCHIVE (sandbox): labelled past packets of Tiers 0-2 only. The Tier 2
    "semi-structured protocol exception" (severity_index in the annex) is NOT codified in the handbook;
    its piecewise rule can only be induced from archived outcomes. Read by the compiler's repair loop.
  * PRIVATE `PRECEDENT_REGISTRY` and the Tier 3 rules: available to the Incumbent only. Tier 3 outcomes
    never appear in the public archive.
  * The exact solver interprets the same rule strings that are printed in the handbook, plus the Tier 2
    convention and the private Tier 3 rules. The rule strings are the single source of truth: nothing in
    the compiler, the novice or the incumbent copies numeric constants from the solver.

Six non-isomorphic domain engines (LEGAL, CPA, CLINICAL, FINANCE, ENGINEERING, SWE_ADMIN) and four
epistemic complexity tiers per task (0 routine, 1 explicit annex alert, 2 piecewise collision, 3 precedent
collision including silent collisions) follow the reference configuration. Macro outcomes for the Table 4 bridge are
loaded from `macro_outcomes.csv` (with a provenance column), not typed into the task specs.
"""

import ast as _ast
import csv
import functools
import hashlib
import math
import os
import juri_paths
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from model_config import CFG


# Domain Precedent Knowledge Graph (PRIVATE: queried in Hop 3 by the Incumbent agent only)
PRECEDENT_REGISTRY: Dict[str, Dict[str, float]] = {
    "LEGAL": {
        "subrogation_mult": 1.38,
        "fiduciary_cap_haircut": 0.42,
        "silent_ratio_gate": 0.135,
        "silent_metric_floor": 0.50,
        "statutory_offset": 1.28,
    },
    "CPA": {
        "vie_impairment_mult": 1.44,
        "valuation_allowance_rate": 0.38,
        "silent_ratio_gate": 0.130,
        "silent_metric_floor": 0.50,
        "covenant_clawback": 1.32,
    },
    "CLINICAL": {
        "cyp3a4_accumulation_factor": 1.52,
        "renal_washout_weight": 0.46,
        "silent_ratio_gate": 0.125,
        "silent_metric_floor": 0.50,
        "toxicity_penalty": 1.36,
    },
    "FINANCE": {
        "csa_convexity_mult": 1.31,
        "springing_sweep_rate": 0.34,
        "silent_ratio_gate": 0.140,
        "silent_metric_floor": 0.50,
        "liquidity_haircut": 1.22,
    },
    "ENGINEERING": {
        "liquefaction_shear_amp": 1.56,
        "buckling_knockdown": 0.44,
        "silent_ratio_gate": 0.120,
        "silent_metric_floor": 0.50,
        "seismic_safety_margin": 1.42,
    },
    "SWE_ADMIN": {
        "quorum_retry_amp": 1.26,
        "partition_rollback_wt": 0.31,
        "silent_ratio_gate": 0.145,
        "silent_metric_floor": 0.50,
        "sla_penalty_mult": 1.16,
    },
}


@dataclass
class ONETTaskSpec:
    occ_soc: str
    occ_title: str
    job_zone: int
    closure_tier: str
    task_id: str
    ring: str                  # Inner_Core, Middle_Buffer, Outer_Periphery
    onet_task: str
    dwa_bundle: str
    overlap_j: float           # Concentric task overlap J_{i,k} in [0, 1]
    task_importance_wt: float  # O*NET Task Importance share within occupation (sums to 1.0 across the 3 rings)
    statutory_gate: int        # 1 if state statute / board rule mandates licensed human sign-off
    cps_licensing_rate: float  # Empirical CPS licensing coverage in [0, 1]
    wage_ratio_inc_chall: float  # Empirical BLS OEWS wage ratio (Incumbent / Challenger)
    accountability_z: float    # Pre-shock O*NET Consequence of Error & Responsibility z-score
    n_subclauses: int          # Number of task-specific validation sub-rules in the public handbook
    collision_density: float   # Overall proportion of packets with cross-document exceptions (kappa >= 1)
    tier2_piecewise_share: float  # Share of collisions that are 1st-order piecewise-compilable (kappa == 2)
    tier3_deep_share: float    # Share of collisions that are 2nd-order precedent couplings (kappa == 3)
    silent_collision_prob: float  # Share of Tier 3 collisions that lack surface alarm keywords ("Silent Collisions")
    domain_family: str         # "LEGAL", "CPA", "CLINICAL", "FINANCE", "ENGINEERING", "SWE_ADMIN"
    primary_cutoff: float      # Task-specific diagnostic magnitude cutoff
    ratio_cutoff: float        # Task-specific secondary threshold cutoff
    # Macro outcomes are attached at import from macro_outcomes.csv (see attach_macro_outcomes); never literals.
    macro_auto_delta_19_26: float = 0.0   # 2019->2026 O*NET Degree of Automation change (z)
    macro_telemetry_gap_z: float = 0.0    # Workplace AI adoption residual vs technical exposure (z)
    macro_semantic_shift_z: float = 0.0   # 2019->2026 O*NET task-centroid shift on the AI axis (z)
    macro_source: str = "UNSET"           # Provenance string from macro_outcomes.csv


PRIMARY_CORPUS: List[ONETTaskSpec] = [
    # 1. LAWYERS (23-1011) - Tier 3 Statutory Bar License
    ONETTaskSpec(
        "23-1011", "Lawyers", 5, "Tier3_Statutory_License", "LAW-C1", "Inner_Core",
        "Advise clients on strategic litigation exposure when cross-jurisdictional fiduciary duties conflict",
        "Analyze conflict-of-laws statutes; Evaluate fiduciary carve-outs; Formulate litigation reserve strategy",
        0.18, 0.46, 1, 0.89, 3.65, 1.48, 7, 0.85, 0.18, 0.76, 0.30, "LEGAL", 520.0, 0.095
    ),
    ONETTaskSpec(
        "23-1011", "Lawyers", 5, "Tier3_Statutory_License", "LAW-B2", "Middle_Buffer",
        "Draft and negotiate commercial contract indemnification and liability cap clauses",
        "Draft commercial indemnity provisions; Reconcile liability caps with insurance riders",
        0.44, 0.34, 1, 0.89, 3.65, 1.18, 5, 0.50, 0.50, 0.20, 0.15, "LEGAL", 480.0, 0.105
    ),
    ONETTaskSpec(
        "23-1011", "Lawyers", 5, "Tier3_Statutory_License", "LAW-P3", "Outer_Periphery",
        "Review standard NDAs and commercial lease agreements against boilerplate compliance checklists",
        "Verify confidentiality term lengths; Check governing venue clauses against standard playbook",
        0.72, 0.20, 0, 0.89, 3.65, 0.45, 2, 0.20, 0.25, 0.00, 0.00, "LEGAL", 450.0, 0.110
    ),

    # 2. PARALEGALS & LEGAL ASSISTANTS (23-2011) - Tier 1 Subordinate
    ONETTaskSpec(
        "23-2011", "Paralegals", 4, "Tier1_Subordinate", "PAR-C1", "Inner_Core",
        "Coordinate multi-party discovery production and privilege log exception flagging",
        "Audit joint-defense privilege logs; Identify third-party waiver exceptions in document productions",
        0.29, 0.36, 0, 0.06, 1.85, 0.38, 4, 0.65, 0.38, 0.46, 0.15, "LEGAL", 500.0, 0.100
    ),
    ONETTaskSpec(
        "23-2011", "Paralegals", 4, "Tier1_Subordinate", "PAR-B2", "Middle_Buffer",
        "Draft routine legal pleadings, motions, and corporate entity filings from templates",
        "Populate motion templates; Calculate statutory filing deadlines and fee schedules",
        0.52, 0.38, 0, 0.06, 1.85, 0.12, 3, 0.35, 0.57, 0.14, 0.00, "LEGAL", 460.0, 0.105
    ),
    ONETTaskSpec(
        "23-2011", "Paralegals", 4, "Tier1_Subordinate", "PAR-P3", "Outer_Periphery",
        "Organize, index, and cross-reference case exhibits and statutory citations",
        "Index Bates-stamped exhibit tables; Verify standard citation formatting",
        0.78, 0.26, 0, 0.06, 1.85, -0.22, 1, 0.10, 0.00, 0.00, 0.00, "LEGAL", 440.0, 0.115
    ),

    # 3. ACCOUNTANTS & AUDITORS (13-2011) - Tier 3 CPA Attestation
    ONETTaskSpec(
        "13-2011", "Accountants_Auditors", 4, "Tier3_Task_Certification", "CPA-C1", "Inner_Core",
        "Evaluate going-concern uncertainty and materiality of coupled contingent liabilities",
        "Assess ASC 450 loss contingencies; Evaluate debt cross-default triggers on going-concern opinion",
        0.21, 0.44, 1, 0.62, 3.10, 1.32, 6, 0.80, 0.25, 0.69, 0.25, "CPA", 510.0, 0.098
    ),
    ONETTaskSpec(
        "13-2011", "Accountants_Auditors", 4, "Tier3_Task_Certification", "CPA-B2", "Middle_Buffer",
        "Reconcile multi-element revenue recognition and deferred tax schedules under ASC 606",
        "Allocate transaction price across performance obligations; Compute deferred tax valuation allowance",
        0.47, 0.35, 1, 0.62, 3.10, 0.96, 4, 0.45, 0.56, 0.22, 0.10, "CPA", 490.0, 0.102
    ),
    ONETTaskSpec(
        "13-2011", "Accountants_Auditors", 4, "Tier3_Task_Certification", "CPA-P3", "Outer_Periphery",
        "Verify ledger journal entries against supporting invoices and categorize operating expenses",
        "Match voucher totals to invoice ledgers; Classify routine OPEX vs CAPEX line items",
        0.74, 0.21, 0, 0.62, 3.10, 0.28, 2, 0.15, 0.33, 0.00, 0.00, "CPA", 460.0, 0.108
    ),

    # 4. FINANCIAL & INVESTMENT ANALYSTS (13-2051) - Tier 2 General Degree
    ONETTaskSpec(
        "13-2051", "Financial_Analysts", 4, "Tier2_General_Degree", "FIN-C1", "Inner_Core",
        "Structure liquidity stress-tests when debt covenants and macro triggers interact non-linearly",
        "Model springing maturity covenants; Evaluate interest-coverage cliffs under liquidity stress",
        0.26, 0.38, 0, 0.18, 2.75, 0.54, 5, 0.70, 0.36, 0.57, 0.20, "FINANCE", 515.0, 0.096
    ),
    ONETTaskSpec(
        "13-2051", "Financial_Analysts", 4, "Tier2_General_Degree", "FIN-B2", "Middle_Buffer",
        "Build discounted cash flow (DCF) valuation models and comparable multiple tables from 10-Ks",
        "Compute WACC-adjusted enterprise value; Normalize EBITDA multiples from financial filings",
        0.54, 0.37, 0, 0.18, 2.75, 0.22, 3, 0.40, 0.62, 0.12, 0.00, "FINANCE", 485.0, 0.104
    ),
    ONETTaskSpec(
        "13-2051", "Financial_Analysts", 4, "Tier2_General_Degree", "FIN-P3", "Outer_Periphery",
        "Compile quarterly earnings variance summaries and standardized financial ratio dashboards",
        "Calculate YoY margin variance; Aggregate quarterly consensus beat/miss tables",
        0.79, 0.25, 0, 0.18, 2.75, -0.12, 2, 0.10, 0.00, 0.00, 0.00, "FINANCE", 455.0, 0.112
    ),

    # 5. NURSE PRACTITIONERS (29-1171) - Tier 3 Contested Scope License
    ONETTaskSpec(
        "29-1171", "Nurse_Practitioners", 5, "Tier3_Contested_License", "NP-C1", "Inner_Core",
        "Manage polypharmacy regimens for multimorbid patients with conflicting organ contraindications",
        "Reconcile cardiology-nephrology drug interactions; Adjust dosing under acute renal decline",
        0.24, 0.45, 1, 0.94, 3.25, 1.66, 7, 0.85, 0.18, 0.76, 0.35, "CLINICAL", 505.0, 0.094
    ),
    ONETTaskSpec(
        "29-1171", "Nurse_Practitioners", 5, "Tier3_Contested_License", "NP-B2", "Middle_Buffer",
        "Diagnose acute episodic presentations and titrate chronic disease medications per protocol",
        "Evaluate hypertension/glycemic panels; Titrate first-line maintenance therapy per guideline",
        0.49, 0.35, 1, 0.94, 3.25, 1.28, 5, 0.50, 0.50, 0.30, 0.15, "CLINICAL", 475.0, 0.100
    ),
    ONETTaskSpec(
        "29-1171", "Nurse_Practitioners", 5, "Tier3_Contested_License", "NP-P3", "Outer_Periphery",
        "Perform clinical intake triage, document history, and order standard screening panels",
        "Record baseline vital signs; Order routine preventive metabolic screening labs",
        0.71, 0.20, 0, 0.94, 3.25, 0.58, 2, 0.20, 0.25, 0.00, 0.00, "CLINICAL", 445.0, 0.108
    ),

    # 6. SOFTWARE QA ANALYSTS & TESTERS (15-1253) - Tier 1 Market Open
    ONETTaskSpec(
        "15-1253", "Software_QA_Analysts", 4, "Tier1_Market_Open", "SQA-C1", "Inner_Core",
        "Diagnose non-deterministic race conditions and asynchronous state bugs across microservices",
        "Trace deadlocks across distributed event queues; Isolate idempotency failures under retry storms",
        0.28, 0.34, 0, 0.02, 2.05, 0.18, 4, 0.65, 0.46, 0.46, 0.15, "SWE_ADMIN", 510.0, 0.100
    ),
    ONETTaskSpec(
        "15-1253", "Software_QA_Analysts", 4, "Tier1_Market_Open", "SQA-B2", "Middle_Buffer",
        "Write automated API schema validation scripts and integration test suites for CI/CD pipelines",
        "Verify REST payload latency and status codes; Validate JSON schema contract compliance",
        0.56, 0.38, 0, 0.02, 2.05, -0.08, 3, 0.30, 0.67, 0.00, 0.00, "SWE_ADMIN", 470.0, 0.106
    ),
    ONETTaskSpec(
        "15-1253", "Software_QA_Analysts", 4, "Tier1_Market_Open", "SQA-P3", "Outer_Periphery",
        "Execute regression test checklists, log UI defect steps, and verify standard patch fixes",
        "Compare build pass rates against regression threshold; Log smoke-test status codes",
        0.82, 0.28, 0, 0.02, 2.05, -0.42, 1, 0.10, 0.00, 0.00, 0.00, "SWE_ADMIN", 435.0, 0.114
    ),
]


EXTENDED_ADDITIONAL_SPECS: List[ONETTaskSpec] = [
    # 7. General & Operations Managers (11-1021) - Tier 2 General Degree
    ONETTaskSpec("11-1021", "Operations_Managers", 5, "Tier2_General_Degree", "OPS-C1", "Inner_Core",
                 "Resolve cross-unit supply-chain allocation conflicts under binding working-capital covenants",
                 "Balance plant capacity utilization; Arbitrate penalty-clause SLAs across business units",
                 0.22, 0.40, 0, 0.09, 2.95, 0.74, 5, 0.75, 0.33, 0.60, 0.20, "FINANCE", 525.0, 0.096),
    ONETTaskSpec("11-1021", "Operations_Managers", 5, "Tier2_General_Degree", "OPS-B2", "Middle_Buffer",
                 "Forecast quarterly departmental resource budgets and operational KPI variance schedules",
                 "Model headcount-to-throughput ratios; Reconcile unit cost variances",
                 0.48, 0.36, 0, 0.09, 2.95, 0.46, 3, 0.45, 0.56, 0.22, 0.10, "FINANCE", 490.0, 0.102),
    ONETTaskSpec("11-1021", "Operations_Managers", 5, "Tier2_General_Degree", "OPS-P3", "Outer_Periphery",
                 "Compile weekly shift-utilization summaries and standard vendor SLA scorecards",
                 "Aggregate shift attendance logs; Tabulate standard delivery metrics",
                 0.75, 0.24, 0, 0.09, 2.95, 0.10, 2, 0.15, 0.33, 0.00, 0.00, "FINANCE", 450.0, 0.110),

    # 8. Management Analysts (13-1111) - Tier 2 General Degree
    ONETTaskSpec("13-1111", "Management_Analysts", 4, "Tier2_General_Degree", "MGT-C1", "Inner_Core",
                 "Design post-merger organizational restructuring under conflicting labor-retention and synergy targets",
                 "Evaluate span-of-control redundancies; Model retention-bonus clawback triggers",
                 0.25, 0.37, 0, 0.07, 2.60, 0.34, 4, 0.65, 0.46, 0.46, 0.15, "FINANCE", 510.0, 0.098),
    ONETTaskSpec("13-1111", "Management_Analysts", 4, "Tier2_General_Degree", "MGT-B2", "Middle_Buffer",
                 "Benchmark SG&A cost structures and build process-efficiency ROI models from client data",
                 "Normalize peer SG&A ratios; Calculate payback periods on workflow automation",
                 0.51, 0.38, 0, 0.07, 2.60, 0.12, 3, 0.35, 0.57, 0.14, 0.00, "FINANCE", 480.0, 0.105),
    ONETTaskSpec("13-1111", "Management_Analysts", 4, "Tier2_General_Degree", "MGT-P3", "Outer_Periphery",
                 "Synthesize stakeholder interview transcripts and format benchmark comparison decks",
                 "Summarize survey tables; Format standard KPI benchmarking slides",
                 0.77, 0.25, 0, 0.07, 2.60, -0.18, 1, 0.10, 0.00, 0.00, 0.00, "FINANCE", 445.0, 0.112),

    # 9. Clinical & Counseling Psychologists (SOC-2018 19-3033; the earlier 19-3031 was a SOC-2010 code) - Tier 3 State Licensure
    ONETTaskSpec("19-3033", "Clinical_Counseling_Psychologists", 5, "Tier3_Statutory_License", "PSY-C1", "Inner_Core",
                 "Formulate differential diagnoses and crisis-intervention plans under acute comorbidity and safety risk",
                 "Assess suicide/self-harm lethality; Differentiate trauma vs organic neurocognitive presentations",
                 0.19, 0.47, 1, 0.86, 3.15, 1.58, 6, 0.85, 0.18, 0.76, 0.35, "CLINICAL", 515.0, 0.095),
    ONETTaskSpec("19-3033", "Clinical_Counseling_Psychologists", 5, "Tier3_Statutory_License", "PSY-B2", "Middle_Buffer",
                 "Score psychometric assessment batteries and draft structured diagnostic evaluation reports",
                 "Integrate MMPI/WAIS subscale profiles; Map symptom clusters to DSM criteria",
                 0.42, 0.34, 1, 0.86, 3.15, 1.18, 4, 0.55, 0.45, 0.27, 0.15, "CLINICAL", 485.0, 0.100),
    ONETTaskSpec("19-3033", "Clinical_Counseling_Psychologists", 5, "Tier3_Statutory_License", "PSY-P3", "Outer_Periphery",
                 "Administer standardized intake symptom inventories and log behavioral session checklists",
                 "Score PHQ-9/GAD-7 intake forms; Record routine session attendance notes",
                 0.70, 0.19, 0, 0.86, 3.15, 0.50, 2, 0.20, 0.25, 0.00, 0.00, "CLINICAL", 450.0, 0.108),

    # 10. Registered Nurses (29-1141) - Tier 3 State Licensure
    ONETTaskSpec("29-1141", "Registered_Nurses", 4, "Tier3_Statutory_License", "RN-C1", "Inner_Core",
                 "Triage rapid hemodynamic decompensation and escalate conflicting post-op medication orders",
                 "Detect occult sepsis/bleeding trends; Hold contraindicated anticoagulant orders",
                 0.23, 0.43, 1, 0.92, 2.85, 1.52, 6, 0.80, 0.25, 0.69, 0.30, "CLINICAL", 500.0, 0.096),
    ONETTaskSpec("29-1141", "Registered_Nurses", 4, "Tier3_Statutory_License", "RN-B2", "Middle_Buffer",
                 "Reconcile discharge medication lists and titrate IV infusion protocols per unit standing orders",
                 "Verify heparin/insulin drip nomograms; Reconcile home vs inpatient formularies",
                 0.46, 0.36, 1, 0.92, 2.85, 1.22, 4, 0.45, 0.56, 0.22, 0.10, "CLINICAL", 470.0, 0.102),
    ONETTaskSpec("29-1141", "Registered_Nurses", 4, "Tier3_Statutory_License", "RN-P3", "Outer_Periphery",
                 "Chart routine vital-sign flowsheets and complete standard admission fall-risk screens",
                 "Enter Braden/Morse scale scores; Log routine shift intake/output totals",
                 0.73, 0.21, 0, 0.92, 2.85, 0.55, 2, 0.15, 0.33, 0.00, 0.00, "CLINICAL", 440.0, 0.110),

    # 11. Computer Systems Analysts (15-1211) - Tier 2 General Degree
    ONETTaskSpec("15-1211", "Systems_Analysts", 4, "Tier2_General_Degree", "SYS-C1", "Inner_Core",
                 "Architect legacy-to-cloud data migration schemas under conflicting latency and compliance constraints",
                 "Resolve cross-region GDPR residency locks; Prevent distributed transaction deadlocks",
                 0.27, 0.36, 0, 0.04, 2.45, 0.25, 5, 0.65, 0.38, 0.54, 0.15, "SWE_ADMIN", 510.0, 0.098),
    ONETTaskSpec("15-1211", "Systems_Analysts", 4, "Tier2_General_Degree", "SYS-B2", "Middle_Buffer",
                 "Translate functional business requirements into ERD data models and API endpoint specifications",
                 "Map business rules to SQL/JSON schemas; Define REST rate-limit specifications",
                 0.53, 0.38, 0, 0.04, 2.45, 0.02, 3, 0.35, 0.57, 0.14, 0.00, "SWE_ADMIN", 475.0, 0.104),
    ONETTaskSpec("15-1211", "Systems_Analysts", 4, "Tier2_General_Degree", "SYS-P3", "Outer_Periphery",
                 "Document system configuration tables and generate standard user-acceptance test matrices",
                 "Export data dictionary tables; Format standard UAT traceability checklists",
                 0.80, 0.26, 0, 0.04, 2.45, -0.30, 1, 0.10, 0.00, 0.00, 0.00, "SWE_ADMIN", 440.0, 0.112),

    # 12. Tax Examiners & Revenue Agents (13-2081) - Tier 3 Statutory Authority
    ONETTaskSpec("13-2081", "Tax_Examiners_Revenue_Agents", 4, "Tier3_Task_Certification", "TAX-C1", "Inner_Core",
                 "Audit multi-entity pass-through transfer pricing and disguised-sale recharacterization exposure",
                 "Evaluate IRC Sec 707/482 substance-over-form; Reconcile interrelated partner basis schedules",
                 0.20, 0.42, 1, 0.58, 2.65, 1.12, 5, 0.75, 0.27, 0.67, 0.25, "CPA", 505.0, 0.097),
    ONETTaskSpec("13-2081", "Tax_Examiners_Revenue_Agents", 4, "Tier3_Task_Certification", "TAX-B2", "Middle_Buffer",
                 "Examine corporate depreciation carryforwards and R&D tax credit qualification schedules",
                 "Verify Sec 174 amortization schedules; Audit qualified research expense substantiation",
                 0.45, 0.36, 1, 0.58, 2.65, 0.84, 3, 0.45, 0.67, 0.11, 0.10, "CPA", 475.0, 0.103),
    ONETTaskSpec("13-2081", "Tax_Examiners_Revenue_Agents", 4, "Tier3_Task_Certification", "TAX-P3", "Outer_Periphery",
                 "Cross-check reported 1099/W-2 withholding totals against standard return schedules",
                 "Match third-party information returns; Compute standard penalty interest tables",
                 0.71, 0.22, 0, 0.58, 2.65, 0.25, 2, 0.15, 0.33, 0.00, 0.00, "CPA", 445.0, 0.109),

    # 13. Economists (19-3011) - Tier 2 Graduate Degree
    ONETTaskSpec("19-3011", "Economists", 5, "Tier2_General_Degree", "ECO-C1", "Inner_Core",
                 "Identify causal policy counterfactuals when simultaneous supply-demand shocks violate exclusion restrictions",
                 "Diagnose instrument invalidity under general equilibrium spillovers; Bound structural parameters",
                 0.21, 0.41, 0, 0.05, 3.05, 0.44, 6, 0.75, 0.33, 0.60, 0.20, "FINANCE", 520.0, 0.095),
    ONETTaskSpec("19-3011", "Economists", 5, "Tier2_General_Degree", "ECO-B2", "Middle_Buffer",
                 "Estimate panel-data elasticity models and construct macroeconomic scenario forecasts",
                 "Run fixed-effects demand estimations; Project baseline CPI/employment trajectories",
                 0.43, 0.36, 0, 0.05, 3.05, 0.18, 4, 0.45, 0.56, 0.22, 0.10, "FINANCE", 485.0, 0.101),
    ONETTaskSpec("19-3011", "Economists", 5, "Tier2_General_Degree", "ECO-P3", "Outer_Periphery",
                 "Clean public census/BLS time series and generate descriptive summary tables",
                 "Harmonize seasonal adjustment series; Compute standard decile summary tables",
                 0.69, 0.23, 0, 0.05, 3.05, -0.15, 2, 0.15, 0.33, 0.00, 0.00, "FINANCE", 450.0, 0.108),

    # 14. Civil Engineers (17-2051) - Tier 3 Professional Engineer (PE) Seal
    ONETTaskSpec("17-2051", "Civil_Engineers", 5, "Tier3_Statutory_License", "ENG-C1", "Inner_Core",
                 "Certify structural foundation load safety when subsurface liquefaction interacts with seismic shear",
                 "Evaluate coupled soil-structure interaction; Seal safety-critical seismic retrofit specifications",
                 0.17, 0.48, 1, 0.74, 3.30, 1.72, 7, 0.85, 0.18, 0.76, 0.35, "ENGINEERING", 530.0, 0.092),
    ONETTaskSpec("17-2051", "Civil_Engineers", 5, "Tier3_Statutory_License", "ENG-B2", "Middle_Buffer",
                 "Calculate stormwater drainage detention capacities and dead/live beam load schedules",
                 "Verify rational-method runoff volumes; Check AISC steel member deflection limits",
                 0.41, 0.33, 1, 0.74, 3.30, 1.32, 5, 0.55, 0.45, 0.27, 0.15, "ENGINEERING", 495.0, 0.098),
    ONETTaskSpec("17-2051", "Civil_Engineers", 5, "Tier3_Statutory_License", "ENG-P3", "Outer_Periphery",
                 "Tabulate material quantity takeoffs and check standard municipal setback tables",
                 "Sum concrete/rebar bill of materials; Verify zoning setback minimums",
                 0.68, 0.19, 0, 0.74, 3.30, 0.55, 2, 0.20, 0.25, 0.00, 0.00, "ENGINEERING", 460.0, 0.106),

    # 15. Architects (17-1011) - Tier 3 Registered Architect Seal
    ONETTaskSpec("17-1011", "Architects", 5, "Tier3_Statutory_License", "ARC-C1", "Inner_Core",
                 "Resolve life-safety egress and fire-rated assembly conflicts in historic mixed-use adaptive reuse",
                 "Reconcile IBC occupancy-separation conflicts; Certify alternative means-of-egress variances",
                 0.19, 0.45, 1, 0.79, 3.15, 1.48, 6, 0.80, 0.25, 0.69, 0.25, "ENGINEERING", 520.0, 0.094),
    ONETTaskSpec("17-1011", "Architects", 5, "Tier3_Statutory_License", "ARC-B2", "Middle_Buffer",
                 "Develop commercial floor-area-ratio schedules and ADA corridor accessibility compliance plans",
                 "Compute gross-to-net leasable ratios; Check ADA turning-radius and ramp slopes",
                 0.44, 0.35, 1, 0.79, 3.15, 1.08, 4, 0.50, 0.50, 0.20, 0.10, "ENGINEERING", 485.0, 0.100),
    ONETTaskSpec("17-1011", "Architects", 5, "Tier3_Statutory_License", "ARC-P3", "Outer_Periphery",
                 "Compile door/window hardware schedules and standard finish specification sheets",
                 "Cross-reference elevation tags to finish schedules; Format CSI spec templates",
                 0.70, 0.20, 0, 0.79, 3.15, 0.40, 2, 0.15, 0.33, 0.00, 0.00, "ENGINEERING", 455.0, 0.108),

    # 16. Postsecondary Business Teachers (25-1011) - Tier 2 Academic Credential
    ONETTaskSpec("25-1011", "Business_Teachers_Postsecondary", 5, "Tier2_General_Degree", "EDU-C1", "Inner_Core",
                 "Evaluate original doctoral research designs and adjudicate academic integrity/methodology disputes",
                 "Assess identification threats in novel designs; Arbitrate complex grading/integrity appeals",
                 0.23, 0.39, 0, 0.08, 2.70, 0.32, 5, 0.70, 0.36, 0.50, 0.15, "FINANCE", 510.0, 0.098),
    ONETTaskSpec("25-1011", "Business_Teachers_Postsecondary", 5, "Tier2_General_Degree", "EDU-B2", "Middle_Buffer",
                 "Design case-study problem sets and grade structured analytical memo submissions",
                 "Construct multi-step case rubrics; Evaluate student financial/strategic calculations",
                 0.49, 0.36, 0, 0.08, 2.70, 0.08, 3, 0.40, 0.62, 0.12, 0.00, "FINANCE", 475.0, 0.104),
    ONETTaskSpec("25-1011", "Business_Teachers_Postsecondary", 5, "Tier2_General_Degree", "EDU-P3", "Outer_Periphery",
                 "Update syllabus reading schedules and score objective multiple-choice quiz banks",
                 "Format weekly LMS module calendars; Tabulate automated quiz scores",
                 0.76, 0.25, 0, 0.08, 2.70, -0.22, 1, 0.10, 0.00, 0.00, 0.00, "FINANCE", 440.0, 0.112),

    # 17. Human Resources Specialists (13-1071) - Tier 1 Market Open
    ONETTaskSpec("13-1071", "HR_Specialists", 4, "Tier1_Market_Open", "HRS-C1", "Inner_Core",
                 "Investigate cross-jurisdictional retaliation claims intersecting with FMLA/ADA accommodations",
                 "Reconcile protected-leave timelines with performance actions; Flag disparate-impact triggers",
                 0.30, 0.35, 0, 0.04, 1.95, 0.28, 4, 0.65, 0.46, 0.46, 0.15, "SWE_ADMIN", 500.0, 0.100),
    ONETTaskSpec("13-1071", "HR_Specialists", 4, "Tier1_Market_Open", "HRS-B2", "Middle_Buffer",
                 "Screen candidate qualification matrices and benchmark salary-band offer packages",
                 "Score resume experience against job reqs; Compute compa-ratio offer bands",
                 0.58, 0.38, 0, 0.04, 1.95, -0.05, 2, 0.30, 0.67, 0.00, 0.00, "SWE_ADMIN", 465.0, 0.106),
    ONETTaskSpec("13-1071", "HR_Specialists", 4, "Tier1_Market_Open", "HRS-P3", "Outer_Periphery",
                 "Process onboarding I-9/benefits checklists and generate standard job-posting templates",
                 "Verify completion of onboarding forms; Populate standard requisition templates",
                 0.84, 0.27, 0, 0.04, 1.95, -0.38, 1, 0.10, 0.00, 0.00, 0.00, "SWE_ADMIN", 435.0, 0.114),

    # 18. Executive Secretaries & Administrative Assistants (43-6011) - Tier 1 Market Open
    ONETTaskSpec("43-6011", "Executive_Secretaries", 3, "Tier1_Market_Open", "SEC-C1", "Inner_Core",
                 "Coordinate multi-time-zone board summit logistics under conflicting diplomatic/protocol constraints",
                 "Resolve quorum scheduling conflicts; Reconcile confidential board packet redactions",
                 0.32, 0.32, 0, 0.01, 1.65, 0.05, 3, 0.55, 0.55, 0.36, 0.10, "SWE_ADMIN", 490.0, 0.102),
    ONETTaskSpec("43-6011", "Executive_Secretaries", 3, "Tier1_Market_Open", "SEC-B2", "Middle_Buffer",
                 "Reconcile executive travel expense reports and draft routine correspondence memos",
                 "Verify per-diem policy caps; Format executive briefing summaries",
                 0.61, 0.38, 0, 0.01, 1.65, -0.24, 2, 0.25, 0.80, 0.00, 0.00, "SWE_ADMIN", 460.0, 0.108),
    ONETTaskSpec("43-6011", "Executive_Secretaries", 3, "Tier1_Market_Open", "SEC-P3", "Outer_Periphery",
                 "Schedule recurring calendar invites and maintain routing directories",
                 "Match meeting slot availability; Update distribution contact lists",
                 0.86, 0.30, 0, 0.01, 1.65, -0.55, 1, 0.05, 0.00, 0.00, 0.00, "SWE_ADMIN", 430.0, 0.115),
]

FULL_EXTENDED_CORPUS: List[ONETTaskSpec] = PRIMARY_CORPUS + EXTENDED_ADDITIONAL_SPECS


@dataclass
class MultiDocWorkPacket:
    packet_id: str
    task_id: str
    occ_soc: str
    occ_title: str
    ring: str
    domain_family: str
    complexity_tier: int       # 0 = Routine, 1 = Explicit Alert, 2 = 1st-Order Piecewise, 3 = 2nd-Order Precedent
    is_collision_case: bool    # True if complexity_tier >= 1
    is_silent_collision: bool  # True if complexity_tier == 3 and surface alarm keywords are suppressed
    params: Dict[str, float]
    doc_primary_intake: str
    doc_secondary_schedule: str
    doc_tertiary_annex: str
    doc_handbook: str
    gt_diagnosis: str
    gt_inference: str
    gt_treatment: float
    # provenance of the label: generated packets carry the exact outcome; archive copies record who established it
    label_source: str = "generator"       # generator | adjudicator | recorded_decision
    label_status: str = "ground_truth"    # ground_truth | adjudicated | recorded (may be wrong)


def _det_float(seed_str: str, low: float, high: float, decimals: int = 2) -> float:
    """Deterministic hash-based float generator."""
    h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:12], 16)
    frac = (h % 10_000_000) / 10_000_000.0
    return round(low + frac * (high - low), decimals)


# =============================================================================
# RULE DSL — single source of truth for the handbook, the solver and the incumbent
# =============================================================================
# Expression grammar: numbers, identifiers (V1..V4, ANNEX_SEVERITY/SEV, PRIMARY_CUTOFF, RATIO_CUTOFF,
# LET names, REG.<registry key>), + - * / **, parentheses, MIN(...), MAX(...), and `a IF cond ELSE b`.

@dataclass
class DomainRulebook:
    family: str
    lets: List[Tuple[str, str]]          # PUBLIC: intermediate quantities, in order
    tier1_rule: Tuple[str, str]          # PUBLIC R1: (label, expr) when DOC3.EXPLICIT_TIER1_OVERRIDE IS TRUE
    breach_rule: Tuple[str, str]         # PUBLIC R2: when V2 >= RATIO_CUTOFF
    base_rule: Tuple[str, str]           # PUBLIC R3: otherwise
    tier2_rule: Tuple[str, str]          # SEMI-STRUCTURED convention (archive-only, not in handbook)
    tier3_lets: List[Tuple[str, str]]    # PRIVATE registry derivations
    tier3_rule: Tuple[str, str]          # PRIVATE Tier 3 rule


RULEBOOKS: Dict[str, DomainRulebook] = {
    "LEGAL": DomainRulebook(
        "LEGAL",
        lets=[("basket_deductible", "0.08 * V4"), ("net_claim", "MAX(0, V1 - basket_deductible)")],
        tier1_rule=("LEGAL_EXPLICIT_VENUE_OVERRIDE", "net_claim * (1 - 0.35 * ANNEX_SEVERITY) - 1.15 * V3"),
        breach_rule=("LEGAL_CAPPED_INDEMNITY_SURCHARGE", "MIN(net_claim * (1 + V2), 0.85 * V1) + 0.50 * V3"),
        base_rule=("LEGAL_STANDARD_ESCROW_RELEASE", "net_claim + 0.25 * V3"),
        tier2_rule=("LEGAL_PIECEWISE_CARVEOUT_ADJUSTMENT", "net_claim * (1 - (0.52 IF V2 >= 0.12 ELSE 0.36) * SEV) - 1.20 * V3 + 0.15 * V4"),
        tier3_lets=[("eff_sev", "SEV * (1 + REG.subrogation_mult * V2)")],
        tier3_rule=("LEGAL_FIDUCIARY_SUBROGATION_REVERSAL", "net_claim * (1 - REG.fiduciary_cap_haircut * eff_sev) - REG.statutory_offset * REG.subrogation_mult * V3"),
    ),
    "CPA": DomainRulebook(
        "CPA",
        lets=[("ssp_weight", "V4 / (V1 + V4)"), ("allocated_rev", "V1 * (0.65 + 0.35 * ssp_weight)")],
        tier1_rule=("CPA_EXPLICIT_SUBSEQUENT_EVENT", "allocated_rev * (1 - 0.38 * ANNEX_SEVERITY) - 1.15 * V3"),
        breach_rule=("CPA_ASC606_DEFERRAL_VALUATION", "allocated_rev * (1 - 0.40 * V2) + 0.40 * V3"),
        base_rule=("CPA_UNQUALIFIED_LEDGER_RECON", "allocated_rev + 0.30 * V3"),
        tier2_rule=("CPA_PIECEWISE_TAX_VALUATION_ALLOWANCE", "allocated_rev * (1 - (0.48 IF V2 >= 0.12 ELSE 0.34) * SEV) - 1.25 * V3 + 0.10 * V4"),
        tier3_lets=[("eff_sev", "SEV * (1 + REG.vie_impairment_mult * V2)")],
        tier3_rule=("CPA_ASC810_VIE_GOING_CONCERN_DERECOGNITION", "allocated_rev * (1 - REG.valuation_allowance_rate * eff_sev) - REG.covenant_clawback * REG.vie_impairment_mult * V3"),
    ),
    "CLINICAL": DomainRulebook(
        "CLINICAL",
        lets=[("egfr_clearance", "MAX(0.25, 1 - 1.80 * V2)"), ("cleared_base", "V1 * (0.55 + 0.45 * egfr_clearance)")],
        tier1_rule=("CLINICAL_ACUTE_CONTRAINDICATION_HOLD", "cleared_base * (1 - 0.42 * ANNEX_SEVERITY) - 1.15 * V3"),
        breach_rule=("CLINICAL_RENAL_ADJUSTED_TITRATION", "cleared_base + 0.65 * V3 + 0.12 * V4"),
        base_rule=("CLINICAL_MAINTENANCE_PROTOCOL", "cleared_base + 0.35 * V3"),
        tier2_rule=("CLINICAL_STEPWISE_ORGAN_TAPER", "cleared_base * (1 - (0.56 IF V2 >= 0.12 ELSE 0.40) * SEV) - 1.22 * V3 + 0.08 * V4"),
        tier3_lets=[("eff_sev", "SEV * (1 + REG.cyp3a4_accumulation_factor * V2)")],
        tier3_rule=("CLINICAL_CYP3A4_POLYPHARMACY_WASHOUT", "cleared_base * (1 - REG.renal_washout_weight * eff_sev) - REG.toxicity_penalty * REG.cyp3a4_accumulation_factor * V3"),
    ),
    "FINANCE": DomainRulebook(
        "FINANCE",
        lets=[("pv_cf1", "V1 / (1 + V2)"), ("pv_cf2", "(0.85 * V4) / ((1 + V2) ** 2)"), ("dcf_core", "pv_cf1 + pv_cf2")],
        tier1_rule=("FINANCE_EXPLICIT_COVENANT_SWEEP", "dcf_core * (1 - 0.32 * ANNEX_SEVERITY) - 1.15 * V3"),
        breach_rule=("FINANCE_SPREAD_STRESS_VALUATION", "dcf_core * (1 - 0.25 * V2) + 0.45 * V3"),
        base_rule=("FINANCE_BASELINE_DCF_APPROVAL", "dcf_core + 0.20 * V3"),
        tier2_rule=("FINANCE_RATCHET_LIQUIDITY_HAIRCUT", "dcf_core * (1 - (0.45 IF V2 >= 0.12 ELSE 0.30) * SEV) - 1.18 * V3 + 0.12 * V4"),
        tier3_lets=[("eff_sev", "SEV * (1 + REG.csa_convexity_mult * V2)")],
        tier3_rule=("FINANCE_CSA_CONVEXITY_COLLATERAL_CALL", "dcf_core * (1 - REG.springing_sweep_rate * eff_sev) - REG.liquidity_haircut * REG.csa_convexity_mult * V3"),
    ),
    "ENGINEERING": DomainRulebook(
        "ENGINEERING",
        lets=[("combo_1", "1.40 * V1"), ("combo_2", "1.20 * V1 + 1.60 * V4"), ("combo_3", "1.20 * V1 + 1.00 * V4 + 180.0 * V2"), ("lrfd_envelope", "MAX(combo_1, combo_2, combo_3) / 1.45")],
        tier1_rule=("ENGINEERING_ZONING_VARIANCE_DEDUCTION", "lrfd_envelope * (1 - 0.40 * ANNEX_SEVERITY) - 1.15 * V3"),
        breach_rule=("ENGINEERING_LRFD_SHEAR_REINFORCEMENT", "lrfd_envelope * (1 + 0.50 * V2) + 0.60 * V3"),
        base_rule=("ENGINEERING_STANDARD_MEMBER_SEAL", "lrfd_envelope + 0.40 * V3"),
        tier2_rule=("ENGINEERING_PIECEWISE_DEFLECTION_SHORING", "lrfd_envelope * (1 - (0.54 IF V2 >= 0.12 ELSE 0.38) * SEV) - 1.25 * V3 + 0.14 * V4"),
        tier3_lets=[("eff_sev", "SEV * (1 + REG.liquefaction_shear_amp * V2)")],
        tier3_rule=("ENGINEERING_SEISMIC_LIQUEFACTION_RETROFIT", "lrfd_envelope * (1 - REG.buckling_knockdown * eff_sev) - REG.seismic_safety_margin * REG.liquefaction_shear_amp * V3"),
    ),
    "SWE_ADMIN": DomainRulebook(
        "SWE_ADMIN",
        lets=[("utilization_rho", "MIN(0.88, 0.35 + 2.40 * V2)"), ("queue_throughput", "V1 * (1 - 0.35 * utilization_rho) + 0.25 * V4")],
        tier1_rule=("SWE_EXPLICIT_SCHEMA_ROLLBACK", "queue_throughput * (1 - 0.30 * ANNEX_SEVERITY) - 1.15 * V3"),
        breach_rule=("SWE_BACKPRESSURE_THROTTLE_SCALE", "queue_throughput * (1 + 0.40 * V2) + 0.55 * V3"),
        base_rule=("SWE_BASELINE_SLA_PASS", "queue_throughput + 0.15 * V3"),
        tier2_rule=("SWE_PIECEWISE_RETRY_CIRCUIT_BREAKER", "queue_throughput * (1 - (0.44 IF V2 >= 0.12 ELSE 0.28) * SEV) - 1.18 * V3 + 0.10 * V4"),
        tier3_lets=[("eff_sev", "SEV * (1 + REG.quorum_retry_amp * V2)")],
        tier3_rule=("SWE_QUORUM_SPLIT_BRAIN_COMPENSATION", "queue_throughput * (1 - REG.partition_rollback_wt * eff_sev) - REG.sla_penalty_mult * REG.quorum_retry_amp * V3"),
    ),
}


def _dsl_to_python(expr: str) -> str:
    s = re.sub(r"\bMIN\(", "min(", expr)
    s = re.sub(r"\bMAX\(", "max(", s)
    s = re.sub(r"\bIF\b", "if", s)
    s = re.sub(r"\bELSE\b", "else", s)
    s = re.sub(r"\bREG\.", "REG_", s)
    return s


@functools.lru_cache(maxsize=None)
def _compile_expr(expr: str):
    return _ast.parse(_dsl_to_python(expr), mode="eval").body


def _ev(node, env: Dict[str, float]) -> float:
    if isinstance(node, _ast.Constant):
        return float(node.value)
    if isinstance(node, _ast.Name):
        return float(env[node.id])
    if isinstance(node, _ast.BinOp):
        l, r = _ev(node.left, env), _ev(node.right, env)
        op = type(node.op)
        if op is _ast.Add:
            return l + r
        if op is _ast.Sub:
            return l - r
        if op is _ast.Mult:
            return l * r
        if op is _ast.Div:
            return l / r
        if op is _ast.Pow:
            return l ** r
        raise ValueError(f"operator not allowed: {op.__name__}")
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
        return -_ev(node.operand, env)
    if isinstance(node, _ast.IfExp):
        return _ev(node.body, env) if _ev(node.test, env) else _ev(node.orelse, env)
    if isinstance(node, _ast.Compare) and len(node.ops) == 1:
        l, r = _ev(node.left, env), _ev(node.comparators[0], env)
        op = type(node.ops[0])
        return float({_ast.GtE: l >= r, _ast.Gt: l > r, _ast.LtE: l <= r, _ast.Lt: l < r, _ast.Eq: l == r}[op])
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id in ("min", "max"):
        f = min if node.func.id == "min" else max
        return float(f(*[_ev(a, env) for a in node.args]))
    raise ValueError(f"expression node not allowed: {type(node).__name__}")


def eval_expr(expr: str, env: Dict[str, float]) -> float:
    """Evaluates one rule-language expression against a variable environment."""
    return _ev(_compile_expr(expr), env)


def _rule_env(spec: ONETTaskSpec, v1: float, v2: float, v3: float, v4: float, sev: float) -> Dict[str, float]:
    env = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "SEV": sev, "ANNEX_SEVERITY": sev,
           "PRIMARY_CUTOFF": spec.primary_cutoff, "RATIO_CUTOFF": spec.ratio_cutoff}
    for k, v in PRECEDENT_REGISTRY[spec.domain_family].items():
        env["REG_" + k] = v
    return env


def evaluate_rulebook(spec: ONETTaskSpec, v1: float, v2: float, v3: float, v4: float, sev: float,
                      complexity_tier: int) -> Tuple[str, str, float]:
    """Interprets the domain rulebook for a given tier. Used by the exact solver and by the incumbent."""
    rb = RULEBOOKS[spec.domain_family]
    env = _rule_env(spec, v1, v2, v3, v4, sev)
    for name, ex in rb.lets:
        env[name] = eval_expr(ex, env)
    diag_band = "HIGH_EXPOSURE" if v1 >= spec.primary_cutoff else "MODERATE_EXPOSURE"
    thresh_band = "THRESHOLD_BREACH" if v2 >= spec.ratio_cutoff else "WITHIN_TOLERANCE"
    if complexity_tier == 0:
        label, ex = rb.breach_rule if v2 >= spec.ratio_cutoff else rb.base_rule
    elif complexity_tier == 1:
        label, ex = rb.tier1_rule
    elif complexity_tier == 2:
        label, ex = rb.tier2_rule
    else:
        for name, e3 in rb.tier3_lets:
            env[name] = eval_expr(e3, env)
        label, ex = rb.tier3_rule
    return f"{diag_band}__{thresh_band}", label, round(eval_expr(ex, env), 2)


def solve_non_isomorphic_domain_gt(spec: ONETTaskSpec, p: Dict[str, float], complexity_tier: int) -> Tuple[str, str, float]:
    """Exact ground truth (D*, I*, T*) = the rulebook interpreted at the packet's true tier."""
    return evaluate_rulebook(spec, p["v1_mag"], p["v2_rat"], p["v3_aux"], p["v4_sched"], p["annex_sev"], complexity_tier)


# =============================================================================
# Document parsing shared by the non-compiled agents (novice, incumbent)
# =============================================================================
_NUM_RE = r"([0-9]+(?:\.[0-9]+)?)"


def parse_doc_fields(doc_intake: str, doc_sched: str, doc_annex: str) -> Dict[str, object]:
    def grab(doc: str, key: str) -> Optional[float]:
        m = re.search(re.escape(key) + r"\s*[:=]\s*" + _NUM_RE, doc)
        return float(m.group(1)) if m else None
    f: Dict[str, object] = {
        "V1": grab(doc_intake, "V1_PRIMARY_MAGNITUDE"),
        "V2": grab(doc_intake, "V2_THRESHOLD_RATIO"),
        "V3": grab(doc_intake, "V3_RESERVE_ADJUSTMENT"),
        "NOISE": grab(doc_intake, "ADMIN_NOISE_ID"),
        "V4": grab(doc_sched, "V4_SCHEDULE_PARAMETER"),
        "EXPLICIT_TIER1_OVERRIDE": re.search(r"EXPLICIT_TIER1_OVERRIDE\s*=\s*TRUE", doc_annex) is not None,
        "TIER3_MEMO": "TIER3_COUPLED_PRECEDENT" in doc_annex,
    }
    for key in ("ANNEX_SEVERITY", "severity_index", "base_haircut"):
        f[key] = grab(doc_annex, key)
    f["ARCHIVE_METRIC"] = grab(doc_annex, "Archive Reference Metric")
    return f


# =============================================================================
# Workplace documents
# =============================================================================
def render_public_handbook(spec: ONETTaskSpec) -> str:
    rb = RULEBOOKS[spec.domain_family]
    L = [
        f"=== PUBLIC O*NET TASK SPECIFICATION & RULE HANDBOOK [{spec.task_id}] ===",
        f"OCCUPATION: {spec.occ_title} ({spec.occ_soc}) | JOB_ZONE: {spec.job_zone} | RING: {spec.ring}",
        f"DOMAIN_FAMILY: {spec.domain_family} | OVERLAP_J: {spec.overlap_j:.2f}",
        f"ONET_TASK: {spec.onet_task}",
        f"DWA_BUNDLE: {spec.dwa_bundle}",
        f"DIAGNOSTIC_CUTOFFS: PRIMARY_CUTOFF={spec.primary_cutoff:.2f} | RATIO_CUTOFF={spec.ratio_cutoff:.4f}",
        "DOCUMENT_FIELDS: V1 := DOC1.V1_PRIMARY_MAGNITUDE | V2 := DOC1.V2_THRESHOLD_RATIO | "
        "V3 := DOC1.V3_RESERVE_ADJUSTMENT | V4 := DOC2.V4_SCHEDULE_PARAMETER | ANNEX_SEVERITY := DOC3.ANNEX_SEVERITY",
        "DOCUMENT_FORMAT: DOC1 and DOC2 carry fields as 'KEY: value' (e.g. 'V1_PRIMARY_MAGNITUDE: 512.40'); DOC3 carries flags "
        "and fields as 'KEY = value' (e.g. 'ALERT: EXPLICIT_TIER1_OVERRIDE = TRUE | ANNEX_SEVERITY = 0.75'); numbers are decimals.",
        "OUTPUT_FORMAT: diagnosis is the two DIAGNOSE labels joined by two underscores, e.g. HIGH_EXPOSURE__THRESHOLD_BREACH; "
        "inference is the INFER label of the first matching RULE; treatment is the TREAT value rounded to 2 decimals.",
        f"PUBLIC_RULEBOOK ({spec.domain_family}; rules are evaluated in order, first match wins):",
    ]
    for name, ex in rb.lets:
        L.append(f"  LET {name} := {ex}")
    L.append("  DIAGNOSE HIGH_EXPOSURE IF V1 >= PRIMARY_CUTOFF ELSE MODERATE_EXPOSURE")
    L.append("  DIAGNOSE THRESHOLD_BREACH IF V2 >= RATIO_CUTOFF ELSE WITHIN_TOLERANCE")
    L.append(f"  RULE R1 WHEN DOC3.EXPLICIT_TIER1_OVERRIDE IS TRUE -> INFER {rb.tier1_rule[0]} ; TREAT {rb.tier1_rule[1]}")
    L.append(f"  RULE R2 WHEN V2 >= RATIO_CUTOFF -> INFER {rb.breach_rule[0]} ; TREAT {rb.breach_rule[1]}")
    L.append(f"  RULE R3 OTHERWISE -> INFER {rb.base_rule[0]} ; TREAT {rb.base_rule[1]}")
    L.append(f"VALIDATION_SUBCLAUSES (COUNT={spec.n_subclauses}):")
    for c in range(1, spec.n_subclauses + 1):
        gate = round(0.015 * c + 0.002 * spec.job_zone, 4)
        bound = round(gate * (1.0 + 0.1 * spec.job_zone) * (1.0 - 0.2 * spec.overlap_j), 5)
        L.append(f"  CHECK C{c} FLAG INVARIANT_{spec.domain_family}_{c} WHEN V1 < {bound}")
    if spec.overlap_j < 0.55:
        L.append(f"  CHECK CR FLAG PROPRIETARY_RATIO_GUARD WHEN V1 / MAX(1, V4) > {round(25.0 / max(0.1, spec.overlap_j), 2)}")
    L.append("NOTE: Semi-structured protocol exceptions (severity_index) and precedent-registry couplings are NOT codified in this public handbook.")
    return "\n".join(L) + "\n"


def _render_four_workplace_docs(
    spec: ONETTaskSpec, pkt_id: str, p: Dict[str, float], c_tier: int, is_silent: bool
) -> Tuple[str, str, str, str]:
    fam = spec.domain_family
    v1, v2, v3, v4 = p["v1_mag"], p["v2_rat"], p["v3_aux"], p["v4_sched"]
    nse, sev, bref = p["noise_metric"], p["annex_sev"], p["bref"]

    doc_handbook = render_public_handbook(spec)

    doc_intake = (
        f"[DOC 1: PRIMARY WORKPLACE INTAKE {pkt_id} | DOMAIN: {fam}]\n"
        f"Occupation: {spec.occ_title} ({spec.occ_soc}) | Ring: {spec.ring}\n"
        f"  V1_PRIMARY_MAGNITUDE: {v1:.2f}\n"
        f"  V2_THRESHOLD_RATIO: {v2:.4f}\n"
        f"  V3_RESERVE_ADJUSTMENT: {v3:.2f}\n"
        f"  ADMIN_NOISE_ID: {nse:.2f}\n"
    )
    doc_sched = (
        f"[DOC 2: SECONDARY AUXILIARY SCHEDULE {pkt_id} | DOMAIN: {fam}]\n"
        f"Cross-Referenced Schedule Parameter:\n"
        f"  V4_SCHEDULE_PARAMETER: {v4:.2f}\n"
        f"  SCHEDULE_VERIFICATION_CODE: {fam}_SCHED_OK\n"
    )
    if c_tier == 0:
        doc_annex = (
            f"[DOC 3: TERTIARY ANNEX & AUDIT LOG {pkt_id}]\n"
            f"Domain Family: {fam} | Exception Status: NO_CONFLICT_IDENTIFIED.\n"
            f"Archive Reference Metric: {bref:.2f} (Non-binding historical log).\n"
        )
    elif c_tier == 1:
        doc_annex = (
            f"[DOC 3: TERTIARY ANNEX & AUDIT LOG {pkt_id}]\n"
            f"Domain Family: {fam} | ALERT: EXPLICIT_TIER1_OVERRIDE = TRUE | ANNEX_SEVERITY = {sev:.2f}\n"
            f"Complexity Tier: TIER1_SINGLE_CLAUSE\n"
        )
    elif c_tier == 2:
        doc_annex = (
            f"[DOC 3: TERTIARY ANNEX & AUDIT LOG {pkt_id}]\n"
            f"Domain Family: {fam} | Semi-Structured Protocol Exception: severity_index={sev:.2f}\n"
            f"Complexity Tier: TIER2_PIECEWISE_COLLISION\n"
        )
    elif not is_silent:
        doc_annex = (
            f"[DOC 3: TERTIARY ANNEX — INNER CORE PRECEDENT MEMO {pkt_id}]\n"
            f"Domain Family: {fam} | Complexity Tier: TIER3_COUPLED_PRECEDENT\n"
            f"Latent Cross-Document Coefficient: base_haircut={sev:.2f}.\n"
            f"Requires 3-hop lookup in {fam} Precedent Registry."
        )
    else:
        # Silent collision: the annex uses the same template as a routine case, with different field values. The only trace of the latent
        # precedent coupling is that the archive metric carries the severity (>= registry metric floor)
        # while V2 exceeds the registry ratio gate. Nothing in the handbook or the public archive names it.
        doc_annex = (
            f"[DOC 3: TERTIARY ANNEX & AUDIT LOG {pkt_id}]\n"
            f"Domain Family: {fam} | Exception Status: NO_CONFLICT_IDENTIFIED.\n"
            f"Archive Reference Metric: {sev:.2f} (Non-binding historical log).\n"
        )
    return doc_intake, doc_sched, doc_annex, doc_handbook


def _build_packets(spec: ONETTaskSpec, n_packets: int, prefix: str) -> List[MultiDocWorkPacket]:
    packets: List[MultiDocWorkPacket] = []
    n_collision = int(round(n_packets * spec.collision_density))
    n_tier3 = int(round(n_collision * spec.tier3_deep_share))
    n_tier2 = int(round(n_collision * spec.tier2_piecewise_share))
    n_tier1 = max(0, n_collision - n_tier2 - n_tier3)
    n_silent = int(round(n_tier3 * spec.silent_collision_prob))

    for idx in range(n_packets):
        pkt_id = f"{spec.task_id}-{prefix}{idx + 1:02d}"
        sd = pkt_id + CFG.seed_salt
        if CFG.random_tier_allocation:
            # Each packet draws its tier from the task's tier probabilities (hash-seeded), so tier counts
            # vary across packet universes and governance shares acquire sampling variance.
            u = _det_float(f"{sd}_tier", 0.0, 1.0, 6)
            if u >= spec.collision_density:
                c_tier, is_silent = 0, False
            else:
                v = u / max(spec.collision_density, 1e-9)
                if v < spec.tier3_deep_share:
                    c_tier = 3
                    is_silent = _det_float(f"{sd}_silent", 0.0, 1.0, 6) < spec.silent_collision_prob
                elif v < spec.tier3_deep_share + spec.tier2_piecewise_share:
                    c_tier, is_silent = 2, False
                else:
                    c_tier, is_silent = 1, False
        elif idx >= n_collision:
            c_tier, is_silent = 0, False
        elif idx < n_tier1:
            c_tier, is_silent = 1, False
        elif idx < (n_tier1 + n_tier2):
            c_tier, is_silent = 2, False
        else:
            c_tier = 3
            is_silent = (idx - (n_tier1 + n_tier2)) < n_silent
        v1_mag = _det_float(f"{sd}_v1", *CFG.gen_v1_range, 2)
        v2_rat = _det_float(f"{sd}_v2", *CFG.gen_v2_range, 4)
        if is_silent and CFG.silent_gate_sufficient:
            gate = PRECEDENT_REGISTRY[spec.domain_family]["silent_ratio_gate"]
            v2_rat = _det_float(f"{sd}_v2s", gate + CFG.gen_silent_v2_margin, CFG.gen_silent_v2_max, 4)
        v3_aux = _det_float(f"{sd}_v3", *CFG.gen_v3_range, 2)
        v4_sched = _det_float(f"{sd}_v4", *CFG.gen_v4_range, 2)
        noise_metric = _det_float(f"{sd}_nse", *CFG.gen_noise_range, 2)
        annex_sev = _det_float(f"{sd}_sev", *CFG.gen_severity_range, 2) if c_tier >= 1 else 0.0
        # The routine metric range lies entirely below the registry floor (0.50) and the severity range entirely
        # above it, so in the reference configuration a deep audit always separates a silent collision from routine
        # work; decoy_note_rate > 0 makes the ranges overlap (robustness block F).
        bref = _det_float(f"{sd}_bref", *CFG.gen_routine_metric_range, 2) if c_tier == 0 else 0.0
        if c_tier == 0 and CFG.decoy_note_rate > 0.0 and _det_float(f"{sd}_decoy", 0.0, 1.0, 4) < CFG.decoy_note_rate:
            # Benign supplemental note whose metric sits in the severity range: a routine case that looks,
            # to the registry gate, like a silent collision (over-escalation risk for the incumbent).
            bref = _det_float(f"{sd}_decoyval", *CFG.gen_decoy_metric_range, 2)

        params = {"v1_mag": v1_mag, "v2_rat": v2_rat, "v3_aux": v3_aux, "v4_sched": v4_sched,
                  "noise_metric": noise_metric, "annex_sev": annex_sev, "bref": bref}
        gt_d, gt_i, gt_t = solve_non_isomorphic_domain_gt(spec, params, c_tier)
        d_in, d_sc, d_an, d_hb = _render_four_workplace_docs(spec, pkt_id, params, c_tier, is_silent)
        packets.append(MultiDocWorkPacket(
            packet_id=pkt_id, task_id=spec.task_id, occ_soc=spec.occ_soc, occ_title=spec.occ_title,
            ring=spec.ring, domain_family=spec.domain_family, complexity_tier=c_tier,
            is_collision_case=(c_tier >= 1), is_silent_collision=is_silent, params=params,
            doc_primary_intake=d_in, doc_secondary_schedule=d_sc, doc_tertiary_annex=d_an, doc_handbook=d_hb,
            gt_diagnosis=gt_d, gt_inference=gt_i, gt_treatment=gt_t,
        ))
    return packets


def build_multi_doc_packets_for_task(spec: ONETTaskSpec, n_packets: Optional[int] = None, split: str = "eval") -> List[MultiDocWorkPacket]:
    """Evaluation packets (`split='eval'`, ids ...-Pnn) or sandbox draws (`split='sandbox'`, ids ...-Snn)."""
    n = CFG.n_eval_packets if n_packets is None else n_packets
    return _build_packets(spec, n, "P" if split == "eval" else "S")


def build_public_sandbox_archive(spec: ONETTaskSpec, n_packets: Optional[int] = None) -> List[MultiDocWorkPacket]:
    """
    The PUBLIC resolved-case archive available to the Challenger's compiler: labelled past packets of
    Tiers 0-2 only. Tier 3 resolutions depend on the private registry and are never archived publicly.
    Sandbox ids (...-Snn) are disjoint from evaluation ids (...-Pnn).
    """
    n = CFG.archive_size if n_packets is None else n_packets
    max_tier = 3 if CFG.archive_includes_tier3 else 2
    return [p for p in _build_packets(spec, n, "S") if p.complexity_tier <= max_tier]


# =============================================================================
# Macro outcomes (Table 4 bridge) — loaded from a provenance-tracked data file
# =============================================================================
_DATA_INPUTS_MACRO_PATH = os.path.join(juri_paths.ROOT, "data_inputs", "macro_outcomes.csv")
MACRO_OUTCOMES_PATH = _DATA_INPUTS_MACRO_PATH if os.path.exists(_DATA_INPUTS_MACRO_PATH) else os.path.join(juri_paths.ROOT, "macro_outcomes.csv")


def load_macro_outcomes(path: str = MACRO_OUTCOMES_PATH) -> Dict[str, Dict[str, str]]:
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return {r["task_id"]: r for r in csv.DictReader(lines)}


def attach_macro_outcomes(corpus: List[ONETTaskSpec], path: str = MACRO_OUTCOMES_PATH) -> Dict[str, int]:
    """Fills the macro fields of each spec from the data file. Returns a count of rows per `source`."""
    rows = load_macro_outcomes(path)
    counts: Dict[str, int] = {}
    for s in corpus:
        r = rows[s.task_id]
        f = lambda v: float(v) if str(v).strip() not in ("", "nan", "NA") else float("nan")
        s.macro_auto_delta_19_26 = f(r["macro_auto_delta_19_26"])
        s.macro_telemetry_gap_z = f(r["macro_telemetry_gap_z"])
        s.macro_semantic_shift_z = f(r["macro_semantic_shift_z"])
        s.macro_source = r["source"]
        counts[s.macro_source] = counts.get(s.macro_source, 0) + 1
    return counts


if os.path.exists(MACRO_OUTCOMES_PATH):
    MACRO_SOURCE_COUNTS = attach_macro_outcomes(FULL_EXTENDED_CORPUS)
else:
    MACRO_SOURCE_COUNTS = {"MISSING_FILE": len(FULL_EXTENDED_CORPUS)}
