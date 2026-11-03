#!/usr/bin/env python3
"""
Auto-Compiled Occupational Task Routine: ECO-P3 (Economists - Outer_Periphery)
Domain Engine: FINANCE | Compiler Generation: v3
Provenance: public handbook (3 LET, 2 DIAGNOSE, 3 RULE, 2 CHECK statements)
"""
import re
from typing import Dict

_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_KV = r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[0-9]+(?:\.[0-9]+)?"


def _field(doc: str, key: str):
    m = re.search(re.escape(key) + r"\s*[:=]\s*" + _NUM, doc)
    return float(m.group(1)) if m else None


def _out(diagnosis, inference, treatment, flags, fired, annex_fields, handled):
    return {"diagnosis": diagnosis, "inference": inference, "treatment": round(treatment, 2),
            "validation_flags": flags, "fired": fired, "unhandled_fields": sorted(set(annex_fields) - set(handled))}


def execute_compiled_routine(doc_intake: str, doc_sched: str, doc_annex: str) -> Dict[str, object]:
    v1 = _field(doc_intake, "V1_PRIMARY_MAGNITUDE")
    v2 = _field(doc_intake, "V2_THRESHOLD_RATIO")
    v3 = _field(doc_intake, "V3_RESERVE_ADJUSTMENT")
    v4 = _field(doc_sched, "V4_SCHEDULE_PARAMETER")
    if v1 is None or v2 is None or v3 is None or v4 is None:
        return {"diagnosis": "PARSE_ERROR", "inference": "PARSE_ERROR", "treatment": 0.0, "validation_flags": [], "fired": "NONE", "unhandled_fields": []}
    validation_flags = []
    annex_fields = re.findall(_KV, doc_annex)
    handled_fields = []
    pv_cf1 = (v1 / (1 + v2))
    pv_cf2 = ((0.85 * v4) / ((1 + v2) ** 2))
    dcf_core = (pv_cf1 + pv_cf2)
    if (v1 < 0.03233):
        validation_flags.append("INVARIANT_FINANCE_1")
    if (v1 < 0.05172):
        validation_flags.append("INVARIANT_FINANCE_2")
    diag_0 = "HIGH_EXPOSURE" if (v1 >= 450.0) else "MODERATE_EXPOSURE"
    diag_1 = "THRESHOLD_BREACH" if (v2 >= 0.108) else "WITHIN_TOLERANCE"
    diagnosis = "__".join([diag_0, diag_1])
    if re.search(r"EXPLICIT_TIER1_OVERRIDE\s*=\s*TRUE", doc_annex):
        annex_severity = _field(doc_annex, "ANNEX_SEVERITY")
        handled_fields.append("ANNEX_SEVERITY")
        return _out(diagnosis, "FINANCE_EXPLICIT_COVENANT_SWEEP", ((dcf_core * (1 - (0.32 * annex_severity))) - (1.15 * v3)), validation_flags, "R1", annex_fields, handled_fields)
    if (v2 >= 0.108):
        return _out(diagnosis, "FINANCE_SPREAD_STRESS_VALUATION", ((dcf_core * (1 - (0.25 * v2))) + (0.45 * v3)), validation_flags, "R2", annex_fields, handled_fields)
    return _out(diagnosis, "FINANCE_BASELINE_DCF_APPROVAL", (dcf_core + (0.20 * v3)), validation_flags, "R3", annex_fields, handled_fields)
