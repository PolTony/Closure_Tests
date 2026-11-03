#!/usr/bin/env python3
"""
Auto-Compiled Occupational Task Routine: MGT-C1 (Management_Analysts - Inner_Core)
Domain Engine: FINANCE | Compiler Generation: v3
Provenance: public handbook (3 LET, 2 DIAGNOSE, 3 RULE, 5 CHECK statements) + 1 exception handler(s) induced from the public resolved-case archive
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
    if (v1 < 0.03059):
        validation_flags.append("INVARIANT_FINANCE_1")
    if (v1 < 0.05054):
        validation_flags.append("INVARIANT_FINANCE_2")
    if (v1 < 0.07049):
        validation_flags.append("INVARIANT_FINANCE_3")
    if (v1 < 0.09044):
        validation_flags.append("INVARIANT_FINANCE_4")
    if ((v1 / max(1, v4)) > 100.0):
        validation_flags.append("PROPRIETARY_RATIO_GUARD")
    diag_0 = "HIGH_EXPOSURE" if (v1 >= 510.0) else "MODERATE_EXPOSURE"
    diag_1 = "THRESHOLD_BREACH" if (v2 >= 0.098) else "WITHIN_TOLERANCE"
    diagnosis = "__".join([diag_0, diag_1])
    if re.search(r"EXPLICIT_TIER1_OVERRIDE\s*=\s*TRUE", doc_annex):
        annex_severity = _field(doc_annex, "ANNEX_SEVERITY")
        handled_fields.append("ANNEX_SEVERITY")
        return _out(diagnosis, "FINANCE_EXPLICIT_COVENANT_SWEEP", ((dcf_core * (1 - (0.32 * annex_severity))) - (1.15 * v3)), validation_flags, "R1", annex_fields, handled_fields)
    severity_index_val = _field(doc_annex, "severity_index")
    if severity_index_val is not None:
        handled_fields.append("severity_index")
        c0, c1, c2 = (0.45, -1.18, 0.12) if v2 >= 0.1181 else (0.3, -1.18, 0.12)
        return _out(diagnosis, "FINANCE_RATCHET_LIQUIDITY_HAIRCUT", dcf_core * (1 - c0 * severity_index_val) + c1 * v3 + c2 * v4, validation_flags, "H1", annex_fields, handled_fields)
    if (v2 >= 0.098):
        return _out(diagnosis, "FINANCE_SPREAD_STRESS_VALUATION", ((dcf_core * (1 - (0.25 * v2))) + (0.45 * v3)), validation_flags, "R2", annex_fields, handled_fields)
    return _out(diagnosis, "FINANCE_BASELINE_DCF_APPROVAL", (dcf_core + (0.20 * v3)), validation_flags, "R3", annex_fields, handled_fields)
