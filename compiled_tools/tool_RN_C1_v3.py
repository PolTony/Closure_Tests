#!/usr/bin/env python3
"""
Auto-Compiled Occupational Task Routine: RN-C1 (Registered_Nurses - Inner_Core)
Domain Engine: CLINICAL | Compiler Generation: v3
Provenance: public handbook (2 LET, 2 DIAGNOSE, 3 RULE, 7 CHECK statements) + 1 exception handler(s) induced from the public resolved-case archive
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
    egfr_clearance = max(0.25, (1 - (1.80 * v2)))
    cleared_base = (v1 * (0.55 + (0.45 * egfr_clearance)))
    if (v1 < 0.03072):
        validation_flags.append("INVARIANT_CLINICAL_1")
    if (v1 < 0.05075):
        validation_flags.append("INVARIANT_CLINICAL_2")
    if (v1 < 0.07079):
        validation_flags.append("INVARIANT_CLINICAL_3")
    if (v1 < 0.09082):
        validation_flags.append("INVARIANT_CLINICAL_4")
    if (v1 < 0.11085):
        validation_flags.append("INVARIANT_CLINICAL_5")
    if (v1 < 0.13089):
        validation_flags.append("INVARIANT_CLINICAL_6")
    if ((v1 / max(1, v4)) > 108.7):
        validation_flags.append("PROPRIETARY_RATIO_GUARD")
    diag_0 = "HIGH_EXPOSURE" if (v1 >= 500.0) else "MODERATE_EXPOSURE"
    diag_1 = "THRESHOLD_BREACH" if (v2 >= 0.096) else "WITHIN_TOLERANCE"
    diagnosis = "__".join([diag_0, diag_1])
    if re.search(r"EXPLICIT_TIER1_OVERRIDE\s*=\s*TRUE", doc_annex):
        annex_severity = _field(doc_annex, "ANNEX_SEVERITY")
        handled_fields.append("ANNEX_SEVERITY")
        return _out(diagnosis, "CLINICAL_ACUTE_CONTRAINDICATION_HOLD", ((cleared_base * (1 - (0.42 * annex_severity))) - (1.15 * v3)), validation_flags, "R1", annex_fields, handled_fields)
    severity_index_val = _field(doc_annex, "severity_index")
    if severity_index_val is not None:
        handled_fields.append("severity_index")
        c0, c1, c2 = (0.56, -1.22, 0.08) if v2 >= 0.1192 else (0.4, -1.22, 0.08)
        return _out(diagnosis, "CLINICAL_STEPWISE_ORGAN_TAPER", cleared_base * (1 - c0 * severity_index_val) + c1 * v3 + c2 * v4, validation_flags, "H1", annex_fields, handled_fields)
    if (v2 >= 0.096):
        return _out(diagnosis, "CLINICAL_RENAL_ADJUSTED_TITRATION", ((cleared_base + (0.65 * v3)) + (0.12 * v4)), validation_flags, "R2", annex_fields, handled_fields)
    return _out(diagnosis, "CLINICAL_MAINTENANCE_PROTOCOL", (cleared_base + (0.35 * v3)), validation_flags, "R3", annex_fields, handled_fields)
