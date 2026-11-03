#!/usr/bin/env python3
"""
Auto-Compiled Occupational Task Routine: ARC-C1 (Architects - Inner_Core)
Domain Engine: ENGINEERING | Compiler Generation: v2
Provenance: public handbook (4 LET, 2 DIAGNOSE, 3 RULE, 7 CHECK statements) + 1 exception handler(s) induced from the public resolved-case archive
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
    combo_1 = (1.40 * v1)
    combo_2 = ((1.20 * v1) + (1.60 * v4))
    combo_3 = (((1.20 * v1) + (1.00 * v4)) + (180.0 * v2))
    lrfd_envelope = (max(combo_1, combo_2, combo_3) / 1.45)
    if (v1 < 0.03608):
        validation_flags.append("INVARIANT_ENGINEERING_1")
    if (v1 < 0.05772):
        validation_flags.append("INVARIANT_ENGINEERING_2")
    if (v1 < 0.07937):
        validation_flags.append("INVARIANT_ENGINEERING_3")
    if (v1 < 0.10101):
        validation_flags.append("INVARIANT_ENGINEERING_4")
    if (v1 < 0.12266):
        validation_flags.append("INVARIANT_ENGINEERING_5")
    if (v1 < 0.1443):
        validation_flags.append("INVARIANT_ENGINEERING_6")
    if ((v1 / max(1, v4)) > 131.58):
        validation_flags.append("PROPRIETARY_RATIO_GUARD")
    diag_0 = "HIGH_EXPOSURE" if (v1 >= 520.0) else "MODERATE_EXPOSURE"
    diag_1 = "THRESHOLD_BREACH" if (v2 >= 0.094) else "WITHIN_TOLERANCE"
    diagnosis = "__".join([diag_0, diag_1])
    if re.search(r"EXPLICIT_TIER1_OVERRIDE\s*=\s*TRUE", doc_annex):
        annex_severity = _field(doc_annex, "ANNEX_SEVERITY")
        handled_fields.append("ANNEX_SEVERITY")
        return _out(diagnosis, "ENGINEERING_ZONING_VARIANCE_DEDUCTION", ((lrfd_envelope * (1 - (0.40 * annex_severity))) - (1.15 * v3)), validation_flags, "R1", annex_fields, handled_fields)
    severity_index_val = _field(doc_annex, "severity_index")
    if severity_index_val is not None:
        handled_fields.append("severity_index")
        c0, c1, c2 = (0.38, -1.25, 0.14)
        return _out(diagnosis, "ENGINEERING_PIECEWISE_DEFLECTION_SHORING", lrfd_envelope * (1 - c0 * severity_index_val) + c1 * v3 + c2 * v4, validation_flags, "H1", annex_fields, handled_fields)
    if (v2 >= 0.094):
        return _out(diagnosis, "ENGINEERING_LRFD_SHEAR_REINFORCEMENT", ((lrfd_envelope * (1 + (0.50 * v2))) + (0.60 * v3)), validation_flags, "R2", annex_fields, handled_fields)
    return _out(diagnosis, "ENGINEERING_STANDARD_MEMBER_SEAL", (lrfd_envelope + (0.40 * v3)), validation_flags, "R3", annex_fields, handled_fields)
