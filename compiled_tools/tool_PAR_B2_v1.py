#!/usr/bin/env python3
"""
Auto-Compiled Occupational Task Routine: PAR-B2 (Paralegals - Middle_Buffer)
Domain Engine: LEGAL | Compiler Generation: v1
Provenance: public handbook (2 LET, 2 DIAGNOSE, 3 RULE, 4 CHECK statements)
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
    basket_deductible = (0.08 * v4)
    net_claim = max(0, (v1 - basket_deductible))
    if (v1 < 0.02885):
        validation_flags.append("INVARIANT_LEGAL_1")
    if (v1 < 0.04767):
        validation_flags.append("INVARIANT_LEGAL_2")
    if (v1 < 0.06648):
        validation_flags.append("INVARIANT_LEGAL_3")
    if ((v1 / max(1, v4)) > 48.08):
        validation_flags.append("PROPRIETARY_RATIO_GUARD")
    diag_0 = "HIGH_EXPOSURE" if (v1 >= 460.0) else "MODERATE_EXPOSURE"
    diag_1 = "THRESHOLD_BREACH" if (v2 >= 0.105) else "WITHIN_TOLERANCE"
    diagnosis = "__".join([diag_0, diag_1])
    if re.search(r"EXPLICIT_TIER1_OVERRIDE\s*=\s*TRUE", doc_annex):
        annex_severity = _field(doc_annex, "ANNEX_SEVERITY")
        handled_fields.append("ANNEX_SEVERITY")
        return _out(diagnosis, "LEGAL_EXPLICIT_VENUE_OVERRIDE", ((net_claim * (1 - (0.35 * annex_severity))) - (1.15 * v3)), validation_flags, "R1", annex_fields, handled_fields)
    if (v2 >= 0.105):
        return _out(diagnosis, "LEGAL_CAPPED_INDEMNITY_SURCHARGE", (min((net_claim * (1 + v2)), (0.85 * v1)) + (0.50 * v3)), validation_flags, "R2", annex_fields, handled_fields)
    return _out(diagnosis, "LEGAL_STANDARD_ESCROW_RELEASE", (net_claim + (0.25 * v3)), validation_flags, "R3", annex_fields, handled_fields)
