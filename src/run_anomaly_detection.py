#!/usr/bin/env python3
"""
Detection versus resolution of silent precedent cases.

The compiled v1-v3 parsers read structured key=value features; they do not read the colon-delimited
"Archive Reference Metric", so their failure on silent cases is partly a feature-extraction restriction of that
parser. This script adds a richer detector that uses only public information: it learns, from the public sandbox
archive of each task, the empirical support (min, max) of every numeric field in the three workplace documents,
including colon-delimited fields, and flags an evaluation packet as unfamiliar when any numeric value lies outside
the archived support of its field or a field name never appears in the archive.

Detection and resolution are reported separately. Detection is the share of silent cases flagged (and the
false-alarm share among routine cases). Resolution is the share of flagged silent cases the v3 compiled tool
answers correctly; a flag says that a case is unfamiliar, it does not compute the private precedent outcome.

Variants: the reference configuration (routine metrics 0.12-0.44, silent metrics 0.64-0.91: disjoint support),
decoy notes (25% of routine cases carry a benign metric in the severity range), and full overlap (routine metrics
drawn from 0.12-0.91, so routine and silent supports overlap). Results are about this detector and this parser;
they do not bound what any other agent could detect.

Output: data_outputs/anomaly_detection.csv
"""
import argparse
import csv
import os
import juri_paths
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import domain_cases as dc
from model_config import set_config, reset_config, CFG

BASE = juri_paths.ROOT
OUT = os.path.join(BASE, "data_outputs")
_NUM = re.compile(r"([A-Za-z][A-Za-z0-9_ ]*?)\s*[:=]\s*(-?\d+(?:\.\d+)?)")

VARIANTS = [
    ("registered_disjoint_support", {}),
    ("decoy_notes_0.25", {"decoy_note_rate": 0.25}),
    ("routine_metric_overlap", {"gen_routine_metric_range": (0.12, 0.91)}),
]


def numeric_fields(pkt) -> Dict[str, float]:
    """Every `name: number` or `name=number` pair in the three workplace documents (not the handbook).
    Identifier-like fields (packet ids, verification codes) carry no number and are skipped."""
    out: Dict[str, float] = {}
    for doc in (pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex):
        for name, val in _NUM.findall(doc):
            out[name.strip().upper().replace(" ", "_")] = float(val)
    return out


def support(archive) -> Dict[str, Tuple[float, float]]:
    lo: Dict[str, float] = {}
    hi: Dict[str, float] = {}
    for p in archive:
        for k, v in numeric_fields(p).items():
            lo[k] = min(lo.get(k, v), v)
            hi[k] = max(hi.get(k, v), v)
    return {k: (lo[k], hi[k]) for k in lo}


def flagged(pkt, sup: Dict[str, Tuple[float, float]]) -> bool:
    for k, v in numeric_fields(pkt).items():
        if k not in sup:
            return True
        a, b = sup[k]
        if v < a or v > b:
            return True
    return False


def run_variant(name: str, overrides: Dict[str, object]) -> Dict[str, object]:
    from run_live_benchmark import run_corpus_evaluation
    set_config(**overrides)
    try:
        tools = os.path.join(BASE, "anomaly_tools", name)
        _, traces, _, _ = run_corpus_evaluation(dc.PRIMARY_CORPUS, f"ANOMALY_{name}", tools)
        v3_ok = {t["packet_id"]: bool(t["A_Tool_v3_e2e"]) for t in traces}
        c = defaultdict(int)
        for spec in dc.PRIMARY_CORPUS:
            sup = support(dc.build_public_sandbox_archive(spec))
            for p in dc.build_multi_doc_packets_for_task(spec, split="eval"):
                f = flagged(p, sup)
                if p.is_silent_collision:
                    kind = "silent"
                elif p.complexity_tier == 0:
                    kind = "routine"
                else:
                    kind = "overt_collision"
                c[f"n_{kind}"] += 1
                c[f"flag_{kind}"] += f
                if kind == "silent":
                    c["v3_correct_silent"] += v3_ok.get(p.packet_id, False)
                    if f:
                        c["v3_correct_flagged_silent"] += v3_ok.get(p.packet_id, False)
    finally:
        reset_config()
        shutil.rmtree(os.path.join(BASE, "anomaly_tools"), ignore_errors=True)
    pct = lambda a, b: round(100.0 * a / b, 1) if b else ""
    return {"variant": name, "overrides": str(overrides), "n_silent": c["n_silent"], "n_routine": c["n_routine"], "n_overt": c["n_overt_collision"],
            "detect_silent_pct": pct(c["flag_silent"], c["n_silent"]), "false_alarm_routine_pct": pct(c["flag_routine"], c["n_routine"]),
            "flag_overt_pct": pct(c["flag_overt_collision"], c["n_overt_collision"]),
            "v3_correct_silent_pct": pct(c["v3_correct_silent"], c["n_silent"]),
            "v3_correct_given_flag_pct": pct(c["v3_correct_flagged_silent"], c["flag_silent"])}


def main() -> None:
    argparse.ArgumentParser(description="Detection vs resolution of silent cases with a public numeric-support detector").parse_args()
    rows = [run_variant(n, o) for n, o in VARIANTS]
    path = os.path.join(OUT, "anomaly_detection.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'Variant':<30} | {'silent n':>8} | {'detected':>8} | {'false alarm':>11} | {'overt flagged':>13} | {'v3 correct':>10} | {'v3 correct | flag':>17}")
    for r in rows:
        print(f"{r['variant']:<30} | {r['n_silent']:>8} | {r['detect_silent_pct']:>7}% | {r['false_alarm_routine_pct']:>10}% | {r['flag_overt_pct']:>12}% | "
              f"{r['v3_correct_silent_pct']:>9}% | {r['v3_correct_given_flag_pct']:>16}%")
    print(f"wrote {os.path.relpath(path, BASE)}. Detection by a public support detector; resolution by the v3 compiled tool.")


if __name__ == "__main__":
    main()
