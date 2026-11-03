#!/usr/bin/env python3
"""Diagnostic summary for the archive-label, client-learning and forecast options.

Reads the per-period task files written by run_dynamics.py and reports, for each diagnostic variant beside its
reference run, period-16 means by statutory gate (and over all tasks) of: tool end-to-end accuracy, poach / settle /
gate shares, status-quo error, visible and silent error, gate state, client belief and wrongly labelled archive cases.
Writes data_outputs/diagnostics_summary.csv. Makes no new simulations.

    python3 run_diagnostics_summary.py
"""
import argparse
import csv
import math
import os
import juri_paths

OUT = os.path.join(juri_paths.ROOT, "data_outputs")
VARIANTS = [  # (diagnostic, reference run, variant run)
    ("archive labels: adjudicated (reference)", "mandatory", "mandatory"),
    ("archive labels: recorded decisions", "mandatory", "mandatory_labels_recorded"),
    ("archive labels: noisy", "mandatory", "mandatory_labels_noisy"),
    ("archive labels: delayed", "mandatory", "mandatory_labels_delayed"),
    ("archive labels: adjudicated (reference)", "discretion", "discretion"),
    ("archive labels: recorded decisions", "discretion", "discretion_labels_recorded"),
    ("archive labels: noisy", "discretion", "discretion_labels_noisy"),
    ("archive labels: delayed", "discretion", "discretion_labels_delayed"),
    ("client belief fixed (reference)", "mandatory", "mandatory"),
    ("client learning from observed errors", "mandatory", "mandatory_clientlearn"),
    ("client belief fixed (reference)", "discretion", "discretion"),
    ("client learning from observed errors", "discretion", "discretion_clientlearn"),
    ("client belief fixed (reference)", "strategic", "strategic"),
    ("client learning from observed errors", "strategic", "strategic_clientlearn"),
    ("forecast: independent draws (reference)", "strategic", "strategic"),
    ("forecast: realised future (perfect-foresight benchmark)", "strategic", "strategic_perfectforesight"),
    ("forecast: independent draws, universe 2", "strategic", "strategic_u2"),
]
OCC_KEY = "occ_title"
COLS = ["tool_e2e", "poach", "settle", "gate", "sq_error", "visible_error", "silent_error", "gate_state", "client_belief",
        "archive_wrong_labels"]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    argparse.ArgumentParser(description="Diagnostic summary of the archive-label, client-learning and forecast options (no new simulations)").parse_args()
    with open(os.path.join(OUT, "results_occupation_summary_18.csv"), newline="") as f:
        gates = {r[OCC_KEY]: str(int(float(r["statutory_gate"]))) for r in csv.DictReader(f)}
    rows = []
    for diag, ref, var in VARIANTS:
        path = os.path.join(OUT, f"dynamics_{var}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run reproduce.py first")
        with open(path, newline="") as f:
            data = list(csv.DictReader(f))
        last = str(max(int(r["period"]) for r in data))
        for label, sel in (("licensed", "1"), ("unlicensed", "0"), ("all", None)):
            rr = [r for r in data if r["period"] == last and (sel is None or gates.get(r["occ_title"]) == sel)]
            row = {"diagnostic": diag, "regime": ref, "run": var, "period": last, "group": label, "n_tasks": len(rr)}
            for c in COLS:
                row[c] = round(mean(fl(r.get(c)) for r in rr), 4)
            rows.append(row)
    with open(os.path.join(OUT, "diagnostics_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'diagnostic':<52} {'run':<28} {'grp':<10} " + " ".join(f"{c[:10]:>10}" for c in COLS))
    for r in rows:
        print(f"{r['diagnostic'][:52]:<52} {r['run']:<28} {r['group']:<10} " + " ".join(f"{r[c]:>10.3f}" for c in COLS))


if __name__ == "__main__":
    main()
