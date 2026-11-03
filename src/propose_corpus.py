#!/usr/bin/env python3
"""
Corpus proposal from O*NET (propose_corpus.py) — JURI Lab

Reads the hand-off's O*NET 31.0 task files (data_inputs/onet31_tasks_lab18.csv and onet31_tasks_proposed_additions.csv)
and, for every occupation, proposes the three ring tasks the laboratory needs:

  inner core      the most important task with the LOWEST task-level AI exposure (the profession's defining work)
  middle buffer   the most important task with mid-range exposure (contested work shared with adjacent occupations)
  outer periphery the most important task with the HIGHEST exposure (routine work)

among the occupation's core-type tasks ranked by O*NET importance. Importance weights are renormalized over the three
chosen tasks, as the laboratory's task_importance_wt requires. This is the first step of the occupation compiler in the
research program: the task statements and their detailed work activities are the input from which a rulebook is drafted
for expert validation. Output: data_inputs/proposed_corpus_tasks.csv and a printed summary.
"""

import csv
import io
import os
import juri_paths
from collections import defaultdict

BASE = juri_paths.ROOT
INP = os.path.join(BASE, "data_inputs")


def load(name):
    with open(os.path.join(INP, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(io.StringIO("".join(l for l in f if not l.startswith("#")))))


def fnum(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def propose(tasks, top_n=9):
    # group by the 8-digit O*NET-SOC code and keep the base occupation (code ending .00) when it exists, so that
    # specialties such as "Search Marketing Strategists" (13-1161.01) do not duplicate their parent occupation
    by_code = defaultdict(list)
    for t in tasks:
        by_code[t["O*NET-SOC Code"]].append(t)
    by_occ = {}
    for code, tt in by_code.items():
        soc = code.split(".")[0]
        if code.endswith(".00") or not any(c.endswith(".00") and c.startswith(soc) for c in by_code):
            by_occ[(soc, tt[0]["Title"])] = tt
    rows = []
    for (soc, title), tt in sorted(by_occ.items()):
        core = [t for t in tt if t["Task Type"].strip().lower() == "core"] or tt
        top = sorted(core, key=lambda t: -fnum(t["importance_IM_1to5"]))[:top_n]
        if len(top) < 3:
            continue
        by_exp = sorted(top, key=lambda t: fnum(t["task_exposure_e_g_2025"]))
        pick = {"Inner_Core": by_exp[0], "Middle_Buffer": by_exp[len(by_exp) // 2], "Outer_Periphery": by_exp[-1]}
        if len({id(v) for v in pick.values()}) < 3:
            continue
        w = {k: fnum(v["importance_IM_1to5"]) for k, v in pick.items()}
        tot = sum(w.values()) or 1.0
        for ring, t in pick.items():
            rows.append({"soc2018": soc, "occ_title": title, "ring": ring, "task_id_onet": t["Task ID"], "task": t["Task"],
                         "importance_IM": t["importance_IM_1to5"], "task_importance_wt": round(w[ring] / tot, 4),
                         "task_exposure_e_g_2025": t["task_exposure_e_g_2025"], "eloundou_gate": t["eloundou_human_label_gate"],
                         "n_core_tasks": len(core), "source": t["source"]})
    return rows


def merge_tasks(*sources):
    """Combine source task files and drop overlapping rows by full identifier (O*NET-SOC code, Task ID) before
    ring selection. The first source that contains a task keeps it."""
    seen, out = set(), []
    for src in sources:
        for t in src:
            key = (t["O*NET-SOC Code"].strip(), t["Task ID"].strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    return out


def check_unique_rings(rows):
    """Assert exactly one task per occupation and ring, and three rings per occupation."""
    from collections import Counter
    c = Counter((r["soc2018"], r["ring"]) for r in rows)
    dup = [k for k, n in c.items() if n != 1]
    if dup:
        raise SystemExit(f"duplicate occupation-ring rows: {dup[:5]}")
    per = Counter(r["soc2018"] for r in rows)
    bad = [k for k, n in per.items() if n != 3]
    if bad:
        raise SystemExit(f"occupations without exactly three rings: {bad[:5]}")


def main():
    lab = load("onet31_tasks_lab18.csv")
    add = load("onet31_tasks_proposed_additions.csv")
    rows = propose(merge_tasks(lab, add))
    check_unique_rings(rows)
    out = os.path.join(INP, "proposed_corpus_tasks.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    occs = sorted({(r["soc2018"], r["occ_title"]) for r in rows})
    print(f"proposed {len(rows)} ring tasks for {len(occs)} occupations -> {os.path.relpath(out, BASE)}")
    print(f"\n{'SOC':<8} {'Occupation':<44} | {'core exposure':>13} | {'buffer exp.':>11} | {'periphery exp.':>14}")
    print("-" * 100)
    for soc, title in occs:
        e = {r["ring"]: fnum(r["task_exposure_e_g_2025"]) for r in rows if r["soc2018"] == soc}
        print(f"{soc:<8} {title[:44]:<44} | {e['Inner_Core']:13.2f} | {e['Middle_Buffer']:11.2f} | {e['Outer_Periphery']:14.2f}")


if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(description='Proposes ring tasks for 57 occupations from the O*NET files').parse_args()   # no options; rejects stray arguments and serves --help
    main()
