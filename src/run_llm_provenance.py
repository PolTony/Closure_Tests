"""Provenance summary of the shipped live AI-arm transcript.

The transcript file data_outputs/llm_transcripts_primary.jsonl holds 794 records, but the reported live run
(18 tasks, 10 evaluation packets per task) issued only a subset of them. This script takes the exact set of request
keys that the reported configuration issues, as recorded by the legacy event replay of that configuration
(results_llm_arm_replay_primary_legacy_replay_replay_provenance.json, written by
`run_llm_arm.py --provider replay --corpus primary --n-eval 10 --replay-mode legacy --tag legacy_replay`), and
summarises served model, explicit server-side fallback events, role counts, tokens and seconds for the selected run
separately from the whole file. It makes no model requests.
"""
import csv
import json
import os
import juri_paths
from collections import Counter

BASE = juri_paths.ROOT
OUT = os.path.join(BASE, "data_outputs")
TRANSCRIPT = os.path.join(OUT, "llm_transcripts_primary.jsonl")
REPLAY_PROV = os.path.join(OUT, "results_llm_arm_replay_primary_legacy_replay_replay_provenance.json")
PRIMARY_MODEL = "claude-opus-5-5"


def summarise(recs, label):
    served = Counter(r.get("model") for r in recs)
    fallback_events = sum(1 for r in recs if r.get("fallback"))
    fallback_field_missing = sum(1 for r in recs if r.get("fallback") is None)
    roles = Counter(r["role"] for r in recs)
    return {
        "scope": label,
        "n_requests": len(recs),
        "served_primary": served.get(PRIMARY_MODEL, 0),
        "served_other": sum(v for k, v in served.items() if k != PRIMARY_MODEL),
        "served_by_model": dict(served),
        "explicit_fallback_events": fallback_events,
        "fallback_field_absent": fallback_field_missing,
        "non_primary_without_fallback_event": sum(1 for r in recs if r.get("model") != PRIMARY_MODEL and not r.get("fallback")),
        "roles": dict(sorted(roles.items())),
        "input_tokens": sum(int((r.get("usage") or {}).get("input", 0)) for r in recs),
        "output_tokens": sum(int((r.get("usage") or {}).get("output", 0)) for r in recs),
        "model_seconds": round(sum(float(r.get("seconds") or 0.0) for r in recs), 1),
    }


def main():
    recs = [json.loads(line) for line in open(TRANSCRIPT, encoding="utf-8") if line.strip()]
    by_key = {r["key"]: r for r in recs}
    if len(by_key) != len(recs):
        raise SystemExit("transcript has duplicate keys; selection by key would be ambiguous")
    if not os.path.exists(REPLAY_PROV):
        raise SystemExit(f"missing {REPLAY_PROV}; run the legacy replay of the reported configuration first")
    prov = json.load(open(REPLAY_PROV, encoding="utf-8"))
    keys = prov["requested_keys"]
    if len(set(keys)) != len(keys):
        raise SystemExit("the reported configuration requested a key twice; check the replay")
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"{len(missing)} requested keys absent from the transcript")
    selected = [by_key[k] for k in keys]
    sel_set = set(keys)
    unused = [r for r in recs if r["key"] not in sel_set]
    rows = [summarise(selected, "selected_reported_run"), summarise(unused, "file_records_not_requested"),
            summarise(recs, "whole_transcript_file")]
    meta = {"transcript": os.path.relpath(TRANSCRIPT, BASE), "selection_source": os.path.relpath(REPLAY_PROV, BASE),
            "selection_provenance": prov.get("provenance"), "primary_model": PRIMARY_MODEL,
            "note": ("The selected run is the exact key set issued by the reported 18-task, 10-packet configuration. "
                     "Records in the file but not requested come from earlier or partial runs and are excluded from "
                     "reported counts. 'explicit_fallback_events' counts records whose fallback list is non-empty; "
                     "records served by a non-primary model without such an event are reported separately.")}
    with open(os.path.join(OUT, "llm_provenance_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "summaries": rows}, f, indent=1)
    flat = ["scope", "n_requests", "served_primary", "served_other", "explicit_fallback_events", "fallback_field_absent",
            "non_primary_without_fallback_event", "input_tokens", "output_tokens", "model_seconds", "roles"]
    with open(os.path.join(OUT, "llm_provenance_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(flat)
        for r in rows:
            w.writerow([json.dumps(r[c], sort_keys=True) if isinstance(r[c], dict) else r[c] for c in flat])
    # per-role table for the selected run (appendix Table A12)
    import statistics as st
    role_rows = []
    for role in sorted({r["role"] for r in selected}):
        rr = [r for r in selected if r["role"] == role]
        role_rows.append({"role": role, "calls": len(rr),
                          "median_seconds": round(st.median(float(r.get("seconds") or 0.0) for r in rr), 1),
                          "mean_output_tokens": round(st.mean(int((r.get("usage") or {}).get("output", 0)) for r in rr)),
                          "served_non_primary": sum(1 for r in rr if r.get("model") != PRIMARY_MODEL),
                          "explicit_fallback_events": sum(1 for r in rr if r.get("fallback"))})
    with open(os.path.join(OUT, "llm_provenance_by_role.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(role_rows[0]))
        w.writeheader(); w.writerows(role_rows)
    for r in role_rows:
        print("  ", r)
    for r in rows:
        print(f"{r['scope']}: n={r['n_requests']} primary={r['served_primary']} other={r['served_other']} "
              f"fallback_events={r['explicit_fallback_events']} roles={r['roles']} tokens in/out={r['input_tokens']}/{r['output_tokens']} "
              f"seconds={r['model_seconds']}")


if __name__ == "__main__":
    main()
