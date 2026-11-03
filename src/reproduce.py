"""Run the declarative reproduction manifest in dependency order.

Each job records its source, input, dependency, output and log hashes. A check fails
on changed or missing provenance. Requested dependencies must finish in this run;
previous successful records never satisfy an upstream job being rerun.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import juri_paths

BASE = juri_paths.ROOT
MANIFEST = os.path.join(BASE, "reproduce", "manifest.json")
OUT = os.path.join(BASE, "data_outputs")
LOGS = os.path.join(OUT, "run_logs")
RUN_MANIFEST = os.path.join(OUT, "RUN_MANIFEST.json")


def _resolve_figures_python():
    env_py = os.environ.get("JURI_FIGURES_PYTHON")
    if env_py:
        return env_py
    for cand in (sys.executable, "/usr/bin/python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3"):
        if cand and os.path.exists(cand):
            res = subprocess.run([cand, "-c", "import matplotlib"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return cand
    return sys.executable


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    # Installed modules live beside juri_paths, rather than in the output workspace.
    source_root = os.path.dirname(juri_paths.module_path("juri_paths.py"))
    out = {f: sha256(os.path.join(source_root, f)) for f in sorted(os.listdir(source_root)) if f.endswith(".py")}
    for root, dirs, names in os.walk(os.path.join(BASE, "data_inputs")):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for n in sorted(names):
            if not n.startswith("."):
                p = os.path.join(root, n)
                out[os.path.relpath(p, BASE)] = sha256(p)
    for item in juri_paths.RESOURCE_ITEMS:
        if item.endswith(".jsonl"):
            out[item] = sha256(os.path.join(BASE, item))
    out["reproduce/manifest.json"] = sha256(MANIFEST)
    for name in sorted(os.listdir(os.path.join(BASE, "reproduce"))):
        if name.endswith((".py", ".sh")):
            out["reproduce/" + name] = sha256(os.path.join(BASE, "reproduce", name))
    return out


def load_manifest():
    with open(MANIFEST) as f:
        man = json.load(f)
    ids = [j["id"] for j in man["jobs"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("manifest has duplicate job ids")
    for j in man["jobs"]:
        for d in j["deps"]:
            if d not in ids:
                raise SystemExit(f"job {j['id']} depends on unknown job {d}")
    return man


def output_path(job, name):
    return os.path.join(BASE if job.get("output_root") == "project" else OUT, name)


def input_hashes(job):
    out = code_hashes()
    jobs = {j["id"]: j for j in load_manifest()["jobs"]}
    paths = list(job.get("inputs", []))
    for dep in job["deps"]:
        paths.extend(os.path.relpath(output_path(jobs[dep], p), BASE) for p in jobs[dep]["outputs"])
        paths.append(f"data_outputs/run_logs/{dep}.log")
    for p in sorted(set(paths)):
        out[p] = sha256(os.path.join(BASE, p))
    return out


def run_job(job):
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"{job['id']}.log")
    t0 = time.time()
    before = input_hashes(job)
    with open(log, "w") as f:
        exe = _resolve_figures_python() if job.get("interpreter") == "figures" else sys.executable
        proc = subprocess.run([exe, "-B", juri_paths.module_path(job["args"][0])] + job["args"][1:], cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    rec = {"id": job["id"], "args": job["args"], "exit": proc.returncode, "seconds": round(time.time() - t0, 1),
           "log": os.path.relpath(log, BASE), "log_sha256": sha256(log), "supports": job["supports"],
           "output_root": job.get("output_root", "data_outputs"), "inputs": before, "outputs": {}}
    missing = []
    for o in job["outputs"]:
        p = output_path(job, o)
        if not os.path.exists(p) or os.path.getmtime(p) < t0 - 1:
            missing.append(o)
        else:
            rec["outputs"][o] = sha256(p)
    rec["missing_or_stale"] = missing
    rec["inputs_changed_during_run"] = before != input_hashes(job)
    rec["ok"] = proc.returncode == 0 and not missing and not rec["inputs_changed_during_run"]
    return rec


def record_errors(job, rec):
    errors = []
    if not rec or not rec.get("ok"):
        return ["missing or failed job record"]
    if rec.get("args") != job["args"] or rec.get("output_root", "data_outputs") != job.get("output_root", "data_outputs"):
        errors.append("job definition changed")
    if set(rec.get("outputs", {})) != set(job["outputs"]):
        errors.append("declared output inventory changed")
    for o, h in rec.get("outputs", {}).items():
        p = output_path(job, o)
        if not os.path.isfile(p) or sha256(p) != h:
            errors.append(f"output missing or changed: {o}")
    log = os.path.join(BASE, rec.get("log", ""))
    if not os.path.isfile(log) or sha256(log) != rec.get("log_sha256"):
        errors.append("printed output log missing or changed")
    try:
        if rec.get("inputs") != input_hashes(job):
            errors.append("source, inputs or dependency artifacts changed")
    except OSError as e:
        errors.append(f"missing input: {e.filename}")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", default="", help="comma-separated job ids; dependencies must have valid records or be selected too")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.parallel < 1:
        ap.error("--parallel must be at least 1")
    man = load_manifest()
    jobs = {j["id"]: j for j in man["jobs"]}
    if args.list:
        for j in man["jobs"]:
            print(f"{j['id']:<40} stage {j['stage']}  {' '.join(j['args'])}")
        return
    rm = {"schema": "boundaries-run-manifest/2", "jobs": {}}
    if os.path.exists(RUN_MANIFEST):
        with open(RUN_MANIFEST) as f:
            rm = json.load(f)
    if args.check:
        bad = 0
        for jid, job in jobs.items():
            for err in record_errors(job, rm["jobs"].get(jid)):
                print(f"FAIL {jid}: {err}")
                bad += 1
        print("CHECK OK" if bad == 0 else f"CHECK FAILED ({bad})")
        raise SystemExit(1 if bad else 0)
    todo = list(jobs) if args.all else list(dict.fromkeys(x.strip() for x in args.jobs.split(",") if x.strip()))
    if not todo:
        ap.error("choose --all, --jobs, --list or --check")
    for x in todo:
        if x not in jobs:
            ap.error(f"unknown job {x}")
    # Invalidate every selected job before scheduling, including a formerly successful dependency.
    done = {k for k in jobs if k not in todo and not record_errors(jobs[k], rm["jobs"].get(k))}
    for x in todo:
        rm["jobs"].pop(x, None)
    rm.update(schema="boundaries-run-manifest/2", python=sys.version.split()[0], platform=platform.platform(),
              code_and_inputs=code_hashes(), updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(OUT, exist_ok=True)

    def save():
        tmp = RUN_MANIFEST + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rm, f, indent=1, sort_keys=True)
        os.replace(tmp, RUN_MANIFEST)

    save()
    pending, failed = list(todo), []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        running = {}
        while pending or running:
            for x in list(pending):
                if any(d in failed for d in jobs[x]["deps"]):
                    pending.remove(x)
                    failed.append(x)
                    print(f"SKIP {x}: a dependency failed", flush=True)
            ready = [x for x in pending if all(d in done for d in jobs[x]["deps"])]
            for x in ready[:args.parallel - len(running)]:
                pending.remove(x)
                running[ex.submit(run_job, jobs[x])] = x
                print(f"START {x}", flush=True)
            if not running:
                if pending:
                    raise SystemExit(f"unsatisfied dependencies for: {', '.join(pending)}; select or rerun their dependencies")
                break
            fut = next(as_completed(running))
            x = running.pop(fut)
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"id": x, "ok": False, "error": f"{type(e).__name__}: {e}"}
            rm["jobs"][x] = rec
            (done.add(x) if rec["ok"] else failed.append(x))
            print(f"{'DONE' if rec['ok'] else 'FAIL'} {x} ({rec.get('seconds', 0)} s)" + (f" {rec.get('error', '')}" if not rec['ok'] else ""), flush=True)
            save()
    print(f"finished: {len(todo) - len(failed)} ok, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
