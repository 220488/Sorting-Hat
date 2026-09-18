#!/usr/bin/env python3
"""Parse Harbor job directories into an exhaustive per-trial CSV.

Captures everything Terminal-Bench/Harbor exposes: outcome, per-test results,
token/cost accounting, all four timing phases, agent metadata, environment
config, reproducibility refs, and full error detail for failed trials.

Usage:
  python3 aws/parse_full.py aws/results/<job>/ -o aws/baseline_full.csv
  (also writes <out>.tests.csv with one row per individual verifier test)
"""
import json, csv, glob, os, argparse
from datetime import datetime

FIELDS = [
    # identity / provenance
    "job_name","trial_name","trial_id","task","task_org","task_ref","task_checksum",
    "dataset_source","started_at","finished_at",
    # experiment factors
    "harness","harness_version","model","provider","n_concurrent","timeout_multiplier",
    "agent_timeout_multiplier","verifier_timeout_multiplier",
    # outcome
    "reward","passed","status",
    # partial credit
    "tests_total","tests_passed","tests_failed","tests_skipped","tests_pending","tests_other",
    "test_pass_rate","failed_test_names","test_duration_s",
    # errors
    "exception_type","exception_message","exception_traceback",
    # tokens & cost
    "n_input_tokens","n_cache_tokens","n_output_tokens","total_tokens","cost_usd",
    "cost_per_1k_tokens",
    # agent behaviour
    "n_episodes","summarization_count","n_api_requests",
    "api_time_total_ms","api_time_mean_ms","api_time_max_ms",
    # timing, per phase
    "env_setup_s","agent_setup_s","agent_exec_s","verifier_s","total_s","overhead_s",
    # environment
    "env_type","verifier_env_mode","override_cpus","override_memory_mb","override_storage_mb",
    # joined task metadata
    "category","difficulty","keywords",
]

def secs(b):
    if not b: return None
    try:
        t0=datetime.fromisoformat(b["started_at"].replace("Z","+00:00"))
        t1=datetime.fromisoformat(b["finished_at"].replace("Z","+00:00"))
        return round((t1-t0).total_seconds(),2)
    except Exception: return None

def load_meta(p="task_metadata.csv"):
    return {r["task"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}

def parse(d, job, meta, tests_out):
    r   = json.load(open(f"{d}/result.json"))
    cfg = json.load(open(f"{d}/config.json"))
    ag  = cfg.get("agent") or {}
    env = cfg.get("environment") or {}
    ar  = r.get("agent_result") or {}
    ai  = r.get("agent_info") or {}
    mi  = ai.get("model_info") or {}
    md  = ar.get("metadata") or {}
    exc = r.get("exception_info") or {}

    reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")

    # per-test detail from the verifier's CTRF report
    s, failed, tdur = {}, [], None
    ctrf = f"{d}/verifier/ctrf.json"
    if os.path.exists(ctrf):
        res = json.load(open(ctrf)).get("results", {})
        s = res.get("summary", {}) or {}
        for t in res.get("tests", []) or []:
            tests_out.append({"trial_id": r.get("id"), "trial_name": r.get("trial_name"),
                              "task": (r.get("task_name") or "").split("/")[-1],
                              "harness": ag.get("name"), "model": mi.get("name") or ag.get("model_name"),
                              "test_name": t.get("name"), "status": t.get("status"),
                              "duration_s": t.get("duration"), "retries": t.get("retries"),
                              "file_path": t.get("file_path")})
            if t.get("status") != "passed": failed.append(t.get("name"))
        if s.get("start") and s.get("stop"):
            tdur = round(s["stop"] - s["start"], 3)

    api = md.get("api_request_times_msec") or []
    ti  = ar.get("n_input_tokens") or 0
    to  = ar.get("n_output_tokens") or 0
    cost = ar.get("cost_usd")
    task = (r.get("task_name") or "").split("/")[-1]
    tm  = meta.get(task, {})
    tot = secs({"started_at": r["started_at"], "finished_at": r["finished_at"]})
    phases = [secs(r.get(k)) for k in ("environment_setup","agent_setup","agent_execution","verifier")]

    if exc:                         status = "ERROR"
    elif reward == 1.0:             status = "PASS"
    elif reward == 0.0:             status = "FAIL"
    else:                           status = "UNKNOWN"

    return {
        "job_name": job, "trial_name": r.get("trial_name"), "trial_id": r.get("id"),
        "task": task, "task_org": (r.get("task_id") or {}).get("org"),
        "task_ref": (r.get("task_id") or {}).get("ref"), "task_checksum": r.get("task_checksum"),
        "dataset_source": r.get("source"),
        "started_at": r.get("started_at"), "finished_at": r.get("finished_at"),
        "harness": ag.get("name"), "harness_version": ai.get("version"),
        "model": mi.get("name") or ag.get("model_name"), "provider": mi.get("provider"),
        "n_concurrent": ag.get("n_concurrent"), "timeout_multiplier": cfg.get("timeout_multiplier"),
        "agent_timeout_multiplier": cfg.get("agent_timeout_multiplier"),
        "verifier_timeout_multiplier": cfg.get("verifier_timeout_multiplier"),
        "reward": reward, "passed": int(reward == 1.0) if reward is not None else None,
        "status": status,
        "tests_total": s.get("tests"), "tests_passed": s.get("passed"),
        "tests_failed": s.get("failed"), "tests_skipped": s.get("skipped"),
        "tests_pending": s.get("pending"), "tests_other": s.get("other"),
        "test_pass_rate": round(s["passed"]/s["tests"], 4) if s.get("tests") else None,
        "failed_test_names": "; ".join(failed) if failed else None,
        "test_duration_s": tdur,
        "exception_type": exc.get("exception_type"),
        "exception_message": exc.get("exception_message"),
        "exception_traceback": (exc.get("exception_traceback") or "")[:4000] or None,
        "n_input_tokens": ti, "n_cache_tokens": ar.get("n_cache_tokens") or 0,
        "n_output_tokens": to, "total_tokens": ti + to, "cost_usd": cost,
        "cost_per_1k_tokens": round(cost/(ti+to)*1000, 6) if cost and (ti+to) else None,
        "n_episodes": md.get("n_episodes"), "summarization_count": md.get("summarization_count"),
        "n_api_requests": len(api) or None,
        "api_time_total_ms": round(sum(api), 1) if api else None,
        "api_time_mean_ms": round(sum(api)/len(api), 1) if api else None,
        "api_time_max_ms": round(max(api), 1) if api else None,
        "env_setup_s": phases[0], "agent_setup_s": phases[1],
        "agent_exec_s": phases[2], "verifier_s": phases[3],
        "total_s": tot,
        "overhead_s": round(tot - sum(p for p in phases if p), 2) if tot and any(phases) else None,
        "env_type": env.get("type"), "verifier_env_mode": r.get("verifier_environment_mode"),
        "override_cpus": env.get("override_cpus"), "override_memory_mb": env.get("override_memory_mb"),
        "override_storage_mb": env.get("override_storage_mb"),
        "category": tm.get("category"), "difficulty": tm.get("difficulty"),
        "keywords": tm.get("keywords"),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs", nargs="+")
    p.add_argument("-o", "--out", default="aws/baseline_full.csv")
    a = p.parse_args()
    meta, rows, tests = load_meta(), [], []
    for job in a.jobs:
        job = job.rstrip("/")
        for d in sorted(glob.glob(os.path.join(job, "*__*"))):
            if os.path.exists(f"{d}/result.json"):
                rows.append(parse(d, os.path.basename(job), meta, tests))
    rows.sort(key=lambda r: (r["task"] or ""))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    tp = a.out.replace(".csv", "") + ".tests.csv"
    if tests:
        with open(tp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)
    n_err = sum(1 for r in rows if r["status"] == "ERROR")
    print(f"{a.out}: {len(rows)} trials, {len(FIELDS)} columns "
          f"({sum(1 for r in rows if r['status']=='PASS')} pass, "
          f"{sum(1 for r in rows if r['status']=='FAIL')} fail, {n_err} error)")
    if tests: print(f"{tp}: {len(tests)} individual test results")

if __name__ == "__main__":
    main()
