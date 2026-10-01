"""Dispatch Phase 0 baseline-reproduction runs across LTN (ours), DeepSoftLog,
and A-NeSI, each in its own conda env, with bounded parallelism.

Usage:
  python scripts/sweep.py --parallel 4
  python scripts/sweep.py --only ltn deepsoftlog --seeds 0 1 2 --conditions 1 2
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_MAP = {"ltn": "neuro_symbolic", "deepsoftlog": "dsl_env", "anesi": "anesi_env"}
SEEDS = [0, 1, 2]
CONDITIONS = [1, 2]  # N values


def ltn_job(n, seed):
    run_dir = f"runs/ltn_standard_n{n}/{seed}"
    cmd = [
        "conda", "run", "-n", ENV_MAP["ltn"], "python", "-m", "src.train",
        "--config", f"configs/ltn_standard_n{n}.yaml", "--seed", str(seed), "--run-dir", run_dir,
    ]
    return cmd, run_dir


def deepsoftlog_job(n, seed):
    run_dir = f"runs/deepsoftlog_n{n}/{seed}"
    cmd = [
        "conda", "run", "-n", ENV_MAP["deepsoftlog"], "python", "scripts/run_deepsoftlog.py",
        "--config", f"configs/deepsoftlog_n{n}.yaml", "--seed", str(seed), "--run-dir", run_dir,
    ]
    return cmd, run_dir


def anesi_job(n, seed):
    run_dir = f"runs/anesi_n{n}/{seed}"
    cmd = [
        "conda", "run", "-n", ENV_MAP["anesi"], "python", "scripts/run_anesi.py",
        "--config", f"configs/anesi_n{n}.yaml", "--seed", str(seed), "--run-dir", run_dir,
    ]
    return cmd, run_dir


JOB_BUILDERS = {"ltn": ltn_job, "deepsoftlog": deepsoftlog_job, "anesi": anesi_job}


def run_job(name, cmd, run_dir):
    abs_run_dir = os.path.join(REPO_ROOT, run_dir)
    os.makedirs(abs_run_dir, exist_ok=True)
    log_path = os.path.join(abs_run_dir, "stdout.log")
    print(f"[sweep] starting {name}")
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"
    print(f"[sweep] finished {name}: {status} (log: {log_path})")
    return name, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(JOB_BUILDERS), choices=list(JOB_BUILDERS))
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--conditions", nargs="*", type=int, default=CONDITIONS)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()

    jobs = [
        (f"{method}_n{n}_seed{seed}", *JOB_BUILDERS[method](n, seed))
        for method in args.only
        for n in args.conditions
        for seed in args.seeds
    ]

    print(f"[sweep] {len(jobs)} jobs, parallelism={args.parallel}")
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(run_job, name, cmd, run_dir) for name, cmd, run_dir in jobs]
        results = [fut.result() for fut in futures]

    failed = [name for name, rc in results if rc != 0]
    print(f"[sweep] done. {len(results) - len(failed)}/{len(results)} succeeded.")
    if failed:
        print(f"[sweep] FAILED: {failed}")


if __name__ == "__main__":
    main()
