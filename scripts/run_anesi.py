"""Runner for the official A-NeSI code (third_party/anesi), meant to be
invoked inside the anesi_env conda environment:

    conda run -n anesi_env python scripts/run_anesi.py \
        --config configs/anesi_n1.yaml --seed 0 --run-dir runs/anesi_n1/0

mnistop.py is a __main__-only script (not an importable train() function)
that always calls wandb.init()/wandb.log() and has no seeding hook of its
own. We run it via subprocess with WANDB_MODE=disabled (no network/account
needed) through a small shim that seeds torch/numpy/random before the
script's own code runs -- data order is otherwise fully deterministic in
this codebase (shuffle=False, no seed passed to the dataset), so this is
what "seed" means for this benchmark. Metrics are captured by parsing the
script's own per-epoch stdout print (not wandb.log: wandb.init() rebinds
wandb.log to its own internal implementation, so a pre-init monkeypatch of
wandb.log gets silently discarded -- confirmed empirically), matching the
exact format printed by mnistop.py's training loop:
    "{Test|Val} accuracy: <f> {Test|Val} Explain: <f> {Test|Val} Digit: <f> Epoch time: <f> {Test|Val} time: <f>"
Never modifies the vendored source except for one documented bugfix patch
(third_party/patches/anesi/0001-fix-train_all-TrainResult-unpacking.patch):
the pinned commit's mnistop.py unpacks model.train_all()'s return value as a
2-tuple, but it has returned a TrainResult dataclass (non-iterable) since
the "Version 2" commit (4366c29, 2023-05-16) -- verified this breaks the
official code out of the box even with its own shipped defaults, at both
that commit and current HEAD, so this is an upstream regression that was
never fixed in the MNIST-addition experiment script, not something specific
to our reproduction.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANESI_ROOT = os.path.join(REPO_ROOT, "third_party", "anesi", "anesi")
MNISTOP_PATH = os.path.join(ANESI_ROOT, "experiments", "mnist_op", "mnistop.py")

_EPOCH_LINE_RE = re.compile(
    r"(Test|Val) accuracy: (\S+) \1 Explain: (\S+) \1 Digit: (\S+) Epoch time: (\S+) \1 time: (\S+)"
)

_SHIM = """
import os
import numpy as np
import torch
import random

_seed = int(os.environ["ANESI_SEED"])
random.seed(_seed)
np.random.seed(_seed)
torch.manual_seed(_seed)

import sys
sys.argv = ["mnistop.py", "--config", os.environ["ANESI_CONFIG_PATH"]]
_mnistop_path = os.environ["ANESI_MNISTOP_PATH"]
with open(_mnistop_path) as f:
    _src = f.read()
exec(compile(_src, _mnistop_path, "exec"), {"__name__": "__main__", "__file__": _mnistop_path})
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override cfg['epochs'], e.g. for a pilot")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    resolved_config_path = os.path.join(run_dir, "resolved_config.yaml")
    with open(resolved_config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    shim_path = os.path.join(run_dir, "_shim.py")
    with open(shim_path, "w") as f:
        f.write(_SHIM)

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = ANESI_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["ANESI_CONFIG_PATH"] = resolved_config_path
    env["ANESI_MNISTOP_PATH"] = MNISTOP_PATH
    env["ANESI_SEED"] = str(args.seed)
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    epoch = 0
    proc = subprocess.Popen(
        [sys.executable, shim_path],
        cwd=os.path.join(ANESI_ROOT, "experiments", "mnist_op"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with open(metrics_path, "a") as mf:
        for line in proc.stdout:
            print(line, end="")
            m = _EPOCH_LINE_RE.search(line)
            if m:
                prefix, acc, explain, digit, epoch_time, eval_time = m.groups()
                record = {
                    "epoch": epoch,
                    "split": prefix.lower(),
                    "accuracy": float(acc),
                    "explain_accuracy": float(explain),
                    "digit_accuracy": float(digit),
                    "epoch_time_sec": float(epoch_time),
                    "eval_time_sec": float(eval_time),
                }
                mf.write(json.dumps(record) + "\n")
                mf.flush()
                epoch += 1
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    with open(metrics_path, "a") as f:
        f.write(json.dumps({"split": "meta", "total_train_time_sec": elapsed}) + "\n")
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        cfg["seed"] = args.seed
        yaml.safe_dump(cfg, f)

    print(f"[anesi] done in {elapsed:.1f}s, metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
