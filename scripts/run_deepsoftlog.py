"""Runner for the official DeepSoftLog code (third_party/deepsoftlog), meant
to be invoked inside the dsl_env conda environment:

    conda run -n dsl_env python scripts/run_deepsoftlog.py \
        --config configs/deepsoftlog_n1.yaml --seed 0 --run-dir runs/deepsoftlog_n1/0

Never modifies the vendored source. Captures metrics by swapping in a
JsonlLogger with the same log()/log_eval()/print() interface as their
WandbLogger, so no wandb account/network is needed and results land directly
in our runs/<exp>/<seed>/metrics.jsonl format.
"""
import argparse
import json
import os
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSL_ROOT = os.path.join(REPO_ROOT, "third_party", "deepsoftlog")


class JsonlLogger:
    def __init__(self, config, metrics_path):
        self.config = config
        self.metrics_path = metrics_path

    def log(self, values):
        pass  # per-batch training diagnostics; not needed for Phase 0 reproduction

    def log_eval(self, values, name="test"):
        record = {"split": name, **{k: float(v) for k, v in values.items()}}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override cfg['nb_epochs'], e.g. for a pilot")
    parser.add_argument("--data-subset", type=int, default=None, help="override cfg['data_subset'], e.g. for a pilot")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["nb_epochs"] = args.epochs
    if args.data_subset is not None:
        cfg["data_subset"] = args.data_subset
    cfg["name"] = f"{cfg['name']}_seed{args.seed}"

    sys.path.insert(0, DSL_ROOT)
    os.chdir(DSL_ROOT)  # their config's `program` path and results/<name>/ save path are relative to this root

    from deepsoftlog.experiments.mnist_addition import addition as dsl_addition

    dsl_addition.WandbLogger = lambda config: JsonlLogger(config, metrics_path)

    t0 = time.time()
    dsl_addition.train(cfg)
    elapsed = time.time() - t0

    with open(metrics_path, "a") as f:
        f.write(json.dumps({"split": "meta", "total_train_time_sec": elapsed}) + "\n")
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"[deepsoftlog] done in {elapsed:.1f}s, metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
