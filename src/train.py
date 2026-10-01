"""CLI entrypoint: python -m src.train --config configs/<name>.yaml --seed 0

Standard setting (Phase 0): no digit-label supervision, l_neural is always
None, only l_sym (1 - SatAgg(axioms)) drives the update via the configured
GradAggregator (default: sum).
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.aggregators import build_aggregator
from src.data.mnist_addition import build_splits
from src.grad_probe import get_shared_backbone_params
from src.kb._common import make_digit_predicate
from src.models.digit_cnn import DigitCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_kb_module(n_digits: int):
    if n_digits == 1:
        from src.kb import axioms_n1 as kb_module
    elif n_digits == 2:
        from src.kb import axioms_n2 as kb_module
    else:
        raise ValueError(f"Unsupported n_digits={n_digits}")
    return kb_module


def p_exists_for_epoch(p_schedule: dict, epoch: int) -> int:
    boundaries = sorted(p_schedule.keys())
    current = p_schedule[boundaries[0]]
    for b in boundaries:
        if epoch >= b:
            current = p_schedule[b]
    return current


def compute_sum_accuracy(model, loader, n_digits: int, device) -> float:
    model.eval()
    weights = 10 ** torch.arange(n_digits - 1, -1, -1, device=device)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            images_op1 = batch["images_op1"].to(device)
            images_op2 = batch["images_op2"].to(device)
            sum_label = batch["sum_label"].to(device)

            pred_op1 = torch.stack(
                [model(images_op1[:, i]).argmax(dim=-1) for i in range(n_digits)], dim=1
            )
            pred_op2 = torch.stack(
                [model(images_op2[:, i]).argmax(dim=-1) for i in range(n_digits)], dim=1
            )
            pred_sum = (pred_op1 * weights).sum(dim=1) + (pred_op2 * weights).sum(dim=1)
            correct += (pred_sum == sum_label).sum().item()
            total += sum_label.shape[0]
    model.train()
    return correct / total


def compute_digit_accuracy(model, loader, n_digits: int, device) -> float:
    """Diagnostic only (never used for training or model selection in the
    standard setting): per-digit accuracy inferred from the trained predicate."""
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            for images_key, labels_key in (
                ("images_op1", "digit_labels_op1"),
                ("images_op2", "digit_labels_op2"),
            ):
                images = batch[images_key].to(device)
                labels = batch[labels_key].to(device)
                for i in range(n_digits):
                    pred = model(images[:, i]).argmax(dim=-1)
                    correct += (pred == labels[:, i]).sum().item()
                    total += pred.shape[0]
    return correct / total


def train_one_epoch(model, kb, aggregator, optimizer, loader, p_exists, probe_gradients=False) -> dict:
    model.train()
    running_l_sym, n_batches = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        l_sym = 1 - kb.sat(batch, p_exists=p_exists)
        if probe_gradients:
            from src import grad_probe

            grad_probe.probe(None, l_sym, get_shared_backbone_params(model), retain_graph=True)
        diagnostics = aggregator.step(None, l_sym, get_shared_backbone_params(model), optimizer)
        optimizer.step()
        running_l_sym += diagnostics["l_sym"]
        n_batches += 1
    return {"train_l_sym": running_l_sym / max(n_batches, 1)}


def run_training(cfg: dict, seed: int, run_dir: str, config_path: str) -> None:
    set_seed(seed)
    device = torch.device("cpu")

    n_digits = cfg["n_digits"]
    splits = build_splits(
        n_digits, seed, root=cfg.get("data_root", "./data/mnist_raw"), val_frac=cfg.get("val_frac", 0.10)
    )
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(splits["train"], batch_size=cfg["batch_size"], shuffle=True, generator=loader_gen)
    val_loader = DataLoader(splits["val"], batch_size=cfg.get("eval_batch_size", 256), shuffle=False)
    test_loader = DataLoader(splits["test"], batch_size=cfg.get("eval_batch_size", 256), shuffle=False)

    model = DigitCNN().to(device)
    digit_predicate = make_digit_predicate(model)
    kb = load_kb_module(n_digits).MnistAdditionKB(digit_predicate, p_forall=cfg.get("p_forall", 2))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    aggregator = build_aggregator(cfg["aggregator"])
    p_schedule = cfg.get("p_schedule", {0: 2})
    probe_gradients = cfg.get("probe_gradients", False)

    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    for epoch in range(cfg["epochs"]):
        p_exists = p_exists_for_epoch(p_schedule, epoch)
        t0 = time.time()
        epoch_stats = train_one_epoch(model, kb, aggregator, optimizer, train_loader, p_exists, probe_gradients)
        epoch_time = time.time() - t0
        val_acc = compute_sum_accuracy(model, val_loader, n_digits, device)

        record = {
            "epoch": epoch,
            "p_exists": p_exists,
            "train_l_sym": epoch_stats["train_l_sym"],
            "val_sum_accuracy": val_acc,
            "epoch_time_sec": epoch_time,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[epoch {epoch}] p_exists={p_exists} l_sym={record['train_l_sym']:.4f} "
            f"val_sum_acc={val_acc:.4f} time={epoch_time:.1f}s"
        )

    test_sum_acc = compute_sum_accuracy(model, test_loader, n_digits, device)
    test_digit_acc = compute_digit_accuracy(model, test_loader, n_digits, device)
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"epoch": "final", "test_sum_accuracy": test_sum_acc, "test_digit_accuracy": test_digit_acc}) + "\n")
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    print(f"[final] test_sum_accuracy={test_sum_acc:.4f} test_digit_accuracy={test_digit_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override cfg['epochs'], e.g. for a pilot run")
    parser.add_argument("--run-dir", default=None, help="override the default runs/<exp_name>/<seed> output dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    exp_name = cfg.get("exp_name", os.path.splitext(os.path.basename(args.config))[0])
    if args.run_dir is not None:
        run_training(cfg, args.seed, args.run_dir, args.config)
        return
    run_dir = os.path.join("runs", exp_name, str(args.seed))
    run_training(cfg, args.seed, run_dir, args.config)


if __name__ == "__main__":
    main()
