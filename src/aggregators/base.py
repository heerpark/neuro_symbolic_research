"""GradAggregator interface: the ONLY place that decides how L_neural and
L_sym combine into a backbone parameter update (per CLAUDE.md's update rule).
Phase 2 methods (PCGrad, CAGrad, FAMO, ConFIG, one-sided PCGrad) implement
this same interface without any change to train.py's call site:

    agg = build_aggregator(cfg["aggregator"])
    for batch in loader:
        optimizer.zero_grad()
        l_neural = compute_l_neural(batch)   # None when no digit labels are used
        l_sym = compute_l_sym(batch)
        diagnostics = agg.step(l_neural, l_sym, get_shared_backbone_params(model), optimizer)
        optimizer.step()

agg.step() must, as a side effect, populate .grad on every tensor in
shared_params (and on any other trainable params via ordinary autograd) so
that the immediately-following optimizer.step() performs the intended
combined update. It returns a flat dict of scalar diagnostics to log into
metrics.jsonl; conflict-aware subclasses may add keys (e.g. cos_sim) without
changing this contract.
"""
from abc import ABC, abstractmethod

import torch

_REGISTRY: dict = {}


def register(name: str, cls) -> None:
    _REGISTRY[name] = cls


class GradAggregator(ABC):
    @abstractmethod
    def step(
        self,
        l_neural: torch.Tensor | None,
        l_sym: torch.Tensor,
        shared_params: list,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        ...

    @staticmethod
    @abstractmethod
    def from_config(cfg: dict) -> "GradAggregator":
        ...


def build_aggregator(cfg: dict) -> GradAggregator:
    name = cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown aggregator '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name].from_config(cfg)
