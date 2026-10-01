"""Default GradAggregator: L = L_neural + lam * L_sym (or pure lam*L_sym when
no digit labels are supervised, e.g. Phase 0's standard setting)."""
import torch

from src.aggregators.base import GradAggregator, register


class SumAggregator(GradAggregator):
    def __init__(self, lam: float = 1.0):
        self.lam = lam

    @staticmethod
    def from_config(cfg: dict) -> "SumAggregator":
        return SumAggregator(lam=cfg.get("lam", 1.0))

    def step(self, l_neural, l_sym, shared_params, optimizer) -> dict:
        if l_neural is not None:
            total = l_neural + self.lam * l_sym
        else:
            total = self.lam * l_sym
        total.backward()
        return {
            "l_neural": float(l_neural.detach()) if l_neural is not None else float("nan"),
            "l_sym": float(l_sym.detach()),
            "l_total": float(total.detach()),
        }


register("sum", SumAggregator)
