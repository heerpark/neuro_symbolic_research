"""Shared LTNtorch object construction for the MNIST-Addition KBs.

API usage verified against the installed LTNtorch 1.0.2 source
(ltn/core.py, ltn/fuzzy_ops.py) and a live smoke test, not guessed:
- ltn.Quantifier(agg_op, quantifier).__call__(vars, formula, cond_vars=, cond_fn=)
  implements guarded quantification.
- ltn.fuzzy_ops.SatAgg(agg_op) aggregates one or more closed (scalar) axioms.
- The official TF LTN reference example (logictensornetworks/logictensornetworks,
  examples/mnist/*_addition_ltn.py) anneals Exists's p (1->2->4->6) while Forall's
  p stays fixed at 2 -- that split is reproduced here.
"""
import torch
import ltn

from src.models.digit_cnn import DigitPredicateModel

DIGIT_DOMAIN = torch.arange(10)


def make_digit_predicate(cnn) -> ltn.Predicate:
    return ltn.Predicate(model=DigitPredicateModel(cnn))


def make_and() -> ltn.Connective:
    return ltn.Connective(ltn.fuzzy_ops.AndProd(stable=True))


def make_forall(p: int = 2) -> ltn.Quantifier:
    return ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p, stable=True), quantifier="f")


def make_exists(p: int) -> ltn.Quantifier:
    return ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=p, stable=True), quantifier="e")


def make_sat_agg() -> ltn.fuzzy_ops.SatAgg:
    return ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True))
