import pytest
import torch

from src.aggregators import SumAggregator, build_aggregator


def test_build_aggregator_sum_default_lam():
    agg = build_aggregator({"name": "sum"})
    assert isinstance(agg, SumAggregator)
    assert agg.lam == 1.0


def test_build_aggregator_sum_custom_lam():
    agg = build_aggregator({"name": "sum", "lam": 0.5})
    assert agg.lam == 0.5


def test_build_aggregator_unknown_name_raises():
    with pytest.raises(ValueError):
        build_aggregator({"name": "does-not-exist"})


def test_sum_aggregator_pure_symbolic_grad_when_no_neural_loss():
    w = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([w], lr=0.1)
    l_sym = w**2  # d/dw = 2w = 4.0

    agg = SumAggregator(lam=0.5)
    diagnostics = agg.step(None, l_sym, [w], optimizer)

    assert torch.isclose(w.grad, torch.tensor(0.5 * 4.0))
    assert diagnostics["l_neural"] != diagnostics["l_neural"]  # NaN


def test_sum_aggregator_combines_both_losses():
    w = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD([w], lr=0.1)
    l_neural = w * 2  # d/dw = 2
    l_sym = w**2  # d/dw = 2w = 6

    agg = SumAggregator(lam=1.0)
    agg.step(l_neural, l_sym, [w], optimizer)

    assert torch.isclose(w.grad, torch.tensor(2.0 + 6.0))
