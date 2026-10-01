import torch

from src.kb import axioms_n1, axioms_n2
from src.kb._common import make_digit_predicate
from src.models.digit_cnn import DigitCNN


def make_batch_n1(batch_size=4):
    return {
        "images_op1": torch.rand(batch_size, 1, 1, 28, 28),
        "images_op2": torch.rand(batch_size, 1, 1, 28, 28),
        "sum_label": torch.tensor([0, 9, 18, 5])[:batch_size],
    }


def make_batch_n2(batch_size=2):
    return {
        "images_op1": torch.rand(batch_size, 2, 1, 28, 28),
        "images_op2": torch.rand(batch_size, 2, 1, 28, 28),
        "sum_label": torch.tensor([0, 198])[:batch_size],
    }


def test_n1_guard_matches_hand_computation():
    d1 = torch.arange(10).view(10, 1)
    d2 = torch.arange(10).view(1, 10)
    for label, expected_count in [(0, 1), (9, 10), (18, 1)]:
        mask = torch.eq(d1 + d2, label)
        assert mask.sum().item() == expected_count


def test_n2_guard_matches_hand_computation():
    # Only (0,0,0,0) sums to 0; only (9,9,9,9) sums to 198 (both operands maxed).
    combos_for_0 = sum(
        1
        for d1 in range(10)
        for d2 in range(10)
        for d3 in range(10)
        for d4 in range(10)
        if 10 * d1 + d2 + 10 * d3 + d4 == 0
    )
    combos_for_198 = sum(
        1
        for d1 in range(10)
        for d2 in range(10)
        for d3 in range(10)
        for d4 in range(10)
        if 10 * d1 + d2 + 10 * d3 + d4 == 198
    )
    assert combos_for_0 == 1
    assert combos_for_198 == 1


def test_n1_sat_is_scalar_probability_and_differentiable():
    cnn = DigitCNN()
    kb = axioms_n1.MnistAdditionKB(make_digit_predicate(cnn), p_forall=2)
    batch = make_batch_n1()

    sat = kb.sat(batch, p_exists=2)
    assert sat.shape == torch.Size([])
    assert torch.isfinite(sat)
    assert 0.0 <= sat.item() <= 1.0 + 1e-6

    (1 - sat).backward()
    assert cnn.fc3.weight.grad is not None
    assert torch.isfinite(cnn.fc3.weight.grad).all()


def test_n2_sat_is_scalar_probability_and_differentiable():
    cnn = DigitCNN()
    kb = axioms_n2.MnistAdditionKB(make_digit_predicate(cnn), p_forall=2)
    batch = make_batch_n2()

    sat = kb.sat(batch, p_exists=2)
    assert sat.shape == torch.Size([])
    assert torch.isfinite(sat)
    assert 0.0 <= sat.item() <= 1.0 + 1e-6

    (1 - sat).backward()
    assert cnn.fc3.weight.grad is not None
    assert torch.isfinite(cnn.fc3.weight.grad).all()
