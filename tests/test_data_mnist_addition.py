import torch

from src.data.mnist_addition import build_splits


def test_split_sizes_n1():
    splits = build_splits(1, seed=0)
    n_train_total = 60000 // 2
    n_val = round(n_train_total * 0.10)
    assert len(splits["train"]) == n_train_total - n_val
    assert len(splits["val"]) == n_val
    assert len(splits["test"]) == 10000 // 2


def test_split_sizes_n2():
    splits = build_splits(2, seed=0)
    n_train_total = 60000 // 4
    n_val = round(n_train_total * 0.10)
    assert len(splits["train"]) == n_train_total - n_val
    assert len(splits["val"]) == n_val
    assert len(splits["test"]) == 10000 // 4


def test_determinism_same_seed():
    a = build_splits(1, seed=42)
    b = build_splits(1, seed=42)
    assert torch.equal(a["train"].sum_label, b["train"].sum_label)
    assert torch.equal(a["train"].images_op1, b["train"].images_op1)
    assert torch.equal(a["val"].sum_label, b["val"].sum_label)


def test_different_seeds_differ():
    a = build_splits(1, seed=1)
    b = build_splits(1, seed=2)
    assert not torch.equal(a["train"].sum_label, b["train"].sum_label)


def test_n1_sum_label_correctness():
    splits = build_splits(1, seed=0)
    ds = splits["train"]
    item = ds[0]
    expected = item["digit_labels_op1"][0] + item["digit_labels_op2"][0]
    assert item["sum_label"] == expected


def test_n2_sum_label_correctness():
    splits = build_splits(2, seed=0)
    ds = splits["train"]
    item = ds[0]
    d1a, d1b = item["digit_labels_op1"]
    d2a, d2b = item["digit_labels_op2"]
    expected = (10 * d1a + d1b) + (10 * d2a + d2b)
    assert item["sum_label"] == expected
