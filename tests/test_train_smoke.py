import torch
from torch.utils.data import DataLoader, Dataset

from src.aggregators import build_aggregator
from src.kb import axioms_n1
from src.kb._common import make_digit_predicate
from src.models.digit_cnn import DigitCNN
from src.train import train_one_epoch


class SyntheticN1Dataset(Dataset):
    def __init__(self, n=12):
        g = torch.Generator().manual_seed(0)
        self.images_op1 = torch.rand(n, 1, 1, 28, 28, generator=g)
        self.images_op2 = torch.rand(n, 1, 1, 28, 28, generator=g)
        self.digit_labels_op1 = torch.randint(0, 10, (n, 1), generator=g)
        self.digit_labels_op2 = torch.randint(0, 10, (n, 1), generator=g)
        self.sum_label = self.digit_labels_op1[:, 0] + self.digit_labels_op2[:, 0]

    def __len__(self):
        return self.images_op1.shape[0]

    def __getitem__(self, i):
        return {
            "images_op1": self.images_op1[i],
            "images_op2": self.images_op2[i],
            "digit_labels_op1": self.digit_labels_op1[i],
            "digit_labels_op2": self.digit_labels_op2[i],
            "sum_label": self.sum_label[i],
        }


def test_train_one_epoch_three_steps_no_exceptions_and_params_change():
    loader = DataLoader(SyntheticN1Dataset(n=12), batch_size=4, shuffle=False)

    model = DigitCNN()
    kb = axioms_n1.MnistAdditionKB(make_digit_predicate(model), p_forall=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    aggregator = build_aggregator({"name": "sum", "lam": 1.0})

    params_before = [p.clone() for p in model.parameters()]
    stats = train_one_epoch(model, kb, aggregator, optimizer, loader, p_exists=2)

    assert torch.isfinite(torch.tensor(stats["train_l_sym"]))
    changed = any(not torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    assert changed
