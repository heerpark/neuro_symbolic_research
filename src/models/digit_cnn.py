"""LeNet-style CNN backbone for MNIST digit classification.

The same softmax output serves both as the CE classification head (via
.logits()) and as the LTN Digit(x, d) predicate truth value (via .forward()),
matching the official LTN reference implementation's design: there is no
separate head splitting neuro and symbolic supervision. shared_backbone_params()
therefore returns every parameter in the model; it exists as a single named
accessor (rather than inlining .parameters() everywhere) so that if a future
phase introduces a genuinely separate head, only this method needs to change.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 10)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        h = F.max_pool2d(F.elu(self.conv1(x)), 2)
        h = F.max_pool2d(F.elu(self.conv2(h)), 2)
        h = h.flatten(1)
        h = F.elu(self.fc1(h))
        h = F.elu(self.fc2(h))
        return self.fc3(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(x), dim=-1)

    def shared_backbone_params(self) -> list:
        return list(self.parameters())


class DigitPredicateModel(nn.Module):
    """Wraps a DigitCNN as the two-argument model LTNtorch's Predicate expects
    for Digit(x, d): x and d arrive already broadcast/flattened to a shared
    leading dimension K by LTNtorch (verified empirically), so this just runs
    the CNN once per image and gathers the truth value at class index d.
    """

    def __init__(self, cnn: DigitCNN):
        super().__init__()
        self.cnn = cnn

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        probs = self.cnn(x)  # [K, 10]
        d_idx = d.reshape(-1, 1).long()
        return torch.gather(probs, dim=-1, index=d_idx).squeeze(-1)  # [K]
