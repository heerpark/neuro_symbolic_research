from src.grad_probe import get_shared_backbone_params
from src.models.digit_cnn import DigitCNN


def test_shared_backbone_is_the_full_model():
    """Documents/enforces the design decision that neuro and symbolic losses
    share the entire network (no separate head), see digit_cnn.py."""
    model = DigitCNN()
    shared = get_shared_backbone_params(model)
    all_params = list(model.parameters())
    assert len(shared) == len(all_params)
    assert all(a is b for a, b in zip(shared, all_params))
