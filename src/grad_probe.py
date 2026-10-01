"""Gradient-conflict probing (Phase 1+). Phase 0 only needs a stable accessor
for "shared backbone params" so train.py's loop shape doesn't change later:
grad_probe.probe(...) will be added in Phase 1 and called from the same
call site train.py already reserves via the `probe_gradients` config flag.
"""


def get_shared_backbone_params(model) -> list:
    return model.shared_backbone_params()
