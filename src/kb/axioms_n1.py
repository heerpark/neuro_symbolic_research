"""N=1 MNIST-Addition axiom (single-digit addends).

    forall x1,x2,l. exists d1,d2 in 0..9:
        (d1 + d2 == l) -> (Digit(x1,d1) & Digit(x2,d2))

Guarded existential over the 10x10 (d1,d2) grid, guard = elementwise
d1+d2==l via LTNtorch's cond_vars/cond_fn mechanism. Verified end-to-end
(shapes, guard correctness, backward pass) against the installed LTNtorch
1.0.2 before being written here.
"""
import torch
import ltn

from src.kb._common import make_and, make_forall, make_exists, make_sat_agg, DIGIT_DOMAIN


class MnistAdditionKB:
    def __init__(self, digit_predicate: ltn.Predicate, p_forall: int = 2):
        self.Digit = digit_predicate
        self.And = make_and()
        self.Forall = make_forall(p=p_forall)

    def sat(self, batch: dict, p_exists: int) -> torch.Tensor:
        images_op1 = batch["images_op1"][:, 0]
        images_op2 = batch["images_op2"][:, 0]
        sum_label = batch["sum_label"]

        x1 = ltn.Variable("x1", images_op1)
        x2 = ltn.Variable("x2", images_op2)
        l = ltn.Variable("l", sum_label)
        d1 = ltn.Variable("d1", DIGIT_DOMAIN)
        d2 = ltn.Variable("d2", DIGIT_DOMAIN)

        x1d, x2d, ld = ltn.diag(x1, x2, l)
        Exists = make_exists(p=p_exists)

        inner = self.And(self.Digit(x1d, d1), self.Digit(x2d, d2))
        exists_out = Exists(
            [d1, d2],
            inner,
            cond_vars=[d1, d2, ld],
            cond_fn=lambda d1_, d2_, l_: torch.eq(d1_.value + d2_.value, l_.value),
        )
        axiom = self.Forall([x1d, x2d, ld], exists_out)
        return make_sat_agg()(axiom)
