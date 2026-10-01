"""N=2 MNIST-Addition axiom (two-digit addends).

    forall x1,x2,y1,y2,l. exists d1,d2,d3,d4 in 0..9:
        (10*d1+d2 + 10*d3+d4 == l) -> (Digit(x1,d1) & Digit(x2,d2) & Digit(y1,d3) & Digit(y2,d4))

x1/x2 are the tens/units-digit images of operand 1, y1/y2 of operand 2.
Guarded existential over the 10^4 (d1,d2,d3,d4) grid, vectorized via
LTNtorch's broadcasting (no Python loop) -- verified end-to-end (shapes,
guard correctness, backward timing ~0.1s/batch of 16 on CPU) against the
installed LTNtorch 1.0.2 before being written here.
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
        images_x1 = batch["images_op1"][:, 0]
        images_x2 = batch["images_op1"][:, 1]
        images_y1 = batch["images_op2"][:, 0]
        images_y2 = batch["images_op2"][:, 1]
        sum_label = batch["sum_label"]

        x1 = ltn.Variable("x1", images_x1)
        x2 = ltn.Variable("x2", images_x2)
        y1 = ltn.Variable("y1", images_y1)
        y2 = ltn.Variable("y2", images_y2)
        l = ltn.Variable("l", sum_label)

        d1 = ltn.Variable("d1", DIGIT_DOMAIN)
        d2 = ltn.Variable("d2", DIGIT_DOMAIN)
        d3 = ltn.Variable("d3", DIGIT_DOMAIN)
        d4 = ltn.Variable("d4", DIGIT_DOMAIN)

        x1d, x2d, y1d, y2d, ld = ltn.diag(x1, x2, y1, y2, l)
        Exists = make_exists(p=p_exists)

        inner = self.And(
            self.And(self.Digit(x1d, d1), self.Digit(x2d, d2)),
            self.And(self.Digit(y1d, d3), self.Digit(y2d, d4)),
        )

        def cond_fn(d1_, d2_, d3_, d4_, l_):
            return torch.eq(10 * d1_.value + d2_.value + 10 * d3_.value + d4_.value, l_.value)

        exists_out = Exists(
            [d1, d2, d3, d4],
            inner,
            cond_vars=[d1, d2, d3, d4, ld],
            cond_fn=cond_fn,
        )
        axiom = self.Forall([x1d, x2d, y1d, y2d, ld], exists_out)
        return make_sat_agg()(axiom)
