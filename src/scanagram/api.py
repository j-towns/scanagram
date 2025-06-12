from jax import make_jaxpr
from jax import tree

from scanagram import core
from scanagram.api_util import check_types


def as_scan(f, example_xs):
    jaxpr, out_shapes = make_jaxpr(f, return_shape=True)(example_xs)
    body_fn_flat, init_carry = core.make_scan(jaxpr)
    def body_fn(carry, xs):
        check_types(carry, init_carry, "carry", "init_carry")
        check_types(xs, example_xs, "xs", "example_xs", True)
        carry, out_flat = body_fn_flat(carry, tree.leaves(xs))
        out = tree.unflatten(tree.structure(out_shapes), out_flat)
        check_types(out, out_shapes, "scan output", "example output", True)
        return carry, out
    return body_fn, init_carry
