from jax import make_jaxpr
from jax import tree
from jax import numpy as jnp

from scanagram import core


def _check_types(xt, yt, xt_name, yt_name, scan_input=False):
    get_dtype = (
        lambda x: x.dtype if hasattr(x, "dtype") else jnp.asarray(x).dtype
    )

    if tree.structure(xt) != tree.structure(yt):
        raise ValueError(
            f"{xt_name} has different pytree structure to {yt_name}."
        )
    if any(
        get_dtype(x) != get_dtype(y)
        for x, y in zip(tree.leaves(xt), tree.leaves(yt))
    ):
        raise ValueError(
            f"{xt_name} contains one or more arrays with different dtypes "
            f"to {yt_name}."
        )
    if scan_input:
        if any(
            jnp.shape(x) != jnp.shape(y)[1:]
            for x, y in zip(tree.leaves(xt), tree.leaves(yt))
        ):
            raise ValueError(
                f"Arrays in {xt_name} must have shapes equal to those in "
                f"{yt_name} with the 0'th axis removed."
            )
    elif any(
        jnp.shape(x) != jnp.shape(y)
        for x, y in zip(tree.leaves(xt), tree.leaves(yt))
    ):
        raise ValueError(
            f"{xt_name} contains one or more arrays with different shapes "
            f"to {yt_name}."
        )


def as_scan(f, example_xs):
    xs_structure = tree.structure(((example_xs,), {}))
    jaxpr, out_shapes = make_jaxpr(f, return_shape=True)(example_xs)
    body_fn_flat, init_carry = core.make_scan(jaxpr)
    def body_fn(carry, xs):
        _check_types(carry, init_carry, "carry", "init_carry")
        _check_types(xs, example_xs, "xs", "example_xs", True)
        carry, out_flat = body_fn_flat(carry, tree.leaves(((xs,), {})))
        out = tree.unflatten(tree.structure(out_shapes), out_flat)
        _check_types(out, out_shapes, "scan output", "example output", True)
        return carry, out
    return body_fn, init_carry
