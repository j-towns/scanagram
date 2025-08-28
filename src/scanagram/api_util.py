import jax.numpy as jnp
from jax import tree

from scanagram.util import all_equal

def check_types(xt, yt, xt_name, yt_name, scan_input=False):
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

def array_len(x):
    return x.shape[0]

def check_lengths(example_xs):
    if not all_equal(map(array_len, tree.leaves(example_xs))):
        raise ValueError(
            "All example arrays must have 0th axes of equal length"
        )

def check_prefill(example_xs, prefills):
    if tree.structure(example_xs) != tree.structure(prefills):
        raise ValueError(
            "Prefill PyTree structure must match example_xs"
        )
    if any(
        x.shape[1:] != p.shape[1:]
        for x, p in zip(tree.leaves(example_xs), tree.leaves(prefills))
    ):
        raise ValueError(
            "Shape of example_xs must match shape of prefills in all axes "
            "except the 0th."
        )
    if not all_equal(map(array_len, tree.leaves(prefills))):
        raise ValueError(
            "All prefills must have 0th axes of equal length"
        )

    if any(
        array_len(p) > array_len(x) for p, x in
        zip(tree.leaves(prefills), tree.leaves(example_xs))
    ):
        raise ValueError(
            "Prefill must have length less than or equal to example_xs."
        )
