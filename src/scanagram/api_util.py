import jax.numpy as jnp
from jax import tree

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
