from jax import make_jaxpr
from jax import tree

from scanagram import core
from scanagram.api_util import check_types, check_lengths, check_prefill


def as_scan(fun, example_xs):
    """Infers a body function and init carry for scan-like ``fun``.

    Args:
      fun: Scan-like input function.

        This function should accept one argument, and satisfy the following
        (causal) property: forall inputs xs, and forall t,

          jnp.all(fun(xs)[:t] == fun(xs[:t]))                             (*)

        This is a mathematically compact way of saying that within ``fun``,
        information cannot flow backwards along the 0'th input/output axis.
        Note that within ``fun`` the position of this axis can be moved using
        transpose/moveaxis, as long as causality along that special axis is
        maintained. We require that each lax primitive call within ``fun`` is
        causal. Pytree inputs and outputs are allowed, in which case condition
        (*) applies along the 0'th axis of all arrays in the input and output.
        For more detail, including how to handle self-attention, see README.md.
      example_xs: Example input to ``fun``.

        A (pytree of) JAX Array(s) with the same shape(s) and dtype(s) as an
        input to ``fun``. To avoid initializing redundant arrays, use any
        Python class that has ``shape`` and ``dtype`` attributes in place of
        arrays, such as ``jax.ShapeDtypeStruct``.

    Returns:
      A pair (body_fun, carry_init) which, when scanned, compute the same value
      as ``fun``. In particular, forall xs

        jnp.all(fun(xs) == lax.scan(body_fun, carry_init, xs)[1])

    Examples:
      Here is an example using ``scanagram.as_scan`` to convert a causal
      convolution to a scan, and checking correctness using ``jnp.allclose``:

      >>> import jax
      >>> from jax import lax
      >>> from jax import random
      >>> import jax.numpy as jnp
      >>>
      >>> import scanagram
      >>>
      >>> n, t, t_window, c_in, c_out = 2, 3, 5, 7, 11
      >>> rng_kernel, rng_xs = random.split(random.PRNGKey(0))
      >>>
      >>> kernel = random.normal(rng_kernel, (t_window, c_in, c_out))
      >>>
      >>> def causal_conv(xs):
      ...     return lax.conv_general_dilated(
      ...         xs, kernel,
      ...         window_strides=[1],
      ...         padding=[[t_window - 1, 0]],  # <- Causal padding
      ...         dimension_numbers=("TNC", "TIO", "TNC")
      ... )
      ...
      >>> example_xs = jax.ShapeDtypeStruct((t, n, c_in), "float32")
      >>> body_fun, carry_init = scanagram.as_scan(causal_conv, example_xs)
      >>>
      >>> xs = random.normal(rng_kernel, (t, n, c_in))
      >>> assert jnp.allclose(
      ...     causal_conv(xs), lax.scan(body_fun, carry_init, xs)[1]
      ... )
      ...
    """
    check_lengths(example_xs)
    jaxpr, out_shapes = make_jaxpr(fun, return_shape=True)(example_xs)
    body_fun_flat, carry_init, _ = core.make_scan(jaxpr)
    def body_fun(carry, xs):
        check_types(carry, carry_init, "carry", "carry_init")
        check_types(xs, example_xs, "xs", "example_xs", True)
        carry, out_flat = body_fun_flat(carry, tree.leaves(xs))
        out = tree.unflatten(tree.structure(out_shapes), out_flat)
        check_types(out, out_shapes, "scan output", "example output", True)
        return carry, out
    return body_fun, carry_init

def as_scan_with_prefill(fun, example_xs, in_prefills):
    check_lengths(example_xs)
    check_prefill(example_xs, in_prefills)
    jaxpr, out_shapes = make_jaxpr(fun, return_shape=True)(example_xs)
    body_fun_flat, carry_init, out_prefill = core.make_scan(
        jaxpr, tree.leaves(in_prefills)
    )
    out_prefill = tree.unflatten(tree.structure(out_shapes), out_prefill)
    def body_fun(carry, xs):
        check_types(carry, carry_init, "carry", "carry_init")
        check_types(xs, example_xs, "xs", "example_xs", True)
        carry, out_flat = body_fun_flat(carry, tree.leaves(xs))
        out = tree.unflatten(tree.structure(out_shapes), out_flat)
        check_types(out, out_shapes, "scan output", "example output", True)
        return carry, out
    return body_fun, carry_init, out_prefill
