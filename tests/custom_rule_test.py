from jax import lax
import jax.numpy as jnp

import numpy as np

from scanagram import test_util
from scanagram import custom_scanagram
from scanagram.util import unzip3


@custom_scanagram
def causal_conv(lhs, rhs):
    window_size = rhs.shape[0]
    padded_lhs = lax.pad(
        lhs, 0., ((0, 0, 0), (window_size - 1, 0, 0), (0, 0, 0))
    )
    return lax.conv_general_dilated(
        padded_lhs, rhs, [1], "VALID",
        dimension_numbers=("NTC", "TIO", "NTC"),
    ),

@causal_conv.defrule
def causal_conv_scanagram_rule(inscanvars, lhs, rhs):
    argnums, axes, strides = unzip3(inscanvars)
    assert argnums == (0,)
    assert axes == (1,)
    assert strides == (1,)
    window_size = rhs.shape[0]
    init_carry = jnp.zeros((lhs.shape[0], window_size - 1, lhs.shape[2]))
    def body_fn(carry, x, rhs_):
        lhs = jnp.concatenate([carry, jnp.expand_dims(x, 1)], 1)
        out = jnp.squeeze(lax.conv_general_dilated(
            lhs, rhs, [1], "VALID",
            dimension_numbers=("NTC", "TIO", "NTC"),
        ), 1)
        carry_new = lax.slice_in_dim(lhs, 1, lhs.shape[1], 1, 1)
        return carry_new, (out,)
    return init_carry, body_fn, [(0, 1, 1)], []

def test_custom_rule():
    rng = np.random.RandomState(0)
    lhs = rng.randn(3, 12, 5)
    rhs = rng.randn(3, 5, 6)
    def f(lhs):
        lhs = jnp.moveaxis(lhs, 0, 1)
        result, = causal_conv(lhs, rhs)
        return jnp.moveaxis(result, 1, 0)

    test_util.check_scan(f, lhs)
