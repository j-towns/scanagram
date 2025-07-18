"""
Demonstration of how to use scanagram.custom_scanagram to define a scan
conversion rule for masked self-attention. We use multi-head attention here,
loosely based on the implementation from RecurrentGemma.
"""
import functools

import numpy as np
import einops
import jax.numpy as jnp
from jax import lax
import jax

import scanagram


_MIN_LOGITS_VALUE = -2.3819763e38  # Set to a large negative number.

def compute_mask(length):
    positions = jnp.arange(length)
    k_positions = positions[jnp.newaxis, :]
    q_positions = positions[:, jnp.newaxis]
    return k_positions <= q_positions

@scanagram.custom_scanagram
def masked_self_attention(qs_ks_vs):
    qs, ks, vs = qs_ks_vs
    _, length, _, _ = qs.shape

    logits = einops.einsum(qs, ks, "b t n h, b s n h -> b n t s")
    logits = logits * h**-0.5
    masked_logits = jnp.where(compute_mask(length), logits, _MIN_LOGITS_VALUE)

    probs = jax.nn.softmax(masked_logits, axis=-1)
    return einops.einsum(probs, vs, "b n t s, b s n h -> b t n h")

def make_cache_init(prefill, length):
    b, prefill_len, n, h = prefill.shape
    zeros_shape = b, length - prefill_len, n, h
    return jnp.concat([
        prefill, jnp.zeros_like(prefill, shape=zeros_shape)
    ], axis=1)

def cache_update(cache, t, k, v):
    cache_ks, cache_vs = cache
    cache_ks = lax.dynamic_update_index_in_dim(cache_ks, k, t, 1)
    cache_vs = lax.dynamic_update_index_in_dim(cache_vs, v, t, 1)
    return cache_ks, cache_vs

@masked_self_attention.def_scanagram
def scan_rule(scan_info, qs_ks_vs):
    assert scan_info.axis == 1
    qs, ks, vs = qs_ks_vs  # These are JAX ShapeDtypeStructs
    b, l, n, h = qs.shape

    if scan_info.prefill is not None:
        qs_prefill, ks_prefill, vs_prefill = scan_info.prefill
        out_prefill = masked_self_attention(scan_info.prefill)

        b_, prefill_len, n_, h_ = qs_prefill.shape
        assert (b, n, h) == (b_, n_, h_)
        assert prefill_len <= l
    else:
        prefill_len = 0
        out_prefill = None
        ks_prefill = vs_prefill = jnp.zeros((b, 0, n, h))

    cache_init = make_cache_init(ks_prefill, l), make_cache_init(vs_prefill, l)

    def body_fn(carry, q_k_v):
        t, cache = carry
        q, k, v = q_k_v
        assert q.shape == (b, n, h)
        assert k.shape == (b, n, h)
        assert v.shape == (b, n, h)
        cache_ks, cache_vs = cache_update(cache, t, k, v)

        logits = einops.einsum(q, cache_ks, "b n h, b s n h -> b n s")
        logits = logits * h**-0.5
        masked_logits = jnp.where(jnp.arange(l) <= t, logits, _MIN_LOGITS_VALUE)
        probs = jax.nn.softmax(masked_logits, axis=-1)
        encoded = einops.einsum(probs, cache_vs, "b n s, b s n h -> b n h")

        return (t + 1, (cache_ks, cache_vs)), encoded

    # Custom scan rule must return the output axis and prefill (wrapped in a
    # scanagram.ScanInfo), the body function, and the initial carry value.
    return (
        scanagram.ScanInfo(1, out_prefill), body_fn, (prefill_len, cache_init)
    )


if __name__ == "__main__":
    # Let's test the correctness of the rule above, on some small random inputs
    prefill_len = 2
    b, l, n, h = 2, 3, 5, 7

    qs_prefill = np.random.randn(prefill_len, b, n, h).astype("float32")
    ks_prefill = np.random.randn(prefill_len, b, n, h).astype("float32")
    vs_prefill = np.random.randn(prefill_len, b, n, h).astype("float32")

    def scan_like_fn(qs_ks_vs):
        qs, ks, vs = qs_ks_vs
        # Inputs and outputs scan along 0'th axis so we need to move it into
        # position:
        qs = jnp.moveaxis(jnp.concat([qs_prefill, qs]), 0, 1)
        ks = jnp.moveaxis(jnp.concat([ks_prefill, ks]), 0, 1)
        vs = jnp.moveaxis(jnp.concat([vs_prefill, vs]), 0, 1)
        result = masked_self_attention((qs, ks, vs))
        return jnp.moveaxis(result, 1, 0)[prefill_len:]

    rng_q, rng_k, rng_v = jax.random.split(jax.random.PRNGKey(0), 3)

    qs_example = np.random.randn(l, b, n, h).astype("float32")
    ks_example = np.random.randn(l, b, n, h).astype("float32")
    vs_example = np.random.randn(l, b, n, h).astype("float32")

    example = qs_example, ks_example, vs_example

    f, init_carry = scanagram.as_scan(scan_like_fn, example)

    np.testing.assert_allclose(
        scan_like_fn(example), lax.scan(f, init_carry, example)[1],
        atol=1e-6, rtol=1e-6
    )
