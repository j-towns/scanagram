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

@masked_self_attention.def_scanagram_with_prefill
def scan_rule(axis, qs_ks_vs):
    assert axis == 1
    qs, ks, vs = qs_ks_vs  # These are JAX ShapeDtypeStructs
    b, l, n, h = qs.shape

    def init_fn(qs_ks_vs_prefill):
        # This function accepts input prefill and returns an initial carry
        # (prefill length and cache state) and the output prefill.
        qs_prefill, ks_prefill, vs_prefill = qs_ks_vs_prefill
        b_, prefill_len, n_, h_ = qs_prefill.shape
        assert (b, n, h) == (b_, n_, h_)
        assert prefill_len <= l
        out_prefill = masked_self_attention(qs_ks_vs_prefill)
        cache_init = (
            make_cache_init(ks_prefill, l), make_cache_init(vs_prefill, l)
        )
        return (prefill_len, cache_init), out_prefill

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

    # Custom scan rule must return the output axis, the init function (which
    # initializes the cache and computes prefill), and the body function.
    return 1, init_fn, body_fn


if __name__ == "__main__":
    # Let's test the correctness of the rule above, on some small random inputs
    prefill_len = 3
    b, l, n, h = 2, 11, 5, 7

    def scan_like_fn(qs_ks_vs):
        qs, ks, vs = qs_ks_vs
        # Inputs and outputs scan along 0'th axis so we need to move it into
        # position:
        qs = jnp.moveaxis(qs, 0, 1)
        ks = jnp.moveaxis(ks, 0, 1)
        vs = jnp.moveaxis(vs, 0, 1)
        result = masked_self_attention((qs, ks, vs))
        return jnp.moveaxis(result, 1, 0)

    qs_example = jax.ShapeDtypeStruct((l, b, n, h), "float32")
    ks_example = jax.ShapeDtypeStruct((l, b, n, h), "float32")
    vs_example = jax.ShapeDtypeStruct((l, b, n, h), "float32")
    example = qs_example, ks_example, vs_example

    qs_prefill = np.random.randn(prefill_len, b, n, h).astype("float32")
    ks_prefill = np.random.randn(prefill_len, b, n, h).astype("float32")
    vs_prefill = np.random.randn(prefill_len, b, n, h).astype("float32")
    prefill = qs_prefill, ks_prefill, vs_prefill

    f, init_carry, out_prefill = scanagram.as_scan_with_prefill(
        scan_like_fn, example, prefill
    )

    qs = np.random.randn(l - prefill_len, b, n, h).astype("float32")
    ks = np.random.randn(l - prefill_len, b, n, h).astype("float32")
    vs = np.random.randn(l - prefill_len, b, n, h).astype("float32")

    prefill_and_inputs = (
        np.concat([qs_prefill, qs]),
        np.concat([ks_prefill, ks]),
        np.concat([vs_prefill, vs]),
    )

    np.testing.assert_allclose(
        scan_like_fn(prefill_and_inputs)[:prefill_len],
        out_prefill,
        atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        scan_like_fn(prefill_and_inputs)[prefill_len:],
        lax.scan(f, init_carry, (qs, ks, vs))[1],
        atol=1e-6, rtol=1e-6
    )
