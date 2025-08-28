from functools import partial

from jax import lax
from jax import ShapeDtypeStruct
from jax import tree
import jax.numpy as jnp
from scanagram.jax_test_util import check_close
from scanagram.util import safe_map, safe_zip
from scanagram.api import as_scan, as_scan_with_prefill

map = safe_map
zip = safe_zip


def check_scan(f, xs, atol=None, rtol=None):
    body_fn, carry_init = as_scan(f, xs)
    carry_out, ys = lax.scan(body_fn, carry_init, xs)
    check_close(f(xs), ys, atol=atol, rtol=rtol)

def check_scan_with_prefill(f, xs, prefills, atol=None, rtol=None):
    prefill_len = len(tree.leaves(prefills)[0])
    example_xs = tree.map(
        lambda x, p: ShapeDtypeStruct(
            (x.shape[0] + p.shape[0],) + x.shape[1:], x.dtype
        ),
        xs, prefills
    )
    body_fn, carry_init, out_prefills = as_scan_with_prefill(
        f, example_xs, prefills
    )
    carry_out, ys = lax.scan(body_fn, carry_init, xs)
    ys_correct = f(
        tree.map(lambda p, x: jnp.concatenate([p, x]), prefills, xs)
    )
    out_prefills_correct = tree.map(lambda y: y[:prefill_len], ys_correct)
    ys_correct = tree.map(lambda y: y[prefill_len:], ys_correct)
    check_close(ys_correct, ys, atol=atol, rtol=rtol)
    check_close(out_prefills_correct, out_prefills, atol=atol, rtol=rtol)
