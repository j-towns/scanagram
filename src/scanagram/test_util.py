from jax import lax
from scanagram.jax_test_util import check_close
from scanagram.util import safe_map, safe_zip
from scanagram.api import as_scan

map = safe_map
zip = safe_zip


def check_scan(f, xs, atol=None, rtol=None):
    body_fn, carry_init = as_scan(f, xs)
    carry_out, ys = lax.scan(body_fn, carry_init, xs)
    check_close(f(xs), ys, atol=atol, rtol=rtol)
