from jax import ShapeDtypeStruct
import jax.numpy as jnp
from jax import lax

from scanagram import custom_scanagram, ScanInfo
from scanagram import jax_test_util
from scanagram import test_util


def test_custom_scanagram():
    xs_shape = (2,)
    xs_dtype = jnp.dtype("int32")

    xs = jnp.array([3, 5])

    def f_ref(xs):
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]

    @custom_scanagram
    def f(xs):
        return jnp.array([xs[0], xs[0] + xs[1]])

    @f.def_scanagram
    def f_scanagram_rule(scan_info, xs):
        assert type(xs) is ShapeDtypeStruct
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        carry_init = 0
        def body_fn(c, x):
            return c + x, c + x
        return ScanInfo(0, 1), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    test_util.check_scan(f, xs)

def test_custom_scanagram_consts():
    xs_shape = (2,)
    xs_dtype = jnp.dtype("int32")

    xs = jnp.array([3, 5])
    ys = jnp.array([5])

    def f_ref(xs):
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return 5 + lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]

    @custom_scanagram
    def f(xs):
        return ys + jnp.array([xs[0], xs[0] + xs[1]])

    @f.def_scanagram
    def f_scanagram_rule(scan_info, xs):
        assert type(xs) is ShapeDtypeStruct
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        carry_init = 0
        def body_fn(c, x):
            return c + x, 5 + c + x
        return ScanInfo(0, 1), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    test_util.check_scan(f, xs)

def test_custom_scanagram_in_pytree():
    xs_shape = (2,)
    xs_dtype = jnp.dtype("int32")

    xs = {'xs': jnp.array([3, 5])}

    def f_ref(xs):
        xs = xs['xs']
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]

    @custom_scanagram
    def f(xs):
        xs = xs['xs']
        return jnp.array([xs[0], xs[0] + xs[1]])

    @f.def_scanagram
    def f_scanagram_rule(scan_info, xs):
        assert type(xs) is dict
        xs = xs['xs']
        assert type(xs) is ShapeDtypeStruct
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        carry_init = 0
        def body_fn(c, x):
            x = x['xs']
            return c + x, c + x
        return ScanInfo(0, 1), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    test_util.check_scan(f, xs)

def test_custom_scanagram_out_pytree():
    xs_shape = (2,)
    xs_dtype = jnp.dtype("int32")

    xs = jnp.array([3, 5])

    def f_ref(xs):
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return {'ys': lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]}

    @custom_scanagram
    def f(xs):
        return {'ys': jnp.array([xs[0], xs[0] + xs[1]])}

    @f.def_scanagram
    def f_scanagram_rule(scan_info, xs):
        assert type(xs) is ShapeDtypeStruct
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        carry_init = 0
        def body_fn(c, x):
            return c + x, {'ys': c + x}
        return ScanInfo(0, 1), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    test_util.check_scan(f, xs)
