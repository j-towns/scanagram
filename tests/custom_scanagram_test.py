from jax import ShapeDtypeStruct
import jax.numpy as jnp
from jax import lax
from jax import jit, jvp, grad

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
        return ScanInfo(0), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    test_util.check_scan(f, xs)

def test_custom_scanagram_prefill():
    xs_dtype = jnp.dtype("int32")

    xs = jnp.array([3, 5])
    prefill = jnp.array([7, 8, 9])

    def g_ref(xs):
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]

    @custom_scanagram
    def g(xs):
        return jnp.array([7, 15, 24, 27, 32])

    @g.def_scanagram
    def g_scanagram_rule(scan_info, xs):
        assert type(xs) is ShapeDtypeStruct
        assert xs.dtype == xs_dtype
        carry_init = 24
        def body_fn(c, x):
            return c + x, c + x
        return ScanInfo(0, jnp.array([7, 15, 24])), body_fn, carry_init

    def f(xs):
        xs = jnp.concatenate([prefill, xs])
        return g(xs)[3:]

    jax_test_util.check_close(jnp.array([27, 32]) , f(xs))
    test_util.check_scan_with_prefill(g, xs, prefill)

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
        return ScanInfo(0), body_fn, carry_init

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
        return ScanInfo(0), body_fn, carry_init

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
        return ScanInfo(0), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    test_util.check_scan(f, xs)

def test_custom_scanagram_jit():
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
        return ScanInfo(0), body_fn, carry_init

    f = jit(f)
    jax_test_util.check_close(f_ref(xs), f(xs))
    test_util.check_scan(f, xs)

def test_custom_scanagram_jvp():
    xs_shape = (2,)
    xs_dtype = jnp.dtype("float32")

    xs = jnp.array([3., 5.])
    tangents = jnp.array([1., 2.])

    def f_ref(xs):
        xs = xs ** 2
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]

    @custom_scanagram
    def f(xs):
        xs = xs ** 2
        return jnp.array([xs[0], xs[0] + xs[1]])

    @f.def_scanagram
    def f_scanagram_rule(scan_info, xs):
        assert type(xs) is ShapeDtypeStruct
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        carry_init = 0
        def body_fn(c, x):
            x = x ** 2
            return c + x, c + x
        return ScanInfo(0), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    jax_test_util.check_close(
        jvp(f, (xs,), (tangents,)), jvp(f_ref, (xs,), (tangents,))
    )

def test_custom_scanagram_jvp_zero():
    xs_shape = (2,)
    xs_dtype = jnp.dtype("float32")

    xs = jnp.array([3., 5.])
    ys = jnp.array([4., 6.])
    tangents = jnp.array([1., 2.])

    def f_ref(xs):
        xs = ys + xs
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]

    @custom_scanagram
    def f(xs):
        xs = ys + xs
        return jnp.array([xs[0], xs[0] + xs[1]])

    @f.def_scanagram
    def f_scanagram_rule(scan_info, xs):
        pass

    jax_test_util.check_close(f_ref(xs), f(xs))
    jax_test_util.check_close(
        jvp(f, (xs,), (tangents,)), jvp(f_ref, (xs,), (tangents,))
    )

def test_custom_scanagram_grad():
    xs_shape = (2,)
    xs_dtype = jnp.dtype("float32")

    xs = jnp.array([3., 5.])

    def f_ref(xs):
        xs = xs ** 2
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        return lax.scan(lambda c, x: (c + x, c + x), 0, xs)[1]

    @custom_scanagram
    def f(xs):
        xs = xs ** 2
        return jnp.array([xs[0], xs[0] + xs[1]])

    @f.def_scanagram
    def f_scanagram_rule(scan_info, xs):
        assert type(xs) is ShapeDtypeStruct
        assert xs.shape == xs_shape
        assert xs.dtype == xs_dtype
        carry_init = 0
        def body_fn(c, x):
            x = x ** 2
            return c + x, c + x
        return ScanInfo(0), body_fn, carry_init

    jax_test_util.check_close(f_ref(xs), f(xs))
    jax_test_util.check_close(
        grad(lambda xs: jnp.sum(f(xs)))(xs),
        grad(lambda xs: jnp.sum(f_ref(xs)))(xs)
    )
