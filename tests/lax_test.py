import functools
import collections
import itertools

from jax import lax
from jax import dtypes
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
import numpy as np
import numpy.testing as np_testing
import pytest

from scanagram import jax_test_util as jtu
from scanagram import test_util
from scanagram.core import ScanConversionError
from scanagram.util import safe_map, safe_zip

###############################################################################
# This section is copied from jax/_src/internal_test_util/lax_test_util.py

# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# For standard unops and binops, we can generate a large number of tests on
# arguments of appropriate shapes and dtypes using the following table.

float_dtypes = jtu.dtypes.all_floating
complex_elem_dtypes = jtu.dtypes.floating
complex_dtypes = jtu.dtypes.complex
inexact_dtypes = jtu.dtypes.all_inexact
int_dtypes = jtu.dtypes.all_integer
uint_dtypes = jtu.dtypes.all_unsigned
bool_dtypes = jtu.dtypes.boolean

default_dtypes = float_dtypes + int_dtypes
number_dtypes = (
    float_dtypes + complex_dtypes + int_dtypes + uint_dtypes
)
all_dtypes = (
    number_dtypes + bool_dtypes
)
python_scalar_types = [bool, int, float, complex]

compatible_shapes = [[(3,)], [(2, 3, 4), (2, 1, 4)], [(1, 1), (1, 3)]]

OpRecord = collections.namedtuple(
    "OpRecord", ["op", "nargs", "dtypes", "rng_factory", "tol"]
)


def op_record(op, nargs, dtypes, rng_factory, tol=None):
  return OpRecord(op, nargs, dtypes, rng_factory, tol)


NamedReducerOpRecord = collections.namedtuple(
    "NamedReducerOpRecord", ["op", "reference_op", "dtypes"]
)

def lax_named_reduce_ops():
  return [
      NamedReducerOpRecord(lax.reduce_sum, np.sum, number_dtypes),
      NamedReducerOpRecord(lax.reduce_prod, np.prod, number_dtypes),
      NamedReducerOpRecord(lax.reduce_max, np.max, all_dtypes),
      NamedReducerOpRecord(lax.reduce_min, np.min, all_dtypes),
      NamedReducerOpRecord(lax.reduce_and, np.bitwise_and.reduce,
                           bool_dtypes + int_dtypes + uint_dtypes),
      NamedReducerOpRecord(lax.reduce_or, np.bitwise_or.reduce,
                           bool_dtypes + int_dtypes + uint_dtypes),
      NamedReducerOpRecord(lax.reduce_xor, np.bitwise_xor.reduce,
                           bool_dtypes + int_dtypes + uint_dtypes),
  ]


def lax_ops():
  return [
      op_record(
          "neg", 1, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record("sign", 1, default_dtypes + uint_dtypes, jtu.rand_small),
      op_record("floor", 1, float_dtypes, jtu.rand_small),
      op_record("ceil", 1, float_dtypes, jtu.rand_small),
      op_record("round", 1, float_dtypes, jtu.rand_default),
      op_record(
          "nextafter",
          2,
          [f for f in float_dtypes if f != dtypes.bfloat16],
          jtu.rand_default,
          tol=0,
      ),
      op_record("is_finite", 1, float_dtypes, jtu.rand_small),
      op_record("exp", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      op_record("exp2", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
      # precision.
      op_record(
          "expm1",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.float64: 1e-8},
      ),
      op_record(
          "log", 1, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      op_record(
          "log1p", 1, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
      # ~float32 precision.
      # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
      op_record(
          "tanh",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.float64: 1e-9, np.complex128: 1e-7},
      ),
      op_record(
          "logistic", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "sin", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "cos", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record("atan2", 2, float_dtypes, jtu.rand_default),
      op_record("sqrt", 1, float_dtypes, jtu.rand_positive),
      op_record("sqrt", 1, complex_dtypes, jtu.rand_default),
      op_record("rsqrt", 1, float_dtypes, jtu.rand_positive),
      op_record("rsqrt", 1, complex_dtypes, jtu.rand_default),
      op_record("cbrt", 1, float_dtypes, jtu.rand_default),
      op_record(
          "square", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "reciprocal",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_positive,
      ),
      op_record(
          "tan",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_default,
          {np.float32: 3e-5},
      ),
      op_record(
          "asin",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.complex128: 5e-12},
      ),
      op_record("acos", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      op_record("atan", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      op_record(
          "asinh",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_default,
          tol={np.complex64: 1e-4, np.complex128: 1e-5},
      ),
      op_record(
          "acosh", 1, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      # TODO(b/155331781): atanh has only ~float precision
      op_record(
          "atanh",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.float64: 1e-9},
      ),
      op_record(
          "sinh", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "cosh", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "lgamma",
          1,
          float_dtypes,
          jtu.rand_positive,
          {
              np.float32: 1e-5,
              np.float64: 1e-14,
          },
      ),
      op_record(
          "digamma",
          1,
          float_dtypes,
          jtu.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record(
          "betainc",
          3,
          float_dtypes,
          jtu.rand_uniform,
          {
              np.float32: 1e-5,
              np.float64: 1e-12,
          },
      ),
      op_record(
          "igamma",
          2,
          [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
          jtu.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record(
          "igammac",
          2,
          [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
          jtu.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record("erf", 1, float_dtypes, jtu.rand_small),
      op_record("erfc", 1, float_dtypes, jtu.rand_small),
      # TODO(b/142976030): the approximation of erfinf used by XLA is only
      # accurate to float32 precision.
      op_record(
          "erf_inv", 1, float_dtypes, jtu.rand_small, {np.float64: 1e-9}
      ),
      op_record("bessel_i0e", 1, float_dtypes, jtu.rand_default),
      op_record("bessel_i1e", 1, float_dtypes, jtu.rand_default),
      op_record("real", 1, complex_dtypes, jtu.rand_default),
      op_record("imag", 1, complex_dtypes, jtu.rand_default),
      op_record("complex", 2, complex_elem_dtypes, jtu.rand_default),
      op_record(
          "conj",
          1,
          complex_elem_dtypes + complex_dtypes,
          jtu.rand_default,
      ),
      op_record(
          "abs", 1, default_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "pow", 2, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      op_record("bitwise_and", 2, bool_dtypes, jtu.rand_small),
      op_record("bitwise_not", 1, bool_dtypes, jtu.rand_small),
      op_record("bitwise_or", 2, bool_dtypes, jtu.rand_small),
      op_record("bitwise_xor", 2, bool_dtypes, jtu.rand_small),
      op_record(
          "population_count", 1, int_dtypes + uint_dtypes, jtu.rand_int
      ),
      op_record("clz", 1, int_dtypes + uint_dtypes, jtu.rand_int),
      op_record(
          "add", 2, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record(
          "sub", 2, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record(
          "mul", 2, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record(
          "div", 2, default_dtypes + complex_dtypes, jtu.rand_nonzero
      ),
      op_record("rem", 2, default_dtypes, jtu.rand_nonzero),
      op_record("max", 2, all_dtypes, jtu.rand_small),
      op_record("min", 2, all_dtypes, jtu.rand_small),
      op_record("eq", 2, all_dtypes, jtu.rand_some_equal),
      op_record("ne", 2, all_dtypes, jtu.rand_small),
      op_record("ge", 2, default_dtypes, jtu.rand_small),
      op_record("gt", 2, default_dtypes, jtu.rand_small),
      op_record("le", 2, default_dtypes, jtu.rand_small),
      op_record("lt", 2, default_dtypes, jtu.rand_small),
      op_record("polygamma", 2, float_dtypes, jtu.rand_positive),
      op_record("zeta", 2, float_dtypes, jtu.rand_positive),
  ]

###############################################################################

@pytest.mark.parametrize(
    'op_name,argnum,rng_factory,shapes,dtype,tol',
    [(rec.op, argnum, rec.rng_factory, shapes, dtype, rec.tol)
     for rec in lax_ops()
     for shape_group in compatible_shapes
     for shapes in itertools.combinations_with_replacement(
         shape_group, rec.nargs
     )
     for dtype in rec.dtypes
     for argnum in range(rec.nargs)])
def test_nary(op_name, argnum, rng_factory, shapes, dtype, tol):
    if shapes[argnum][0] == 1 and any(s[0] > 1 for s in shapes):
        return
    rng = rng_factory(np.random)
    args = tuple(rng(shape, dtype) for shape in shapes)
    def f(xs):
        args_ = list(args)
        args_[argnum] = xs
        return getattr(lax, op_name)(*args_)
    test_util.check_scan(f, args[argnum], atol=tol, rtol=tol)

def test_nary_other_axis():
    rng = np.random.RandomState(0)
    xs = rng.randn(2, 3, 4).astype("float32")
    y = rng.randn(3, 2, 4).astype("float32")
    def f(xs):
        return jnp.moveaxis(lax.add(jnp.moveaxis(xs, 0, 1), y), 1, 0)
    test_util.check_scan(f, xs)

def test_nary_prefill():
    rng = np.random.RandomState(0)
    xs = rng.randn(12, 3).astype("float32")
    ys = rng.randn(15, 3).astype("float32")
    prefill = rng.randn(3, 3).astype("float32")
    def f(xs):
        return ys * xs
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_nary_prefill_batch():
    rng = np.random.RandomState(0)
    xs = rng.randn(12, 3).astype("float32")
    ys = rng.randn(1, 3).astype("float32")
    prefill = rng.randn(3, 3).astype("float32")
    def f(xs):
        return ys * xs
    test_util.check_scan_with_prefill(f, xs, prefill)

@pytest.mark.parametrize(
    'op,shape,axes,dtype',
    [(rec.op, shape, axes, dtype)
     for rec in lax_named_reduce_ops()
     for (shape, axes) in [[(3, 4, 5), (1,)], [(3, 4, 5), (1, 2)]]
     for dtype in rec.dtypes])
def test_reduce_named(op, shape, axes, dtype):
    rng_factory = (jtu.rand_default if dtypes.issubdtype(dtype, np.integer)
                   else jtu.rand_small)
    rng = rng_factory(np.random)
    arg = rng(shape, dtype)
    fun = functools.partial(op, axes=axes)
    test_util.check_scan(fun, arg)

@pytest.mark.parametrize(
    'op,shape,axes,dtype',
    [(rec.op, shape, axes, dtype)
     for rec in lax_named_reduce_ops()[:1]
     for (shape, axes) in [[(3, 4, 5), (1,)], [(3, 4, 5), (1, 2)]]
     for dtype in rec.dtypes])
def test_reduce_named_prefill(op, shape, axes, dtype):
    rng_factory = (jtu.rand_default if dtypes.issubdtype(dtype, np.integer)
                   else jtu.rand_small)
    rng = rng_factory(np.random)
    arg = rng(shape, dtype)
    prefill = rng(shape, dtype)
    test_util.check_scan_with_prefill(
        functools.partial(op, axes=axes), arg, prefill
    )

def test_scan():
    rng = np.random.RandomState(0)
    init_carry = np.zeros(2, "float32")
    def f(xs):
        carry_out, ys = lax.scan(
            lambda carry, x: (carry + x, carry + x), init_carry, xs
        )
        return ys
    xs = rng.randn(5, 2).astype("float32")
    test_util.check_scan(f, xs)

def test_scan_carry():
    rng = np.random.RandomState(0)
    init_carry = np.ones(2, "float32")
    def f(init_carry):
        def body_fn(carry, x):
            return carry + x, None

        carry_out, _ = lax.scan(
            body_fn, init_carry, jnp.arange(6).reshape(3, 2)
        )
        return carry_out
    test_util.check_scan(f, init_carry)

def test_scan_consts():
    rng = np.random.RandomState(0)
    init_carry = np.ones(2, "float32")
    c = np.ones(2, "float32")
    def f(c):
        def body_fn(carry, x):
            return carry, c + x

        _, ys = lax.scan(
            body_fn, init_carry, jnp.arange(6).reshape(3, 2)
        )
        return jnp.moveaxis(ys, 1, 0)
    test_util.check_scan(f, c)

def test_scan_non_zero_axis():
     rng = np.random.RandomState(0)
     init_carry = np.arange(3, dtype="float32")

     def f(xs):
         xs_moved = jnp.moveaxis(xs, 0, 1)
         carry_out, ys = lax.scan(
             lambda carry, x: (carry + 1, carry + x),
             init_carry, xs_moved
         )
         return jnp.moveaxis(ys, 1, 0)

     xs = rng.randn(3, 2).astype("float32")
     test_util.check_scan(f, xs)

def test_scan_prefill():
    rng = np.random.RandomState(0)
    init_carry = np.zeros(2, "float32")
    def f(xs):
        _, ys = lax.scan(
            lambda carry, x: (carry + x, carry + x), init_carry, xs
        )
        return ys
    xs = rng.randn(5, 2).astype("float32")
    prefill = rng.randn(2, 2).astype("float32")
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_scan_some_inputs():
    rng = np.random.RandomState(0)
    init_carry = np.zeros(2, "float32")
    xs = rng.randn(6, 2).astype("float32")
    ys = rng.randn(6, 2).astype("float32")

    def body_fn(carry, x_and_y):
        x, y = x_and_y
        return carry + x, carry + y + x

    def f(xs):
        _, out = lax.scan(body_fn, init_carry, (xs, ys))
        return out
    test_util.check_scan(f, xs)

def test_scan_some_inputs_prefill():
    rng = np.random.RandomState(0)
    init_carry = np.zeros(2, "float32")
    xs = rng.randn(6, 2).astype("float32")
    ys = rng.randn(8, 2).astype("float32")
    prefill = rng.randn(2, 2).astype("float32")

    def body_fn(carry, x_and_y):
        x, y = x_and_y
        return carry + x, carry + y + x

    def f(xs):
        _, out = lax.scan(body_fn, init_carry, (xs, ys))
        return out
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_transpose():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.transpose(lax.transpose(xs, (1, 2, 0)), (2, 1, 0))
    xs = rng.randn(2, 3, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_transpose_prefill():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.transpose(lax.transpose(xs, (1, 2, 0)), (2, 1, 0))
    xs = rng.randn(2, 3, 4).astype("float32")
    prefill = rng.randn(5, 3, 4).astype("float32")
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_transpose_wrong_axis():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.transpose(xs, (1, 2, 0))
    xs = rng.randn(2, 3, 4).astype("float32")
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, xs)

def test_broadcast_in_dim():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.broadcast_in_dim(xs, (2, 3, 4, 5), (0, 1, 2))
    xs = rng.randn(2, 1, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_broadcast_in_dim_prefill():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.broadcast_in_dim(xs, (5, 3, 4, 5), (0, 1, 2))
    xs = rng.randn(2, 1, 4).astype("float32")
    prefill = rng.randn(3, 1, 4).astype("float32")
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_broadcast_in_dim_other_axis():
    rng = np.random.RandomState(0)
    def f(xs):
        xs = jnp.moveaxis(xs, 0, 1)
        return jnp.moveaxis(
            lax.broadcast_in_dim(xs, (3, 2, 4, 5), (0, 1, 2)), 1, 0
        )
    xs = rng.randn(2, 1, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_broadcast_in_dim_broadcast_dimensions():
    rng = np.random.RandomState(0)
    def f(xs):
        xs = jnp.moveaxis(xs, 0, 1)
        ys = lax.broadcast_in_dim(xs, (1, 2, 3, 4), (1, 2, 3))
        return jnp.moveaxis(ys, 2, 0)
    xs = rng.randn(3, 2, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_conv_batch():
    rng = np.random.RandomState(0)
    lhs = rng.randn(2, 3, 4, 5).astype("float32")
    rhs = rng.randn(1, 2, 5, 6).astype("float32")
    def f(x):
        return lax.conv_general_dilated(
            x, rhs, window_strides=[1, 1], padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
    test_util.check_scan(f, lhs)

def test_conv_causal():
    window_size = 2
    rng = np.random.RandomState(0)
    lhs = rng.randn(6, 4, 5).astype("float32")
    rhs = rng.randn(window_size, 5, 6).astype("float32")
    def f(x):
        return lax.conv_general_dilated(
            x, rhs, window_strides=[1], padding=[(window_size - 1, 0)],
            dimension_numbers=("TNC", "TIO", "TNC"),
        )
    test_util.check_scan(f, lhs)

def test_conv_causal_prefill_large():
    window_size = 2
    rng = np.random.RandomState(0)
    lhs = rng.randn(6, 4, 5).astype("float32")
    prefill = rng.randn(4, 4, 5).astype("float32")
    rhs = rng.randn(window_size, 5, 6).astype("float32")
    def f(x):
        return lax.conv_general_dilated(
            x, rhs, window_strides=[1], padding=[(window_size - 1, 0)],
            dimension_numbers=("TNC", "TIO", "TNC"),
        )
    test_util.check_scan_with_prefill(f, lhs, prefill)

def test_conv_causal_prefill_small():
    window_size = 4
    rng = np.random.RandomState(0)
    lhs = rng.randn(6, 4, 5).astype("float32")
    prefill = rng.randn(2, 4, 5).astype("float32")
    rhs = rng.randn(window_size, 5, 6).astype("float32")
    def f(x):
        return lax.conv_general_dilated(
            x, rhs, window_strides=[1], padding=[(window_size - 1, 0)],
            dimension_numbers=("TNC", "TIO", "TNC"),
        )
    test_util.check_scan_with_prefill(f, lhs, prefill)

def test_conv_rhs_dilation():
    window_size = 2
    rhs_dilation = 2
    rng = np.random.RandomState(0)
    lhs = rng.randn(12, 4, 5).astype("float32")
    rhs = rng.randn(window_size, 5, 6).astype("float32")
    def f(x):
        return lax.conv_general_dilated(
            x, rhs, window_strides=[1], padding=[(
                rhs_dilation * (window_size - 1), 0)],
            rhs_dilation=(rhs_dilation,),
            dimension_numbers=("TNC", "TIO", "TNC"),
        )
    test_util.check_scan(f, lhs)

def test_conv_transposed():
    window_size = 2
    rng = np.random.RandomState(0)
    lhs = rng.randn(6, 4, 5).astype("float32")
    rhs = rng.randn(window_size, 5, 6).astype("float32")
    def f(x):
        y = jnp.moveaxis(x, 0, 1)
        return jnp.moveaxis(lax.conv_general_dilated(
            y, rhs, window_strides=[1], padding=[(window_size - 1, 0)],
            dimension_numbers=("NTC", "TIO", "NTC"),
        ), 0, 1)
    test_util.check_scan(f, lhs)

def test_slice():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 5).astype("float32")
    def f(operand):
        return lax.slice(operand, (0, 2, 1), (6, 3, 5), (1, 1, 2))
    test_util.check_scan(f, operand)

def test_slice_prefill():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 5).astype("float32")
    prefill = rng.randn(2, 4, 5).astype("float32")
    def f(operand):
        return lax.slice(operand, (0, 2, 1), (8, 3, 5), (1, 1, 2))
    test_util.check_scan_with_prefill(f, operand, prefill)

def test_slice_none_stride():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 5).astype("float32")
    def f(operand):
        return lax.slice(operand, (0, 2, 1), (6, 3, 5))
    test_util.check_scan(f, operand)

def test_pad():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4).astype("float32")
    def f(operand):
        return lax.pad(operand, 3., [(0, 0, 0), (1, 2, 3)])
    test_util.check_scan(f, operand)

def test_pad_prefill():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4).astype("float32")
    def f(operand):
        return lax.pad(operand, 3., [(0, 0, 0), (1, 2, 3)])
    test_util.check_scan(f, operand)

def test_concatenate():
    rng = np.random.RandomState(0)
    x, y = rng.randn(6, 4).astype("float32"), rng.randn(6, 3).astype("float32")
    def f(x):
        return lax.concatenate([x, y], 1)
    test_util.check_scan(f, x)

def test_concatenate_prefill():
    rng = np.random.RandomState(0)
    x, y = rng.randn(4, 4).astype("float32"), rng.randn(6, 3).astype("float32")
    prefill = rng.randn(2, 4).astype("float32")
    def f(x):
        return lax.concatenate([x, y], 1)
    test_util.check_scan_with_prefill(f, x, prefill)

def test_concatenate_both_scanned():
    rng = np.random.RandomState(0)
    x, y = rng.randn(6, 4).astype("float32"), rng.randn(6, 3).astype("float32")
    def f(x_and_y):
        x, y = x_and_y
        return lax.concatenate([x, y], 1)
    test_util.check_scan(f, (x, y))

def test_dot_general_both_batch():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(6, 4, 3).astype("float32"),
        rng.randn(6, 3, 4).astype("float32")
    )
    def f(x_and_y):
        x, y = x_and_y
        return lax.dot_general(x, y, (([1], [2]), ([0], [0])))
    test_util.check_scan(f, (x, y))

def test_dot_general_both_batch_prefill():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(6, 4, 3).astype("float32"),
        rng.randn(6, 3, 4).astype("float32")
    )
    x_prefill, y_prefill = (
        rng.randn(2, 4, 3).astype("float32"),
        rng.randn(2, 3, 4).astype("float32")
    )
    def f(x_and_y):
        x, y = x_and_y
        return lax.dot_general(x, y, (([1], [2]), ([0], [0])))
    test_util.check_scan_with_prefill(f, (x, y), (x_prefill, y_prefill))

def test_dot_general_lhs_batch():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(6, 4, 3).astype("float32"),
        rng.randn(6, 3, 4).astype("float32")
    )
    def f(x):
        return lax.dot_general(x, y, (([1], [2]), ([0], [0])))
    test_util.check_scan(f, x)

def test_dot_general_lhs_batch_prefill():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(4, 4, 3).astype("float32"),
        rng.randn(6, 3, 4).astype("float32")
    )
    prefill = rng.randn(2, 4, 3).astype("float32")
    def f(x):
        return lax.dot_general(x, y, (([1], [2]), ([0], [0])))
    test_util.check_scan_with_prefill(f, x, prefill)

def test_dot_general_lhs_non_batch():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(6, 3, 4).astype("float32"),
        rng.randn(6, 3, 4).astype("float32")
    )
    def f(x):
        return lax.dot_general(
            jnp.moveaxis(x, 0, 1), y, (([2], [2]), ([1], [0]))
        )
    test_util.check_scan(f, x)

def test_dot_general_lhs_non_batch_prefill():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(4, 3, 4).astype("float32"),
        rng.randn(6, 3, 4).astype("float32")
    )
    prefill = rng.randn(2, 3, 4).astype("float32")
    def f(x):
        return lax.dot_general(
            jnp.moveaxis(x, 0, 1), y, (([2], [2]), ([1], [0]))
        )
    test_util.check_scan_with_prefill(f, x, prefill)

def test_dot_general_rhs_batch():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(6, 4, 3).astype("float32"),
        rng.randn(6, 3, 4).astype("float32")
    )
    def f(y):
        return lax.dot_general(x, y, (([1], [2]), ([0], [0])))
    test_util.check_scan(f, y)

def test_dot_general_rhs_batch_prefill():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(6, 4, 3).astype("float32"),
        rng.randn(4, 3, 4).astype("float32")
    )
    prefill = rng.randn(2, 3, 4).astype("float32")
    def f(y):
        return lax.dot_general(x, y, (([1], [2]), ([0], [0])))
    test_util.check_scan_with_prefill(f, y, prefill)

def test_dot_general_rhs_non_batch():
    rng = np.random.RandomState(0)
    x, y = (
        rng.randn(6, 3, 4).astype("float32"),
        rng.randn(4, 3, 4).astype("float32")
    )
    prefill = rng.randn(2, 3, 4).astype("float32")
    def f(y):
        return lax.dot_general(
            jnp.moveaxis(x, 0, 1), y, (([2], [2]), ([1], [0]))
        )
    test_util.check_scan_with_prefill(f, y, prefill)

def test_reshape_pre():
    rng = np.random.RandomState(0)
    x = rng.randn(3, 6, 4).astype("float32")
    def f(x):
        x = jnp.moveaxis(x, 0, 1)
        return jnp.moveaxis(lax.reshape(x, (3, 2, 3, 4)), 2, 0)
    test_util.check_scan(f, x)

def test_reshape_pre_prefill():
    rng = np.random.RandomState(0)
    x = rng.randn(1, 6, 4).astype("float32")
    prefill = rng.randn(2, 6, 4).astype("float32")
    def f(x):
        x = jnp.moveaxis(x, 0, 1)
        return jnp.moveaxis(lax.reshape(x, (3, 2, 3, 4)), 2, 0)
    test_util.check_scan_with_prefill(f, x, prefill)

def test_reshape_post():
    rng = np.random.RandomState(0)
    x = rng.randn(3, 6, 4).astype("float32")
    def f(x):
        return lax.reshape(x, (3, 2, 3, 4))
    test_util.check_scan(f, x)

def test_reshape_post_prefill():
    rng = np.random.RandomState(0)
    x = rng.randn(1, 6, 4).astype("float32")
    prefill = rng.randn(2, 6, 4).astype("float32")
    def f(x):
        return lax.reshape(x, (3, 2, 3, 4))
    test_util.check_scan_with_prefill(f, x, prefill)

def test_reshape_dimensions():
    rng = np.random.RandomState(0)
    x = rng.randn(3, 6, 4).astype("float32")
    def f(x):
        return jnp.moveaxis(lax.reshape(x, (2, 3, 4, 3), (1, 2, 0)), 3, 0)
    test_util.check_scan(f, x)

def test_reshape_dimensions_prefill():
    rng = np.random.RandomState(0)
    x = rng.randn(1, 6, 4).astype("float32")
    prefill = rng.randn(2, 6, 4).astype("float32")
    def f(x):
        return jnp.moveaxis(lax.reshape(x, (2, 3, 4, 3), (1, 2, 0)), 3, 0)
    test_util.check_scan_with_prefill(f, x, prefill)

def test_reshape_dimensions():
    rng = np.random.RandomState(0)
    x = rng.randn(3, 6, 4).astype("float32")
    def f(x):
        return jnp.moveaxis(lax.reshape(x, (2, 3, 4, 3), (1, 2, 0)), 3, 0)
    test_util.check_scan(f, x)

def test_pjit():
    rng = np.random.RandomState(0)

    @jax.jit
    def f(xs):
        return lax.transpose(lax.transpose(xs, (1, 2, 0)), (2, 1, 0))
    xs = rng.randn(2, 3, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_pjit_some_outvars_not_scanned():
    rng = np.random.RandomState(0)

    @jax.jit
    def g(xs):
        return (
            lax.transpose(lax.transpose(xs, (1, 2, 0)), (2, 1, 0)),
            jnp.ones(3)
        )

    def f(xs):
        xs, ones = g(xs)
        return xs + ones

    xs = rng.randn(2, 3, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_pjit_second_arg_scanned():
    rng = np.random.RandomState(0)

    def f(xs):
        return jax.jit(jnp.add)(xs, ys)
    xs = rng.randn(2, 3, 4).astype("float32")
    ys = rng.randn(2, 3, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_checkpoint():
    rng = np.random.RandomState(0)

    @jax.checkpoint
    def f(xs):
        return lax.transpose(lax.transpose(xs, (1, 2, 0)), (2, 1, 0))
    xs = rng.randn(2, 3, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_checkpoint_name():
    rng = np.random.RandomState(0)

    @jax.checkpoint
    def f(xs):
        return lax.transpose(
            checkpoint_name(
                lax.transpose(xs, (1, 2, 0)), "x"
            ), (2, 1, 0)
        )
    xs = rng.randn(2, 3, 4).astype("float32")
    test_util.check_scan(f, xs)

def test_split():
    rng = np.random.RandomState(0)

    def f(xs):
        return lax.split(xs, (2, 4), 1)
    xs = rng.randn(2, 6).astype("float32")
    test_util.check_scan(f, xs)

def test_split_prefill():
    rng = np.random.RandomState(0)

    def f(xs):
        return lax.split(xs, (2, 4), 1)
    xs = rng.randn(2, 6).astype("float32")
    prefill = rng.randn(3, 6).astype("float32")
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_custom_vjp():
    rng = np.random.RandomState(0)

    @jax.custom_vjp
    def f(xs):
        return 2 * xs

    f.defvjp(
        lambda xs: (2 * xs, None),
        lambda res, g: (2 * g,),
    )

    xs = rng.randn(2, 6).astype("float32")
    test_util.check_scan(f, xs)

def test_custom_jvp():
    rng = np.random.RandomState(0)

    @jax.custom_jvp
    def f(xs):
        return 2 * xs

    f.defjvp(lambda xs, ts: (2 * xs, 2 * ts))

    xs = rng.randn(2, 6).astype("float32")
    test_util.check_scan(f, xs)

def test_squeeze():
    rng = np.random.RandomState(0)
    xs = rng.randn(3, 1, 4).astype("float32")
    def f(xs):
        return lax.squeeze(xs, [1])
    test_util.check_scan(f, xs)

def test_reduce_p_basic():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def add_op(x, y):
        return x + y

    def f(xs):
        return lax.reduce(xs, jnp.array(0.0), add_op, dimensions=(1,))
    test_util.check_scan(f, xs)

def test_reduce_p_multiple_axes():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def add_op(x, y):
        return x + y

    def f(xs):
        return lax.reduce(xs, jnp.array(0.0), add_op, dimensions=(1, 2))
    test_util.check_scan(f, xs)

def test_reduce_p_custom_computation():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4).astype("float32")

    def max_op(x, y):
        return jnp.maximum(x, y)

    def f(xs):
        return lax.reduce(xs, jnp.array(-jnp.inf), max_op, dimensions=(1,))
    test_util.check_scan(f, xs)

def test_reduce_p_multiple_operands():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4).astype("float32")
    ys = rng.randn(6, 4).astype("float32")

    def sum_op(accs, vals):
        x_acc, y_acc = accs
        x, y = vals
        return x_acc + x, y_acc + y

    def f(inputs):
        xs, ys = inputs
        return lax.reduce((xs, ys), (jnp.array(0.0), jnp.array(0.0)),
                         sum_op, dimensions=(1,))
    test_util.check_scan(f, (xs, ys))

def test_reduce_p_prefill():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4).astype("float32")
    prefill = rng.randn(3, 4).astype("float32")

    def add_op(x, y):
        return x + y

    def f(xs):
        result = lax.reduce(xs, jnp.array(0.0), add_op, dimensions=(1,))
        return result
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_reduce_p_axis_error():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def add_op(x, y):
        return x + y

    def f(xs):
        # This should fail because we're trying to reduce along the scan axis (0)
        return lax.reduce(xs, jnp.array(0.0), add_op, dimensions=(0, 1))

    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, xs)

def test_reduce_p_complex_computation():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4).astype("float32")

    def weighted_sum_op(acc, x):
        # More complex computation: weighted sum with position
        return acc + x * (acc + 1.0)

    def f(xs):
        return lax.reduce(xs, jnp.array(0.0), weighted_sum_op, dimensions=(1,))
    test_util.check_scan(f, xs)

def test_rev_basic():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        return lax.rev(xs, dimensions=(1,))
    test_util.check_scan(f, xs)

def test_rev_multiple_dimensions():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        return lax.rev(xs, dimensions=(1, 2))
    test_util.check_scan(f, xs)

def test_rev_1d():
    rng = np.random.RandomState(0)
    xs = rng.randn(8).astype("float32")

    def f(xs):
        # This should fail - trying to reverse the scan axis
        return lax.rev(xs, dimensions=(0,))
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, xs)

def test_rev_axis_error():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        # This should fail - trying to reverse the scan axis (0)
        return lax.rev(xs, dimensions=(0, 1))
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, xs)

def test_rev_with_prefill():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")
    prefill = rng.randn(3, 4, 5).astype("float32")

    def f(xs):
        return lax.rev(xs, dimensions=(1,))
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_rev_other_axis():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        xs = jnp.moveaxis(xs, 0, 1)
        xs = lax.rev(xs, dimensions=(0,))
        return jnp.moveaxis(xs, 1, 0)
    test_util.check_scan(f, xs)

def test_rev_empty_dimensions():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        # Reverse no dimensions (should be identity)
        return lax.rev(xs, dimensions=())
    test_util.check_scan(f, xs)

def test_sort_basic():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        return lax.sort(xs, dimension=1)
    test_util.check_scan(f, xs)

def test_sort_last_dimension():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        return lax.sort(xs, dimension=2)
    test_util.check_scan(f, xs)

def test_sort_axis_error():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        # This should fail - trying to sort along the scan axis (0)
        return lax.sort(xs, dimension=0)
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, xs)

def test_sort_1d_error():
    rng = np.random.RandomState(0)
    xs = rng.randn(8).astype("float32")

    def f(xs):
        # This should fail - trying to sort the only dimension (scan axis)
        return lax.sort(xs, dimension=0)
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, xs)

def test_sort_multiple_arrays():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4).astype("float32")
    ys = rng.randn(6, 4).astype("float32")

    def f(inputs):
        xs, ys = inputs
        return lax.sort((xs, ys), dimension=1)
    test_util.check_scan(f, (xs, ys))

def test_sort_with_prefill():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4).astype("float32")
    prefill = rng.randn(3, 4).astype("float32")

    def f(xs):
        return lax.sort(xs, dimension=1)
    test_util.check_scan_with_prefill(f, xs, prefill)

def test_sort_other_axis():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4, 5).astype("float32")

    def f(xs):
        # Move scan axis from 0 to 1, sort axis 0 (was axis 1), then move back
        xs = jnp.moveaxis(xs, 0, 1)
        xs = lax.sort(xs, dimension=0)
        return jnp.moveaxis(xs, 1, 0)
    test_util.check_scan(f, xs)

def test_sort_is_stable_false():
    rng = np.random.RandomState(0)
    xs = rng.randn(6, 4).astype("float32")

    def f(xs):
        return lax.sort(xs, dimension=1, is_stable=False)
    test_util.check_scan(f, xs)

def test_sort_integer_values():
    rng = np.random.RandomState(0)
    xs = rng.randint(0, 10, size=(6, 4)).astype("int32")

    def f(xs):
        return lax.sort(xs, dimension=1)
    test_util.check_scan(f, xs)

def test_gather_basic():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 5).astype("float32")
    indices = jnp.array([[1], [2], [3]])

    def f(operand):
        # Gather from non-scan axis (axis 1), keeping scan axis intact
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(0, 2),
                collapsed_slice_dims=(1,),
                start_index_map=(1,)
            ),
            slice_sizes=(6, 1, 5)
        )
    test_util.check_scan(f, operand)

def test_gather_with_prefill():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 5).astype("float32")
    prefill = rng.randn(3, 4, 5).astype("float32")
    indices = jnp.array([[1], [2], [3]])

    def f(operand):
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(0, 2),
                collapsed_slice_dims=(1,),
                start_index_map=(1,)
            ),
            slice_sizes=(9, 1, 5)
        )
    test_util.check_scan_with_prefill(f, operand, prefill)

def test_gather_slice_non_scan_axis():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 8, 5).astype("float32")
    indices = jnp.array([[1], [2], [3]])

    def f(operand):
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(0, 2),
                collapsed_slice_dims=(1,),
                start_index_map=(1,)
            ),
            slice_sizes=(6, 1, 5)
        )
    test_util.check_scan(f, operand)

def test_gather_axis_error():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 5).astype("float32")
    indices = jnp.array([[0], [1], [2]])

    def f(operand):
        # This should fail - trying to gather along scan axis (0)
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(1, 2),
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            ),
            slice_sizes=(1, 4, 5)
        )
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, operand)

def test_gather_start_indices():
    rng = np.random.RandomState(0)
    operand = rng.randn(4, 5).astype("float32")
    indices = rng.randint(0, 4, size=(6, 1)).astype("int32")

    def f(indices):
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            ),
            slice_sizes=(1, 5)
        )
    test_util.check_scan(f, indices)

def test_gather_start_indices_with_prefill():
    rng = np.random.RandomState(0)
    operand = rng.randn(4, 5).astype("float32")
    indices = rng.randint(0, 4, size=(6, 1)).astype("int32")
    prefill_indices = rng.randint(0, 4, size=(2, 1)).astype("int32")

    def f(indices):
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            ),
            slice_sizes=(1, 5)
        )
    test_util.check_scan_with_prefill(f, indices, prefill_indices)

def test_gather_other_axis():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 8).astype("float32")
    indices = jnp.array([[1], [2], [3]])

    def f(operand):
        # Move scan axis from 0 to 2, gather from axis 1, then move back
        operand = jnp.moveaxis(operand, 0, 2)
        result = lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(0, 2),
                collapsed_slice_dims=(1,),
                start_index_map=(1,)
            ),
            slice_sizes=(4, 1, 6)
        )
        return jnp.moveaxis(result, 2, 0)
    test_util.check_scan(f, operand)

def test_gather_multiple_indices():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 8, 10).astype("float32")
    indices = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    def f(operand):
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(0,),
                collapsed_slice_dims=(1, 2),
                start_index_map=(1, 2)
            ),
            slice_sizes=(6, 1, 1)
        )
    test_util.check_scan(f, operand)

def test_gather_reordered_output_axis():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 8).astype("float32")
    indices = jnp.array([[1], [2], [3]])

    def f(operand):
        # Test case where output scan axis != input scan axis
        # Input scan axis is 0, but gather output will have scan axis at position 1
        # Use moveaxis to move it back to position 0
        result = lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(1, 2),
                collapsed_slice_dims=(1,),
                start_index_map=(1,)
            ),
            slice_sizes=(6, 1, 8)
        )
        return jnp.moveaxis(result, 1, 0)
    test_util.check_scan(f, operand)

def test_gather_operand_batching_dims_not_supported():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 5).astype("float32")
    # Indices shape (6, 2, 1) to match operand_batching_dims=(0,)
    indices = jnp.array([[[2], [3]], [[1], [4]], [[0], [2]], [[4], [1]], [[3], [0]], [[1], [2]]])

    def f(operand):
        # Test operand_batching_dims - should fail because batching is not supported
        return lax.gather(
            operand, indices,
            lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(1,),
                start_index_map=(1,),
                operand_batching_dims=(0,),
                start_indices_batching_dims=(0,)
            ),
            slice_sizes=(1, 1)
        )
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, operand)
