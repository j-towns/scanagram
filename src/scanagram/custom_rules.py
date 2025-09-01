import functools
from functools import partial
from typing import Callable

from jax.extend.core import Primitive, ClosedJaxpr, jaxpr_as_fun, Jaxpr
from jax import make_jaxpr
from jax import tree
from jax import tree_util
from jax.interpreters import ad, mlir
from jax._src.interpreters import xla
from jax import ShapeDtypeStruct
from jax.extend.linear_util import wrap_init
from jax.api_util import debug_info
import jax.numpy as jnp

from scanagram import util
from scanagram.core import ScanConversionError, register_rule


# TODO: Different error type for the errors which pertain specifically to
# custom_scanagram and don't occur during conversion.

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

###############################################################################
# This section is copied from jax/_src/interpreters/partial_eval.py

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

def _convert_constvars_jaxpr(jaxpr):
    """Moves the constvars to the start of invars."""
    dbg = jaxpr.debug_info._replace(
        arg_names=("",) * len(jaxpr.constvars) + jaxpr.debug_info.arg_names)
    lifted_jaxpr = Jaxpr(constvars=(),
                         invars=jaxpr.constvars + jaxpr.invars,
                         outvars=jaxpr.outvars, eqns=jaxpr.eqns,
                         effects=jaxpr.effects, debug_info=dbg)
    return lifted_jaxpr

###############################################################################

class custom_scanagram:
    fun: Callable
    rule: Callable | None = None
    _use_prefill: bool | None = None

    def __init__(self, fun):
        functools.update_wrapper(self, fun)
        self.fun = fun

    def def_scanagram(self, rule):
        if self.rule is not None:
            raise ScanConversionError(
                "Cannot define a custom scanagram rule more than once."
            )
        self.rule = rule
        self._use_prefill = False
        return rule

    def def_scanagram_with_prefill(self, rule):
        if self.rule is not None:
            raise ScanConversionError(
                "Cannot define a custom scanagram rule more than once."
            )
        self.rule = rule
        self._use_prefill = True
        return rule

    def __call__(self, arg):
        arg_flat, in_structure = tree.flatten(arg)
        closed_jaxpr, out_shapes = make_jaxpr(
            self.fun, return_shape=True
        )(arg)
        out_structure = tree.structure(out_shapes)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts
        closed_call = ClosedJaxpr(_convert_constvars_jaxpr(jaxpr), ())
        in_structure = tree_util.treedef_tuple((
            tree.structure(consts), in_structure
        ))
        if self.rule is None:
            raise ValueError(
                "Function decorated with custom_scanagram must have a rule "
                "defined."
            )
        out_flat = custom_scanagram_p.bind(
            *consts, *arg_flat, call=closed_call, rule=self.rule,
            in_tree=in_structure, out_tree=out_structure,
            use_prefill=self._use_prefill
        )
        return tree.unflatten(out_structure, out_flat)

def empty_prefill(axis, aval):
    shape = list(aval.shape)
    shape[axis] = 0
    return jnp.zeros(shape, aval.dtype)

def custom_scanagram_rule(
    inscanvars, *avals_flat, call, rule, in_tree, out_tree, use_prefill
):
    argnums, axes = util.unzip2(inscanvars)
    assert avals_flat == tuple(call.in_avals)
    consts, arg = tree.unflatten(in_tree, [
        ShapeDtypeStruct(a.shape, a.dtype) for a in avals_flat
    ])
    num_consts = len(tree.leaves(consts))
    xs_argnums = tuple(range(num_consts, len(avals_flat)))
    if not set(argnums) <= set(xs_argnums):
        raise ScanConversionError(
            "Scanning over a variable which is closed over in a function "
            "with the custom_scanagram decorator is not supported."
        )
    if argnums != xs_argnums:
        raise ScanConversionError(
            "All input arrays to custom_scanagram-decorated function must be "
            "scanned over."
        )
    if not util.all_equal(axes):
        raise ScanConversionError(
            "All input arrays to custom_scanagram-decorated function must be "
            "scanned along the same axis."
        )
    axis = axes[0]
    assert util.all_equal(
        a.shape[axis] for n, a in enumerate(avals_flat) if n in argnums
    )
    if use_prefill:
        out_axis, init_fn, body_fn = rule(axis, arg)
    else:
        out_axis, body_fn, carry_init = rule(axis, arg)
    def init_fn_flat(*prefills_flat):
        if use_prefill:
            _, arg_prefill = tree.unflatten(in_tree, prefills_flat)
            carry_init_, out_prefill = init_fn(arg_prefill)
            out_prefill_flat, out_structure = tree.flatten(out_prefill)
            if not out_structure == out_tree:
                raise ScanConversionError(
                    "Output prefill from custom scanagram rule has a pytree "
                    "structure which doesn't match that of the custom_scanagram-"
                    "decorated function."
                )
        else:
            carry_init_ = carry_init
            out_prefill_flat = map(
                partial(empty_prefill, out_axis), call.out_avals
            )
        return carry_init_, out_prefill_flat
    def body_fn_flat(carry, *args_flat):
        _, arg = tree.unflatten(in_tree, args_flat)
        carry_new, out = body_fn(carry, arg)
        out_flat, out_structure = tree.flatten(out)
        if not out_structure == out_tree:
            raise ScanConversionError(
                "Output of body_fn provided in custom_scanagram rule must "
                "have the same pytree structure as the output of the "
                "custom_scanagram-decorated function."
            )
        # TODO: more detailed type checking here
        return carry_new, out_flat
    out_info = [(n, out_axis) for n in range(len(call.out_avals))]
    return out_info, [], init_fn_flat, body_fn_flat

def custom_scanagram_impl(*args, call, rule, in_tree, out_tree, use_prefill):
    del rule, in_tree, out_tree
    return jaxpr_as_fun(call)(*args)

def custom_scanagram_abstract_eval(*in_avals, call, **kwargs):
    return call.out_avals

def custom_scanagram_jvp(primals, tangents, *, call, rule, in_tree, out_tree,
                         use_prefill):
    jvp_fun = ad.jvp(wrap_init(
        jaxpr_as_fun(call),
        debug_info=debug_info('jvp', jaxpr_as_fun(call), primals, {})
    ))
    return jvp_fun.call_wrapped(primals, tangents)

custom_scanagram_p = Primitive("custom_scanagram_call")
custom_scanagram_p.multiple_results = True
custom_scanagram_p.def_impl(custom_scanagram_impl)
custom_scanagram_p.def_abstract_eval(custom_scanagram_abstract_eval)
ad.primitive_jvps[custom_scanagram_p] = custom_scanagram_jvp
xla.register_initial_style_primitive(custom_scanagram_p)
mlir.register_lowering(custom_scanagram_p, mlir.lower_fun(
    custom_scanagram_impl, multiple_results=True))
register_rule(custom_scanagram_p, custom_scanagram_rule)
