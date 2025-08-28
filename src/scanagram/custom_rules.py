from typing import Any
from dataclasses import dataclass
import functools
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
from scanagram import core


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

    def __init__(self, fun):
        functools.update_wrapper(self, fun)
        self.fun = fun

    def def_scanagram(self, rule):
        self.rule = rule
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
            in_tree=in_structure, out_tree=out_structure
        )
        return tree.unflatten(out_structure, out_flat)

@dataclass
class ScanInfo:
    axis: int
    prefill: Any = None

def custom_scanagram_rule(
    inscanvars, length, *prefills_flat, call, rule, in_tree, out_tree
):
    argnums, axes = util.unzip2(inscanvars)
    prefill_consts, prefill_arg = tree.unflatten(in_tree, prefills_flat)
    _, arg = tree.unflatten(in_tree, [
        ShapeDtypeStruct(a.shape, a.dtype) for a in call.in_avals
    ])
    num_consts = len(tree.leaves(prefill_consts))
    if not (set(argnums) <= set(range(num_consts, len(prefills_flat)))):
        raise ScanConversionError(
            "Scanning over a variable which is closed over in a function "
            "with the custom_scanagram decorator is not supported"
        )
    if argnums != tuple(range(num_consts, len(prefills_flat))):
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
        p.shape[axis] for n, p in enumerate(prefills_flat)
        if n in argnums
    )
    prefill_len = prefills_flat[argnums[0]].shape[axis]
    if prefill_len == 0:
        prefill_arg = None
    InVarInfo = ScanInfo(axes[0], prefill_arg)
    out_info, body_fn, carry_init = rule(InVarInfo, arg)
    def body_fn_flat(carry, *args_flat):
        _, arg = tree.unflatten(in_tree, args_flat)
        carry_new, out = body_fn(carry, arg)
        if not tree.structure(out) == out_tree:
            raise ScanConversionError(
                "Output of body_fn provided in custom_scanagram rule must "
                "have the same pytree structure as the output of the "
                "custom_scanagram-decorated function."
            )
        # TODO: more detailed type checking here
        return carry_new, tree.leaves(out)
    if out_info.prefill is None:
        out_prefill_shapes = map(list, (a.shape for a in call.out_avals))
        for i in range(len(out_prefill_shapes)):
            out_prefill_shapes[i][axis] = 0
        out_prefill_flat = [
            jnp.zeros(s, a.dtype)
            for s, a in zip(out_prefill_shapes, call.out_avals)
        ]
        out_info = [(n, out_info.axis) for n in range(len(call.out_avals))]
    else:
        if not tree.structure(out_info.prefill) == out_tree:
            raise ScanConversionError(
                "Output prefill from custom_scanagram rule must have same "
                "pytree structure as the output of the "
                "custom_scanagram-decorated function."
            )
        # TODO: more detailed type checking here
        out_prefill_flat, _ = tree.flatten(out_info.prefill)
        out_info = [(n, out_info.axis) for n in range(len(call.out_avals))]
    return carry_init, body_fn_flat, out_info, out_prefill_flat, []

def custom_scanagram_impl(*args, call, rule, in_tree, out_tree):
    del rule, in_tree, out_tree
    return jaxpr_as_fun(call)(*args)

def custom_scanagram_abstract_eval(*in_avals, call, **kwargs):
    return call.out_avals

def custom_scanagram_jvp(primals, tangents, *, call, rule, in_tree, out_tree):
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
