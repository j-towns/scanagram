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
from jax import jvp
from jax.extend.linear_util import wrap_init
from jax.api_util import debug_info

from scanagram import util
from scanagram.core import ScanConversionError, register_rule


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

def custom_scanagram_rule(
    inscanvars, *args_flat, call, rule, in_tree, out_tree
):
    argnums, axes = util.unzip2(inscanvars)
    args_flat = [ShapeDtypeStruct(a.shape, a.dtype) for a in args_flat]
    consts, arg = tree.unflatten(in_tree, args_flat)
    if not (set(argnums)
            <= set(range(len(tree.leaves(consts)), len(args_flat)))):
        raise ScanConversionError(
            "Scanning over a variable which is closed over in a function "
            "with the custom_scanagram decorator is not supported"
        )
    if set(argnums) != set(range(len(tree.leaves(consts)), len(args_flat))):
        raise ScanConversionError(
            "All input arrays to custom_scanagram-decorated function must be "
            "scanned over."
        )
    if not util.all_equal(axes):
        raise ScanConversionError(
            "All input arrays to custom_scanagram-decorated function must be "
            "scanned along the same axis."
        )
    InVarInfo = ScanInfo(axes[0])
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
        return carry_new, tree.leaves(out)
    out_info = [(n, out_info.axis) for n in range(out_tree.num_leaves)]
    return carry_init, body_fn_flat, out_info, []

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
