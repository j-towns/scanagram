from functools import partial
from typing import Any

from jax.extend.core import (
    ClosedJaxpr, Jaxpr, Primitive, Var, Literal, JaxprEqn
)
from jax.extend.core.primitives import custom_vjp_call_p
from jax.core import Atom, AbstractValue
from jax._src.pjit import pjit_p

from scanagram.util import safe_map
from scanagram.custom_rules import custom_rules


map = safe_map

rules = {}

def register_rule(p: Primitive, rule):
    rules[p] = rule

###############################################################################
# This section is copied from jax._src.core

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

def last_used(jaxpr: Jaxpr) -> dict[Var, JaxprEqn | None]:
    """Returns a mapping from every var in jaxpr to what equation uses it
    last."""
    last_used: dict[Var, JaxprEqn | None] = {
        v: None for v in jaxpr.outvars if not isinstance(v, Literal)}
    for eqn in reversed(jaxpr.eqns):
        for v in eqn.invars:
            if not isinstance(v, Literal) and v not in last_used:
                last_used[v] = eqn
    return last_used

def clean_up_dead_vars(eqn: JaxprEqn, env: dict[Var, Any],
                       last_used: dict[Var, JaxprEqn | None]):
    """Remove all eqn.invars from env if eqn is the last time they were
    used."""
    for v in {v for v in eqn.invars if not isinstance(v, Literal)}:
        if last_used[v] is eqn:
            # Delete ref to variable when it is no longer needed by next
            # equations.
            del env[v]
###############################################################################

class Deleted:
    pass
deleted = Deleted()

def body_fn(closed_jaxpr: ClosedJaxpr, body_fns, scanvars, carry, xs):
    jaxpr = closed_jaxpr.jaxpr
    body_fns = list(reversed(body_fns))
    carry_old = list(reversed(carry))
    carry_new = []

    env = {}
    def read(v: Atom):
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val) -> None:
        env[v] = val

    map(write, jaxpr.constvars, closed_jaxpr.consts)
    map(write, jaxpr.invars, xs)
    lu = last_used(jaxpr)

    for e in jaxpr.eqns:
        subfuns, bind_params = e.primitive.get_bind_params(e.params)
        invals = map(read, e.invars)

        if any(type(v) is Var and v in scanvars for v in e.invars):
            carry_in, eqn_body_fn = carry_old.pop(), body_fns.pop()
            carry_out, ans = eqn_body_fn(carry_in, *invals)
            carry_new.append(carry_out)
        else:
            ans = e.primitive.bind(*subfuns, *invals, **bind_params)

        if e.primitive.multiple_results:
            map(write, e.outvars, ans)
        else:
            write(e.outvars[0], ans)
        clean_up_dead_vars(e, env, lu)
    return carry_new, map(read, jaxpr.outvars)

def check_outvars(outvars, scanvars):
    if any(o not in scanvars for o in outvars):
        # TODO: More detail here...
        raise ScanConversionError(
            "All of the outputs of the transformed function must be "
            "scanned over."
        )
    if any(scanvars[o][0] != 0 for o in outvars):
        # TODO: ...and here.
        raise ScanConversionError(
            "All outputs of the transformed function must be scanned over "
            "axis 0."
        )
    if any(scanvars[o][1] != 1 for o in outvars):
        # TODO: ...and here.
        raise ScanConversionError(
            "All outputs of the transformed function must not be "
            "strided/scaled along the scanned axis."
        )


def make_carry_init(closed_jaxpr: ClosedJaxpr, inscanvars=None):
    top_level = inscanvars is None
    jaxpr = closed_jaxpr.jaxpr
    carry_init = []
    eqn_body_fns = []

    env = {}

    def write(v: Var, val: Any) -> None:
        env[v] = val

    def maybe_read(v: Atom) -> Any:
        if isinstance(v, Literal):
            return v.val
        elif v in env:
            if isinstance(env[v], Deleted):
                raise ScanConversionError(
                    "Using scan carry output is not supported"
                )
            else:
                return env[v]
        else:
            return v.aval
    map(write, jaxpr.constvars, closed_jaxpr.consts)

    # Map from Var to scan axis
    inscanvars = inscanvars or [(n, 0, 1) for n in range(len(jaxpr.invars))]
    scanvars = {jaxpr.invars[n]: (a, s) for n, a, s in inscanvars}
    for e in jaxpr.eqns:
        inscanvars = [
            (i, *scanvars[v]) for i, v in enumerate(e.invars)
            if type(v) is Var and v in scanvars
        ]
        in_vals = map(maybe_read, e.invars)
        if inscanvars:
            # TODO: Raise NotImplementedError if rule isn't defined
            init, eqn_body_fn, outscanvars, to_delete = (
                rules[e.primitive](inscanvars, *in_vals, **e.params)
            )
            to_delete = [e.outvars[i] for i in to_delete]
            map(write, to_delete, len(to_delete) * [deleted])
            scanvars.update((e.outvars[i], (a, s)) for i, a, s in outscanvars)
            carry_init.append(init)
            eqn_body_fns.append(eqn_body_fn)
        elif not any(isinstance(v, AbstractValue) for v in in_vals):
            subfuns, bind_params = e.primitive.get_bind_params(e.params)
            ans = e.primitive.bind(*subfuns, *in_vals, **bind_params)
            if e.primitive.multiple_results:
                map(write, e.outvars, ans)
            else:
                write(e.outvars[0], ans)

    if top_level:
        check_outvars(jaxpr.outvars, scanvars)
        return eqn_body_fns, set(scanvars), carry_init
    else:
        outscanvars = tuple(
            (i,) + scanvars[v] for i, v in enumerate(jaxpr.outvars)
        )
        return eqn_body_fns, set(scanvars), carry_init, outscanvars

def make_scan(closed_jaxpr: ClosedJaxpr):
    eqn_body_fns, scanvars, carry_init = make_carry_init(closed_jaxpr)
    return partial(body_fn, closed_jaxpr, eqn_body_fns, scanvars), carry_init

class ScanConversionError(Exception):
    pass

def call_rule(inscanvars, jaxpr, *args):
    body_fns, scanvars, carry_init, outscanvars = make_carry_init(
        jaxpr, inscanvars
    )
    def body_fn_(carry, *args):
        return body_fn(jaxpr, body_fns, scanvars, carry, args)
    return carry_init, body_fn_, outscanvars, []

def pjit_rule(
    inscanvars, *args, jaxpr, in_shardings, out_shardings, in_layouts,
    out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline,
    compiler_options_kvs,
):
    # TODO: Figure out how to handle (non-default cases of) all the params
    if name.startswith("custom_scanagram_"):
        name = name[len("custom_scanagram_"):]
        if name in custom_rules:
            return custom_rules[name](inscanvars, *args)
    return call_rule(inscanvars, jaxpr, *args)
register_rule(pjit_p, pjit_rule)

def custom_vjp_call_rule(
    inscanvars, *args, call_jaxpr, fwd_jaxpr_thunk, num_consts, bwd, out_trees,
    symbolic_zeros
):
    # TODO: Maybe warn the user of undefined behavour if you take the gradient
    # of this?
    return call_rule(inscanvars, call_jaxpr, *args)
register_rule(custom_vjp_call_p, custom_vjp_call_rule)
