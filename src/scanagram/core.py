from dataclasses import dataclass
from functools import partial
from typing import Any

from jax.extend.core import (
    ClosedJaxpr, Jaxpr, Primitive, Var, Literal, JaxprEqn
)
from jax.core import Atom, AbstractValue
from jax.tree_util import register_dataclass
import jax.numpy as jnp

from scanagram.util import safe_map, unzip2, all_equal


map = safe_map

rules = {}

def register_rule(p: Primitive, rule):
    rules[p] = rule

def default_prefill(aval):
    assert aval.ndim
    return jnp.zeros((0,) + aval.shape[1:], aval.dtype)

def typecheck_prefill(prefill, axis, aval):
    assert prefill.shape[:axis] == aval.shape[:axis]
    assert prefill.shape[axis] <= aval.shape[axis]
    assert prefill.shape[axis + 1:] == aval.shape[axis + 1:]
    assert prefill.dtype == aval.dtype

###############################################################################
# This section is copied from jax/_src/core.py

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

def check_outvars(outscanvars, num_outs):
    ns, axes = unzip2(outscanvars)
    if ns != tuple(range(num_outs)):
        # TODO: More detail here...
        raise ScanConversionError(
            "All of the outputs of the transformed function must be "
            "scanned over."
        )
    if axes != num_outs * (0,):
        # TODO: ...and here.
        raise ScanConversionError(
            "All outputs of the transformed function must be scanned over "
            "axis 0."
        )

def get_length(inscanvars, avals):
    lengths = [avals[n].shape[axis] for n, axis in inscanvars]
    # Should already be guaranteed by check_lengths
    assert all_equal(lengths)
    return lengths[0]

def make_carry_init(closed_jaxpr: ClosedJaxpr, inscanvars, args):
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
            return env[v]
        else:
            raise ScanConversionError(
                "Using or returning scan carry output is not supported."
            )

    map(write, jaxpr.constvars, closed_jaxpr.consts)
    map(write, jaxpr.invars, args)

    length = get_length(inscanvars, closed_jaxpr.in_avals)

    # Map from Var to scan axis
    scanvars = {jaxpr.invars[n]: a for n, a in inscanvars}
    for e in jaxpr.eqns:
        inscanvars = [
            (n, scanvars[v]) for n, v in enumerate(e.invars)
            if type(v) is Var and v in scanvars
        ]
        in_vals = map(maybe_read, e.invars)
        if inscanvars:
            # TODO: Raise NotImplementedError if rule isn't defined
            init, eqn_body_fn, outscanvars, outvals, to_delete = (
                rules[e.primitive](inscanvars, length, *in_vals, **e.params)
            )
            for n, v in enumerate(outvals):
                if n not in to_delete:
                    write(e.outvars[n], v)
            scanvars.update((e.outvars[n], a) for n, a in outscanvars)
            for n, axis in outscanvars:
                typecheck_prefill(outvals[n], axis, e.outvars[n].aval)
            carry_init.append(init)
            eqn_body_fns.append(eqn_body_fn)
        else:
            subfuns, bind_params = e.primitive.get_bind_params(e.params)
            ans = e.primitive.bind(*subfuns, *in_vals, **bind_params)
            if e.primitive.multiple_results:
                map(write, e.outvars, ans)
            else:
                write(e.outvars[0], ans)

    outscanvars = tuple(
        (i, scanvars[v]) for i, v in enumerate(jaxpr.outvars) if v in scanvars
    )
    outvals = map(maybe_read, jaxpr.outvars)
    return eqn_body_fns, set(scanvars), carry_init, outscanvars, outvals

def make_scan(closed_jaxpr: ClosedJaxpr, prefills=None):
    if prefills is None:
        prefills = map(default_prefill, closed_jaxpr.in_avals)
    inscanvars = [(n, 0) for n in range(len(closed_jaxpr.in_avals))]
    eqn_body_fns, scanvars, carry_init, outscanvars, outvals = make_carry_init(
        closed_jaxpr, inscanvars, prefills
    )
    check_outvars(outscanvars, len(closed_jaxpr.out_avals))
    return (
        partial(body_fn, closed_jaxpr, eqn_body_fns, scanvars), carry_init,
        outvals
    )

class ScanConversionError(Exception):
    pass
