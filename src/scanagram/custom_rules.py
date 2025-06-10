from typing import Callable
import jax
from jax.stages import OutInfo
import jax.numpy as jnp
import numpy as np


custom_rules = {}


class custom_scanagram():
    fun: Callable
    rule: Callable | None = None
    name: str | None = None

    def __init__(self, fun):
        name = fun.__name__
        if name in custom_rules:
            raise ValueError(
                "Functions with custom scanagram rules must have distinct "
                f"names. Got two definitions with name {name}."
            )
        self.name = name
        fun.__name__ = "custom_scanagram_" + fun.__name__
        self.fun = jax.jit(fun)

    def defrule(self, rule):
        if self.name in custom_rules:
            raise ValueError(
                "Trying to define a custom scanagram rule which already "
                f"exists, with name {self.name}."
            )
        custom_rules[self.name] = rule
        return rule

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise NotImplementedError(
                "Keyword arguments are not yet supported in custom scanagram "
                "rules."
            )
        if not all(isinstance(a, (jnp.ndarray, np.ndarray)) for a in args):
            raise NotImplementedError(
                "Custom scanagram rules do not yet support pytree-valued "
                "inputs."
            )
        traced = self.fun.trace(*args)
        if (not isinstance(traced.out_info, tuple)
                or not all(isinstance(o, OutInfo) for o in traced.out_info)):
            raise NotImplementedError(
                "The output of a function with a custom scanagram rule must "
                "be a tuple of arrays."
            )
        return self.fun(*args)
