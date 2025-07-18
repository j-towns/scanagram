# Scanagram
The backbone neural network in an autoregressive generative model must operate
in three modes:
 - **Training:** Evaluation on full sequences, usually with parallelization
   along the sequence axis.
 - **Prefill:** State initialization, possibly with an input prompt, at the
   beginning of inference.
 - **Inference:** Process a single element at-a-time, used for autoregressive
   generation.

An implementor can write a single neural network function which is
_polymorphic_, and infers, based on the shapes of its inputs, and/or other
flags, which mode to operate in. Code written in this way can be unclear, and
this begs the question, can we separate the three concerns?

One simple option is to write three distinct functions, one for each mode, and
then write tests to ensure consistency between them. This works, and might be
more readable, but maintainence becomes (even more) tedious.

_Here I'm trying out a different approach._ The implementor writes the training
mode, and the other two modes are inferred automatically using a JAX
transformation.

### What does Scanagram do?
Scanagram provides a single function, `as_scan`. The input to `as_scan` is a
function which is 'scan-like'. Roughly speaking, this means that along the
zero'th axis, data cannot flow backwards.  Functions/neural networks which
satisfy this property are often called 'causal'. A useful property of these
functions is that _they can be expressed in terms of JAX's
[`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
function_.

To be a little more precise, if `g` is causal, there exists a pair `(f, init)`
such that for all input sequences `xs`,
```python
jnp.all(g(xs) == lax.scan(f, init, xs)[1])
```
The `as_scan` function attempts to automatically infer a valid `(f, init)`
pair, for causal input `g`. As well as `g`, it also requires an example `xs`
with the correct pytree structure, shape(s) and dtype(s). For more technical
detail see [below](#how-does-it-work).

### Examples
How exactly can we use `as_scan` to automate the implementation of prefill and
inference?  Let's start with the pure inference case, with no prompt/prefill.
Assume we have a scan-like function `g`, already implemented and in scope. Then
training, state initialization and generation could look something like this:
```python
from functools import partial

import jax.numpy as jnp
import jax
import scanagram

######################################
# Assume g is defined somewhere here #
######################################

def shift(xs):
    return jnp.concat([jnp.zeros_like(xs, shape=(0, *xs.shape[1:])), xs[:-1]])

@jax.jit
def train_loss(xs):
    logits = g(shift(xs))
    return cross_entropy(xs, logits)

@partial(jax.jit, static_argnums=1)
def generate(rng, length):
    example_xs = jnp.zeros((length, *xs_shape), xs_dtype)
    f, carry_init = scanagram.as_scan(g, example_xs)
    rngs = jax.random.split(rng, length)

    def gen_step(carry_and_x, rng):
        carry, x = carry_and_x
        carry_new, logits = f(carry, x)
        x_new = random.categorical(rng, logits)
        return (carry_new, x_new), x_new

    _, xs = lax.scan(
        gen_step, (carry_init, jnp.zeros(xs_shape, xs_dtype)), rngs
    )
    return xs
```

In order to handle prefill/prompting, we need to wrap `g` in a function which
concatenates the prompt to the input, and slices off the unneeded section at
the beginning of the output. This wrapped version of `g`, which we'll call
`g_prompted`, is still scan-like and thus can be transformed by Scanagram:
```python
@partial(jax.jit, static_argnums=2)
def generate(rng, prompt, length):
    def g_prompted(xs):
        return g(jnp.concat([prompt, xs]))[len(prompt):]

    example_xs = jnp.zeros((length, *xs_shape), xs_dtype)
    f, carry_init = scanagram.as_scan(g_prompted, example_xs)
    rngs = jax.random.split(rng, length)

    def gen_step(carry_and_x, rng):
        carry, x = carry_and_x
        carry_new, logits = f(carry, x)
        x_new = random.categorical(rng, logits)
        return (carry_new, x_new), x_new

    _, xs = lax.scan(
        gen_step, (carry_init, jnp.zeros(xs_shape, xs_dtype)), rngs
    )
    return xs
```

### How does it work?
TODO

### Which operations are supported
TODO
