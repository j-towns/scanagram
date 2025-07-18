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
flags, which mode to operate in. Code written in this way can be unclear and
tedious to maintain, and this begs the question, can we tidy things up by
separating the three concerns?

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
Scanagram's `as_scan` function is implemented using an _initial style JAX
transformation_. That means that it works by first tracing the scan-like input
function to a [jaxpr](https://docs.jax.dev/en/latest/jaxpr.html). This is an
internal language used by JAX, which can be easily interpreted since it is a
Python data structure.

JAX functions are composed from a set of _primitive_ functions. For
transformations like `grad` and `vmap`, the key is to define how each primitive
should be transformed (by writing a transformation rule for each one), and then
how to transform a whole function, using the rules for each primitive.

We can take the same approach for Scanagram—we define rules for converting
each primitive to a scan (where possible), and also an interpreter for the
jaxpr language which converts a whole function, applying the rules for each
primitive it encounters. Because JAX does a lot of the hard work, Scanagram
turns out to be pretty simple. The [core](src/scanagram/core.py), where the
interpreter lives, is currently only 200 lines of code.

Doing things in this way means that we need to assume not just that `g` is
scan-like, but also that each primitive used to evaluate `g` on its argument
`xs` is also scan-like. This might sound like a strong assumption, but actually
it's quite natural.

Let's formally re-iterate what we mean by 'scan-like'. We said above that a
function g is scan-like (or causal) if there exists an `f` such that for all
inputs xs, we have
```python
jnp.all(g(xs) == lax.scan(f, init, xs)[1])
```

Here is an equivalent formulation: g is scan-like if for all integer `t` and
for all input `xs`, we have
```python
jnp.all(g(xs)[:t] == g(xs[:t]))
```
You might take some convincing that these two properties really are equivalent.
For now I'll leave the proof to you as an exercise 😀. This second version is
convenient because the symmetry between the two sides of the equation is clear.
This symmetry can easily be used to show that if two functions `g1` and `g2`
are scan-like, then so is the composition `lambda xs: g1(g2(xs))` (again feel
free to work out a proof yourself if you want to).

All of this formal math basically tells us that being causal/scan-like is a
convenient property which respects function composition. If each layer in a
neural network is causal, then the overall network is guaranteed to be causal
too. Implementors of autoregressive models have long used this property
implicitly without needing to state or prove it formally.

### Which operations are supported?
Not all JAX primitives are scan-like. The main things that should be supported
are:
 - __Causal convolution__ Specifically, a call to `jax.lax.conv_general_dilated`
   which uses appropriate (causal) padding, with only 1 spatial dimension.
 - __Scan__ Obviously `scan` itself is scan-like!
 - Any operation without interaction along the sequence axis (some of these
   are still [TODO](https://github.com/j-towns/scanagram/issues/1)).

Although the input and output of `g` must scan along the 0'th axis (this is
to align the API with that of `jax.lax.scan`), within `g` the scan
axis can be moved to different positions using ops like `transpose` and
`moveaxis`.

### What about causal self-attention?
Causal self-attention is scan-like, but it isn't a JAX primitive, and it is
composed from primitives which are not causal. But don't panic! There is a way
to decorate a composite function like self-attention to tell Scanagram that the
composite _is_ causal, even if it is made from parts which are not. Once this
decorator has been added, a conversion rule can be manually defined.

The API for handling custom rules is inspired by JAX's API for [defining custom
derivatives](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).
I am planning to write a more detailed tutorial on how to use custom rules.
For now, a full example showing how to define a rule for masked multi-head
attention can be found in
[examples/masked\_self\_attention.py](examples/masked_self_attention.py), which
I hope can be adapted for your needs. In future it would be helpful to write a
library of commonly used self-attention functions and their scan conversion
rules, but for now I'm leaving it up to users to adapt the example code to
their use-case.

### Custom Scanagram rules
Sometimes we want to use a composite function which is scan-like, but built out
of JAX primitives which are not. The most common example is causal (AKA masked)
self-attention. 
## Installation
I haven't uploaded a package to PyPI yet, so for now the package can be
installed directly from Github using
```
pip install git+https://github.com/j-towns/scanagram.git
```

## Project status and plans
This project should be treated as experimental, with APIs likely to change and
features added and removed. I plan to do the following as soon as I have time:
 - More coverage of JAX ops (see
   [https://github.com/j-towns/scanagram/issues/1](https://github.com/j-towns/scanagram/issues/1).
 - A tutorial on how to define custom rules.
 - More examples, including end-to-end/realistic examples
 - Setup CI and packaging/deployment to PyPI

## Acknowledgements
This project would not have been possible without
[Julius Kunze](https://juliuskunze.com/). Julius helped me out a lot with [an
earlier attempt at solving the same
problem](https://github.com/j-towns/fastar), and provided helpful encouragement
and advice. My MSc students [Daniel](https://daniel-gallo.github.io/) and
[Konrad](https://github.com/konradszewczyk) also helped to test Scanagram on a
large-scale end-to-end model. Thanks also to [Jacob
Menick](https://github.com/jacobmenick) for encouragement.
