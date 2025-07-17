# Scanagram
The backbone neural network in an autoregressive generative model must operate in three modes:
 - **Training:** Evaluation on full sequences, usually with parallelization along the sequence axis.
 - **Prefill:** State initialization, possibly with an input prompt, at the beginning of inference.
 - **Inference:** Process a single element at-a-time, used for autoregressive generation.

An implementor can write a single neural network function which is _polymorphic_,
and infers, based on the shapes of its inputs, and/or other flags, which mode to operate in. Code
written in this way can be unclear, and this begs the question, can we separate the three
concerns?

One simple option is to write three distinct functions, one for each mode, and then write tests
to ensure consistency between them. This works, and might be more readable, but maintainence
becomes (even more) tedious.

_Here I'm trying out a different approach._ The implementor writes the training mode, and
the other two modes are inferred automatically using a JAX transformation.

### What does Scanagram do?
Scanagram provides a single function, `as_scan`. The input to `as_scan` is a function which is
'scan-like'. Roughly speaking, this means that along the zero'th axis, data cannot flow backwards.
Functions/neural networks which satisfy this property are often called 'causal'. A useful property
of these functions is that _they can be expressed in terms of JAX's
[`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) function_.

To be a little more precise, if `g` is causal, there exists a pair `(f, init)` such that
for all input sequences `xs`,
```python
jnp.all(g(xs) == lax.scan(f, init, xs)[1])
```
The `as_scan` function attempts to automatically infer a valid `(f, init)` pair, for causal input
`g`. It also requires an example `xs` with the correct pytree structure, shapes and dtypes, in order
to infer a jaxpr representation of `g`. For more detail see [below](#how-does-it-work).

### Examples
How can we use `as_scan` to automate the implementation of prefill and inference?

### How does it work?
TODO

### Which operations are supported
TODO
