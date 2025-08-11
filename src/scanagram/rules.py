from functools import partial
import numpy as np
from jax import numpy as jnp, lax
from jax.extend.core import jaxpr_as_fun
from jax.extend.core import primitives
import jax._src.pjit
# Backward compatibility with jax < 0.7
jit_p = (
    jax._src.pjit.jit_p
    if hasattr(jax._src.pjit, "jit_p") else
    jax._src.pjit.pjit_p
)
from scanagram.util import safe_map, safe_zip, all_equal, unzip_scanvars

from scanagram.core import register_rule, ScanConversionError
from scanagram import core
from scanagram.core import ScanInfo

map = safe_map
zip = safe_zip


# Used when op is expressible as a (v)map over the scanned axis of all scanned
# input variables
def batch_rule(op, inscanvars, *in_avals, **bind_params):
    argnums, axes, prefills = unzip_scanvars(inscanvars)
    assert all_equal(axes)
    axis = axes[0]
    assert all_equal(p.shape[axis] for p in prefills)
    prefills_map = dict(zip(argnums, prefills))
    out_prefill = op.bind(
        *[prefills_map[n] if n in prefills_map
          else a for n, a in enumerate(in_avals)], **bind_params
    )
    carry_init = None

    def body_fn(carry, *args):
        assert carry is None
        args = list(args)
        for argnum, axis_val in zip(argnums, axes):
            args[argnum] = jnp.expand_dims(args[argnum], axis_val)
        ans = op.bind(*args, **bind_params)
        if op.multiple_results:
            return None, tuple(lax.squeeze(a, [axis]) for a in ans)
        else:
            return None, lax.squeeze(ans, [axis])

    if op.multiple_results:
        out_scanvars = [(i, ScanInfo(axis, pf)) for i, pf in enumerate(out_prefill)]
    else:
        out_scanvars = [(0, ScanInfo(axis, out_prefill))]

    return carry_init, body_fn, out_scanvars, []


nary_ops = [
    lax.abs_p,
    lax.acos_p,
    lax.acosh_p,
    lax.add_p,
    lax.and_p,
    lax.asin_p,
    lax.asinh_p,
    lax.atan_p,
    lax.atan2_p,
    lax.atanh_p,
    lax.bessel_i0e_p,
    lax.bessel_i1e_p,
    lax.cbrt_p,
    lax.ceil_p,
    lax.clz_p,
    lax.complex_p,
    lax.conj_p,
    lax.convert_element_type_p,
    lax.cos_p,
    lax.cosh_p,
    lax.digamma_p,
    lax.div_p,
    lax.eq_p,
    lax.erf_inv_p,
    lax.erf_p,
    lax.erfc_p,
    lax.exp_p,
    lax.exp2_p,
    lax.expm1_p,
    lax.floor_p,
    lax.ge_p,
    lax.gt_p,
    lax.igamma_grad_a_p,
    lax.igamma_p,
    lax.igammac_p,
    lax.imag_p,
    lax.integer_pow_p,
    lax.is_finite_p,
    lax.le_p,
    lax.lgamma_p,
    lax.log_p,
    lax.log1p_p,
    lax.logistic_p,
    lax.lt_p,
    lax.max_p,
    lax.min_p,
    lax.mul_p,
    lax.ne_p,
    lax.neg_p,
    lax.nextafter_p,
    lax.not_p,
    lax.or_p,
    lax.polygamma_p,
    lax.population_count_p,
    lax.pow_p,
    lax.real_p,
    lax.regularized_incomplete_beta_p,
    lax.rem_p,
    lax.round_p,
    lax.rsqrt_p,
    lax.select_n_p,
    lax.shift_left_p,
    lax.shift_right_arithmetic_p,
    lax.shift_right_logical_p,
    lax.sign_p,
    lax.sin_p,
    lax.sinh_p,
    lax.sqrt_p,
    lax.square_p,
    lax.sub_p,
    lax.tan_p,
    lax.tanh_p,
    lax.xor_p,
    lax.zeta_p,
]

def nary_op_rule(op, inscanvars, *avals, **kwargs):
    argnums, axes, prefills = unzip_scanvars(inscanvars)
    axis = axes[0]
    if not all(a == axis for a in axes[1:]):
        # TODO: more detail
        raise ScanConversionError(
            "All scanned inputs to nary op must be scanned along same axis"
        )
    if (any(avals[n].shape[axis] == 1 for n in argnums)
            and any(a.shape[axis] > 1 for a in avals)):
        # TODO: more detail
        raise ScanConversionError(
            "Broadcasting scanned variable along scanned axis is not "
            "supported"
        )
    if not all_equal(p.shape[axis] for p in prefills):
        raise ScanConversionError(
            f"Different length prefills encountered in {op}"
        )
    if all(a.ndim == 0 or a.shape[axis] == 1
           for i, a in enumerate(avals) if i not in argnums):
        return batch_rule(op, inscanvars, *avals, **kwargs)
    prefill_len = prefills[0].shape[axis]
    prefills_map = dict(zip(argnums, prefills))
    out_prefill = op.bind(
        *[prefills_map[n] if n in prefills_map
          else (lax.slice_in_dim(a, 0, prefill_len, axis=axis)
          if a.shape[axis] > 1
          else a) for n, a in enumerate(avals)], **kwargs
    )
    init = 0
    def body_fn(counter, *args):
        args = [
            a if i in argnums else lax.dynamic_index_in_dim(
                a, prefill_len + counter, axis, False
            ) for i, a in enumerate(args)
        ]
        ans = op.bind(*args, **kwargs)
        return counter + 1, ans
    return init, body_fn, [(0, ScanInfo(axis, out_prefill))], []

for op in nary_ops:
    register_rule(op, partial(nary_op_rule, op))

reduce_ops = [
    lax.argmax_p,
    lax.argmin_p,
    lax.reduce_and_p,
    lax.reduce_max_p,
    lax.reduce_min_p,
    lax.reduce_or_p,
    lax.reduce_prod_p,
    lax.reduce_sum_p,
    lax.reduce_xor_p,
]
def reduce_rule(op, inscanvars, xs_aval, axes):
    _, [inscan_axis], _ = unzip_scanvars(inscanvars)
    if inscan_axis in set(axes):
        raise ScanConversionError(
            "Global scan operating along reduce axis is not supported."
        )
    return batch_rule(op, inscanvars, xs_aval, axes=axes)
for op in reduce_ops:
    register_rule(op, partial(reduce_rule, op))

def reduce_p_rule(
        inscanvars, *operands_and_inits, computation, dimensions, jaxpr
):
    num_operands = len(operands_and_inits) // 2
    argnums, axes, prefills = unzip_scanvars(inscanvars)
    if any(a >= num_operands for a in argnums):
        raise ScanConversionError(
            "Global scan over init_values in reduce_p is not supported."
        )
    if any(a in dimensions for a in axes):
        raise ScanConversionError(
            "Global scan operating along reduce axis is not supported."
        )
    return batch_rule(lax.reduce_p, inscanvars, *operands_and_inits,
                      computation=computation, dimensions=dimensions, jaxpr=jaxpr)
register_rule(lax.reduce_p, reduce_p_rule)

def rev_rule(inscanvars, operand, dimensions):
    _, [inscan_axis], _ = unzip_scanvars(inscanvars)
    if inscan_axis in set(dimensions):
        raise ScanConversionError(
            "Global scan along a reversed axis is not supported."
        )
    return batch_rule(lax.rev_p, inscanvars, operand, dimensions=dimensions)
register_rule(lax.rev_p, rev_rule)

def sort_rule(inscanvars, *operands, dimension, is_stable, num_keys):
    argnums, axes, prefills = unzip_scanvars(inscanvars)
    if not all_equal(axes):
        raise ScanConversionError(
            "All scanned inputs to sort must be scanned along the same axis. "
            "Support for different scan axes may be added in future."
        )
    inscan_axis = axes[0]
    if inscan_axis == dimension:
        raise ScanConversionError(
            "Global scan along a sorted axis is not supported."
        )

    return batch_rule(
        lax.sort_p, inscanvars, *operands, dimension=dimension,
        is_stable=is_stable, num_keys=num_keys
    )
register_rule(lax.sort_p, sort_rule)

def scan_rule(
    inscanvars, *xs_avals, _split_transpose, jaxpr, length, linear, num_carry,
    num_consts, reverse, unroll
):
    argnums, axes, prefills = unzip_scanvars(inscanvars)
    xs_argnums = set(range(num_consts + num_carry, len(xs_avals)))
    if not set(argnums) <= xs_argnums:
        raise ScanConversionError(
            "Global scan along an axis of a constant or carry in a call to "
            "scan. This is not currently supported, but could be in future."
        )
    if not all(a == 0 for a in axes):
        # TODO: Make this error more specific
        raise ScanConversionError(
            "Mismatch between global scan axis and lax scan axis."
        )
    if not all_equal(p.shape[0] for p in prefills):
        raise ScanConversionError(
            "Different length prefills detected in lax scan."
        )

    prefill_len = prefills[0].shape[0]
    prefill_map = dict(zip(argnums, prefills))
    out_prefills = lax.scan_p.bind(
        *[prefill_map[n] if n in prefill_map else
          (a[:prefill_len] if n in xs_argnums else a)
          for n, a in enumerate(xs_avals)
          ], _split_transpose=_split_transpose, jaxpr=jaxpr,
        length=prefill_len, linear=linear, num_carry=num_carry,
        num_consts=num_consts, reverse=reverse, unroll=unroll
    )
    carry, out_prefills = (
        tuple(out_prefills[:num_carry]), out_prefills[num_carry:]
    )

    consts = xs_avals[:num_consts]
    def body_fun(i_and_carry, *args):
        i, old_carry = i_and_carry
        args = tuple(
            a if n in argnums else
            lax.dynamic_index_in_dim(a, prefill_len + i, keepdims=False)
            for n, a in enumerate(args)
        )
        new_carry_and_x = jaxpr_as_fun(jaxpr)(*(
            consts + old_carry + args[num_consts + num_carry:]
        ))
        carry = tuple(new_carry_and_x[:num_carry])
        return (i + 1, carry), new_carry_and_x

    out_scanvars = [
        (n, ScanInfo(0, p)) for n, p in zip(
            range(num_carry, len(jaxpr.out_avals)),
            out_prefills
        )
    ]
    out_to_delete = list(range(num_carry))
    return (0, carry), body_fun, out_scanvars, out_to_delete
register_rule(lax.scan_p, scan_rule)

def broadcast_in_dim_rule(
    inscanvars, operand, shape, broadcast_dimensions, sharding
):
    [_], [inscan_axis], [prefill] = unzip_scanvars(inscanvars)
    if sharding is not None:
        raise ScanConversionError(
            "Sharding in broadcast_in_dim not yet supported."
        )
    if (operand.shape[inscan_axis]
            < shape[broadcast_dimensions[inscan_axis]]):
        raise ScanConversionError(
            "Global scan along broadcasting axis is not supported."
        )
    shape = list(shape)
    shape[broadcast_dimensions[inscan_axis]] = 1
    shape = tuple(shape)

    out_axis = broadcast_dimensions[inscan_axis]
    out_prefill_shape = (
        shape[:broadcast_dimensions[inscan_axis]] +
        (prefill.shape[inscan_axis],)
        + shape[broadcast_dimensions[inscan_axis] + 1:]
    )
    out_prefill = lax.broadcast_in_dim_p.bind(
        prefill, shape=out_prefill_shape,
        broadcast_dimensions=broadcast_dimensions, sharding=sharding,
    )

    def body_fn(carry, x):
        assert carry is None
        return None, jnp.squeeze(lax.broadcast_in_dim_p.bind(
                jnp.expand_dims(x, inscan_axis), shape=shape,
                broadcast_dimensions=broadcast_dimensions, sharding=sharding
            ), out_axis)
    return None, body_fn, [(0, ScanInfo(out_axis, out_prefill))], []
register_rule(lax.broadcast_in_dim_p, broadcast_in_dim_rule)

def _perm_inverse(p):
    p = np.asarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(len(p))
    return s

def transpose_rule(inscanvars, operand, permutation):
    [argnum], [in_axis], [prefill] = unzip_scanvars(inscanvars)
    assert argnum == 0
    out_prefill = lax.transpose_p.bind(prefill, permutation=permutation)
    out_axis = _perm_inverse(permutation)[in_axis]
    def body_fn(carry, x):
        return None, lax.squeeze(
            lax.transpose(
                jnp.expand_dims(x, in_axis), permutation
            ), [out_axis]
        )
    return None, body_fn, [(0, ScanInfo(out_axis, out_prefill))], []
register_rule(lax.transpose_p, transpose_rule)

def conv_general_dilated_rule(
    inscanvars, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, batch_group_count, precision,
    preferred_element_type
):
    inscan_argnums, inscan_axes, prefills = unzip_scanvars(inscanvars)
    if 1 in inscan_argnums:
        raise ScanConversionError(
            "Global scan is not currently supported over rhs of "
            "conv_general_dilated."
        )
    [inscan_axis], [prefill] = inscan_axes, prefills
    if inscan_axis == dimension_numbers.lhs_spec[0]:
        if batch_group_count > 1:
            raise ScanConversionError(
                "Global scan is not yet supported over conv lhs batch axis "
                "with batch_group_count > 1."
            )
        return batch_rule(
            lax.conv_general_dilated_p, inscanvars, lhs, rhs,
            window_strides=window_strides, padding=padding,
            lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count, precision=precision,
            preferred_element_type=preferred_element_type
        )
    if lhs.ndim > 3:
        raise ScanConversionError(
            "Converting conv with spatial dimension > 1 not yet supported."
        )
    if inscan_axis == dimension_numbers.lhs_spec[1]:
        raise ScanConversionError(
            "Global scan along feature dimension of conv lhs is not "
            "supported."
        )
    assert inscan_axis == dimension_numbers.lhs_spec[2]
    outscan_axis = dimension_numbers.out_spec[2]
    window_stride, = window_strides
    rhs_dilation, = rhs_dilation
    lhs_dilation, = lhs_dilation
    window_size = rhs.shape[dimension_numbers.rhs_spec[2]]
    if padding != ((rhs_dilation * (window_size - 1), lhs_dilation - 1),):
        raise ScanConversionError(
            "Only causal padding is supported in conv."
        )
    if window_stride != 1:
        raise ScanConversionError(
            "Strided convolution along scanned axis is not currently "
            "supported."
        )
    out_prefill = lax.conv_general_dilated_p.bind(
        prefill, rhs, window_strides=window_strides, padding=padding,
        lhs_dilation=(lhs_dilation,), rhs_dilation=(rhs_dilation,),
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count, precision=precision,
        preferred_element_type=preferred_element_type,
    )
    prefill_len = prefill.shape[inscan_axis]

    carry_len = rhs_dilation * (window_size - 1)
    carry_zeros_shape = list(lhs.shape)
    carry_zeros_shape[inscan_axis] = max(0, carry_len - prefill_len)
    carry_init = (0, jnp.concatenate([
        jnp.zeros(carry_zeros_shape, lhs.dtype),
        lax.slice_in_dim(
            prefill, max(prefill_len - carry_len, 0), prefill_len,
            axis=inscan_axis
        )
    ], inscan_axis
    ))
    def body_fn(i_and_carry, x, rhs):
        i, carry = i_and_carry
        lhs = lax.concatenate(
            [carry, jnp.expand_dims(x, inscan_axis)], inscan_axis
        )
        out = lax.conv_general_dilated_p.bind(
            lhs, rhs, window_strides=(1,), padding=((0, 0),),
            lhs_dilation=(1,), rhs_dilation=(rhs_dilation,),
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count, precision=precision,
            preferred_element_type=preferred_element_type,
        )
        out = lax.squeeze(out, [outscan_axis])
        carry_new = lax.slice_in_dim(
            lhs, 1, lhs.shape[inscan_axis], 1, inscan_axis
        )
        return (i + 1, carry_new), out
    return carry_init, body_fn, [(0, ScanInfo(outscan_axis, out_prefill))], []
register_rule(lax.conv_general_dilated_p, conv_general_dilated_rule)

def slice_rule(
   inscanvars, operand, start_indices, limit_indices, strides
):
    _, [in_axis], [prefill] = unzip_scanvars(inscanvars)
    if not (start_indices[in_axis] in {0, np.shape(prefill)[in_axis]}
            and limit_indices[in_axis] == operand.shape[in_axis]):
        raise ScanConversionError(
            "Slice along the scanned axis can only be used to remove prefill."
        )
    if strides is not None and strides[in_axis] != 1:
        # TODO: Consider relaxing this constraint
        raise ScanConversionError(
            "Strided slice along scan axis is not supported."
        )

    prefill_limit_indices = list(limit_indices)
    prefill_limit_indices[in_axis] = min(
        limit_indices[in_axis], prefill.shape[in_axis]
    )
    out_prefill = lax.slice_p.bind(
        prefill, start_indices=start_indices,
        limit_indices=tuple(prefill_limit_indices),
        strides=strides
    )

    start_indices_ = list(start_indices)
    start_indices_.pop(in_axis)
    limit_indices_ = list(limit_indices)
    limit_indices_.pop(in_axis)
    if strides is not None:
        strides_ = list(strides)
        strides_.pop(in_axis)
    else:
        strides_ = None

    def body_fn(carry, operand):
        assert carry is None
        return carry, lax.slice(
            operand, start_indices_, limit_indices_, strides_
        )

    return None, body_fn, [(0, ScanInfo(in_axis, out_prefill))], []
register_rule(lax.slice_p, slice_rule)

def pad_rule(
    inscanvars, operand, padding_value, padding_config
):
    assert len(inscanvars) == 1
    [argnum], [axis], [prefill] = unzip_scanvars(inscanvars)
    assert argnum == 0  # Shouldn't be possible to scan over scalar
                        # padding_value
    scan_pad_start, scan_pad_end, scan_pad_interior = padding_config[axis]
    if any(padding_config[axis]):
        raise ScanConversionError(
            "Padding along scanned axis is not currently supported."
        )
    prefill_padding_config = (
        padding_config if prefill.shape[axis] > 0 else
        padding_config[:axis] + ((0, 0, 0),) + padding_config[axis + 1:]
    )
    out_prefill = lax.pad_p.bind(
        prefill, padding_value, padding_config=prefill_padding_config
    )
    padding_config_ = list(padding_config)
    padding_config_.pop(axis)
    def body_fn(i, operand, padding_value):
        return i + 1, lax.pad(operand, padding_value, padding_config_)
    return 0, body_fn, [(0, ScanInfo(axis, out_prefill))], []
register_rule(lax.pad_p, pad_rule)

def concatenate_rule(inscanvars, *operands, dimension):
    argnums, axes, prefills = unzip_scanvars(inscanvars)
    if not all_equal(axes):
        raise ScanConversionError(
            "All scanned arguments to concatenate must be scanned along the "
            "same axis."
        )
    axis = axes[0]
    if axis == dimension:
        if (len(operands) == 2 and tuple(argnums) == (1,)
                and np.size(prefills[0]) == 0):
            # Introducing prefill
            out_prefill = operands[0]
            def body_fn(carry, prefill_, x):
                assert carry is None
                return None, x
            return None, body_fn, [(0, ScanInfo(axis, out_prefill))], []
        else:
            raise ScanConversionError(
                "Global scan along concatenation dimension is only supported "
                "for introducing prefill."
            )
    if not all_equal(p.shape[axis] for p in prefills):
        raise ScanConversionError(
            "Different length prefills detected in concatenate."
        )
    prefill_len = prefills[0].shape[axis]
    prefill_map = dict(zip(argnums, prefills))
    out_prefill = lax.concatenate_p.bind(
        *[prefill_map[n] if n in prefill_map
          else lax.slice_in_dim(a, 0, prefill_len, axis=axis)
          for n, a in enumerate(operands)
          ], dimension=dimension,
    )
    carry_init = 0
    def body_fn(i, *operands):
        operands = [
            jnp.expand_dims(
                o if n in argnums else lax.dynamic_index_in_dim(
                    o, prefill_len + i, axis, False
                ), axis
            )
            for n, o in enumerate(operands)
        ]
        ans = jnp.squeeze(
            lax.concatenate_p.bind(*operands, dimension=dimension), axis,
        )
        return i + 1, ans
    return carry_init, body_fn, [(0, ScanInfo(axis, out_prefill))], []
register_rule(lax.concatenate_p, concatenate_rule)

def dot_general_rule(
    inscanvars, lhs, rhs, dimension_numbers, precision, preferred_element_type,
    out_sharding,
):
    ((lhs_contracting_axes, rhs_contracting_axes),
     (lhs_batch_axes, rhs_batch_axes)) = dimension_numbers
    # Two supported options:
    #  (1) scan axis along batch axis of both inputs
    #  (2) scan axis along batch/non-contracting axis of one input
    argnums, axes, prefills = unzip_scanvars(inscanvars)
    if len(argnums) == 2:  # Option (1)
        assert argnums == (0, 1)
        lhs_axis, rhs_axis = axes
        if out_sharding is not None:
            # TODO: Check if it's ok to relax this
            raise ScanConversionError(
                "Out sharding is not yet supported in dot_general"
            )
        if lhs_axis not in lhs_batch_axes or rhs_axis not in rhs_batch_axes:
            raise ScanConversionError(
                "When scanning over both inputs to dot_general, the scanned "
                "axes must both be batch axes."
            )
        if lhs_batch_axes.index(lhs_axis) != rhs_batch_axes.index(rhs_axis):
            raise ScanConversionError(
                "Scanning over two non-corresponding batch axis inputs to "
                "dot_general is not supported."
            )
        out_axis = lhs_batch_axes.index(lhs_axis)
        out_prefill = lax.dot_general_p.bind(
            *prefills, dimension_numbers=dimension_numbers,
            precision=precision, preferred_element_type=preferred_element_type,
            out_sharding=out_sharding
        )
        def body_fn(carry, lhs, rhs):
            assert carry is None
            ans = jnp.squeeze(lax.dot_general_p.bind(
                jnp.expand_dims(lhs, lhs_axis), jnp.expand_dims(rhs, rhs_axis),
                dimension_numbers=dimension_numbers, precision=precision,
                preferred_element_type=preferred_element_type,
                out_sharding=out_sharding,
            ), out_axis)
            return None, ans
        return None, body_fn, [(0, ScanInfo(out_axis, out_prefill))], []
    [argnum], [axis], [prefill] = argnums, axes, prefills
    prefill_len = prefill.shape[axis]
    if argnum == 0:  # Option (2)
        if axis in lhs_contracting_axes:
            raise ScanConversionError(
                "Scanning over a contracting axis in dot_general is not "
                "supported."
            )
        lhs_axes = list(range(lhs.ndim))
        for a in lhs_batch_axes + lhs_contracting_axes:
            lhs_axes.remove(a)
        out_axis = (
            lhs_batch_axes.index(axis) if axis in lhs_batch_axes else
            len(lhs_batch_axes) + lhs_axes.index(axis)
        )
        if axis in lhs_batch_axes:
            rhs_axis = rhs_batch_axes[lhs_batch_axes.index(axis)]
            rhs = lax.slice_in_dim(rhs, 0, prefill_len, axis=rhs_axis)
        out_prefill = lax.dot_general_p.bind(
            prefill, rhs, dimension_numbers=dimension_numbers,
            precision=precision, preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )
        def body_fn(i, lhs, rhs):
            if axis in lhs_batch_axes:
                rhs = lax.dynamic_index_in_dim(rhs, i, rhs_axis, keepdims=True)
            ans = jnp.squeeze(lax.dot_general_p.bind(
                jnp.expand_dims(lhs, axis), rhs,
                dimension_numbers=dimension_numbers, precision=precision,
                preferred_element_type=preferred_element_type,
                out_sharding=out_sharding,
            ), out_axis)
            return i + 1, ans
    else:  # Option (2)
        assert argnum == 1
        if axis in rhs_contracting_axes:
            raise ScanConversionError(
                "Scanning over a contracting axis in dot_general is not "
                "supported."
            )
        rhs_axes = list(range(rhs.ndim))
        for a in rhs_batch_axes + rhs_contracting_axes:
            rhs_axes.remove(a)
        out_axis = (
            rhs_batch_axes.index(axis) if axis in rhs_batch_axes else
            lhs.ndim - len(lhs_contracting_axes) + rhs_axes.index(axis)
        )

        if axis in rhs_batch_axes:
            lhs_axis = lhs_batch_axes[rhs_batch_axes.index(axis)]
            lhs = lax.slice_in_dim(lhs, 0, prefill_len, axis=lhs_axis)
        out_prefill = lax.dot_general_p.bind(
            lhs, prefill, dimension_numbers=dimension_numbers,
            precision=precision, preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )
        def body_fn(i, lhs, rhs):
            if axis in rhs_batch_axes:
                lhs = lax.dynamic_index_in_dim(
                    lhs, i, lhs_axis, keepdims=True
                )
            ans = jnp.squeeze(lax.dot_general_p.bind(
                lhs, jnp.expand_dims(rhs, axis),
                dimension_numbers=dimension_numbers, precision=precision,
                preferred_element_type=preferred_element_type,
                out_sharding=out_sharding,
            ), out_axis)
            return i + 1, ans
    return 0, body_fn, [(0, ScanInfo(out_axis, out_prefill))], []
register_rule(lax.dot_general_p, dot_general_rule)

def reshape_rule(
    inscanvars, operand, *dyn_shape, new_sizes, dimensions, sharding
):
    if dyn_shape:
        raise ScanConversionError(
            "Dyanamic shapes in reshape are not currently supported in "
            "scan conversion."
        )
    [_], [axis], [prefill] = unzip_scanvars(inscanvars)
    if sharding is not None:
        raise ScanConversionError(
            "Sharding is not currently supported in scan conversion of "
            "reshape."
        )
    if dimensions is None:
        dimensions = tuple(range(operand.ndim))
    pre_size = np.prod(
        np.take(operand.shape, dimensions)[:dimensions.index(axis)]
    )
    size = 1
    a = -1
    while size <= pre_size:
        a = a + 1
        size = size * new_sizes[a]
    if operand.shape[axis] != new_sizes[a]:
        raise ScanConversionError("Reshape must preserve scanned axis.")
    out_prefill_sizes = list(new_sizes)
    out_prefill_sizes[a] = prefill.shape[axis]
    out_prefill_sizes = tuple(out_prefill_sizes)
    new_sizes = list(new_sizes)
    new_sizes.pop(a)
    new_sizes = tuple(new_sizes)
    out_prefill = lax.reshape_p.bind(
        prefill, new_sizes=out_prefill_sizes, dimensions=dimensions,
        sharding=sharding
    )
    dimensions = list(dimensions)
    dimensions.remove(axis)
    dimensions = [d - 1 if d > axis else d for d in dimensions]
    dimensions = tuple(dimensions)
    def body_fn(carry, x):
        assert carry is None
        return None, lax.reshape_p.bind(
            x, new_sizes=new_sizes, dimensions=dimensions, sharding=sharding
        )
    return None, body_fn, [(0, ScanInfo(a, out_prefill))], []
register_rule(lax.reshape_p, reshape_rule)

def split_rule(inscanvars, operand, sizes, axis):
    [_], [scan_axis], [prefill] = unzip_scanvars(inscanvars)
    if scan_axis == axis:
        raise ScanConversionError(
            "Applying split along scanned axis is not supported."
        )
    out_prefills = lax.split_p.bind(prefill, sizes=sizes, axis=axis)
    axis_ = axis if axis < scan_axis else axis - 1
    def body_fn(carry, x):
        assert carry is None
        return None, lax.split(x, sizes, axis_)
    outscanvars = [
        (n, ScanInfo(scan_axis, p)) for n, p in enumerate(out_prefills)
    ]
    return None, body_fn, outscanvars, []
register_rule(lax.split_p, split_rule)

def squeeze_rule(inscanvars, array, dimensions):
    [(_, axis)] = inscanvars
    if axis in dimensions:
        # TODO: Maybe this is actually fine?
        raise ScanConversionError("Cannot squeeze scanned axis.")
    return batch_rule(lax.squeeze_p, inscanvars, array, dimensions=dimensions)
register_rule(lax.squeeze_p, squeeze_rule)

def call_rule(inscanvars, jaxpr, *args):
    body_fns, scanvars, carry_init, outscanvars = core.make_carry_init(
        jaxpr, inscanvars, args
    )
    def body_fn(carry, *args):
        return core.body_fn(jaxpr, body_fns, scanvars, carry, args)
    return carry_init, body_fn, outscanvars, []

def jit_rule(
    inscanvars, *args, jaxpr, in_shardings, out_shardings, in_layouts,
    out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline,
    compiler_options_kvs,
):
    # TODO: Figure out how to handle (non-default cases of) all the params
    return call_rule(inscanvars, jaxpr, *args)
register_rule(jit_p, jit_rule)

def custom_vjp_call_rule(inscanvars, *args, call_jaxpr, **_):
    # TODO: Maybe warn the user of undefined behavour if you take the gradient
    # of this?
    return call_rule(inscanvars, call_jaxpr, *args)
register_rule(primitives.custom_vjp_call_p, custom_vjp_call_rule)

def custom_jvp_call_rule(inscanvars, *args, call_jaxpr, **_):
    # TODO: Maybe warn the user of undefined behavour if you take the gradient
    # of this?
    return call_rule(inscanvars, call_jaxpr, *args)
register_rule(primitives.custom_jvp_call_p, custom_jvp_call_rule)

def gather_rule(
    inscanvars, operand, start_indices, dimension_numbers, slice_sizes,
    indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None
):
    if len(inscanvars) > 1:
        raise ScanConversionError(
            "gather with multiple scanned inputs is not supported"
        )

    [argnum], [axis], [prefill] = unzip_scanvars(inscanvars)

    if argnum == 1:
        if axis == start_indices.ndim - 1:
            raise ScanConversionError(
                "Scanning over the index vector dim (the last axis of "
                "start_indices) is not supported."
            )
        return batch_rule(
            lax.gather_p, inscanvars, operand, start_indices,
            dimension_numbers=dimension_numbers, slice_sizes=slice_sizes,
            indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
            mode=mode, fill_value=fill_value
        )

    start_index_map = dimension_numbers.start_index_map
    operand_batching_dims = dimension_numbers.operand_batching_dims

    if axis in start_index_map:
        # The scan axis is being dynamically indexed
        raise ScanConversionError(
            "gather with dynamic start_index along scanned axis is not "
            "supported"
        )

    if operand_batching_dims != ():
        raise ScanConversionError(
            "gather with operand_batching_dims is not currently supported"
        )

    # Ensure scan axis size is preserved
    if slice_sizes[axis] != operand.shape[axis]:
        raise ScanConversionError(
            "gather must preserve scan axis size: got "
            f"slice_sizes[{axis}]={slice_sizes[axis]} "
            f"but operand.shape[{axis}]={operand.shape[axis]}"
        )

    # Compute output prefill with adjusted slice_sizes
    prefill_len = prefill.shape[axis]
    prefill_slice_sizes = list(slice_sizes)
    prefill_slice_sizes[axis] = prefill_len

    out_prefill = lax.gather_p.bind(
        prefill, start_indices,
        dimension_numbers=dimension_numbers,
        slice_sizes=tuple(prefill_slice_sizes),
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices, mode=mode, fill_value=fill_value
    )

    # Calculate output scan axis
    collapsed_slice_dims = dimension_numbers.collapsed_slice_dims

    if axis in collapsed_slice_dims:
        raise ScanConversionError(
            "Scan axis cannot be in collapsed_slice_dims as it would be eliminated"
        )

    non_collapsed_dims = [
        d for d in range(operand.ndim) if d not in collapsed_slice_dims
    ]
    out_axis = dimension_numbers.offset_dims[non_collapsed_dims.index(axis)]

    carry_init = None

    def body_fn(carry, operand, start_indices_arg):
        assert carry is None
        operand_expanded = jnp.expand_dims(operand, axis)
        expanded_slice_sizes = list(slice_sizes)
        expanded_slice_sizes[axis] = 1
        ans = lax.gather_p.bind(
            operand_expanded, start_indices_arg,
            dimension_numbers=dimension_numbers,
            slice_sizes=tuple(expanded_slice_sizes),
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode, fill_value=fill_value
        )
        return None, lax.squeeze(ans, [out_axis])

    return carry_init, body_fn, [(0, ScanInfo(out_axis, out_prefill))], []

register_rule(lax.gather_p, gather_rule)
