from functools import partial
from itertools import repeat
import numpy as np
from jax import numpy as jnp, lax
from jax import tree
from jax.extend.core import jaxpr_as_fun
from jax.extend.core import primitives
from jax._src.pjit import pjit_p
from scanagram.util import safe_map, unzip3, safe_zip, all_equal

from scanagram.core import register_rule, ScanConversionError
from scanagram import core
from scanagram.core import ScanInfo

map = safe_map
zip = safe_zip


def unzip_scanvars(scanvars):
    argnums = []
    axes = []
    strides = []
    prefills = []
    for n, s in scanvars:
        argnums.append(n)
        axes.append(s.axis)
        strides.append(s.stride)
        prefills.append(s.prefill)
    return argnums, axes, strides, prefills

# Used when scanning along a batch dimension of op
def batch_rule(op, inscanvars, *in_avals, **bind_params):
    assert not op.multiple_results
    argnums, axes, strides, prefills = unzip_scanvars(inscanvars)
    assert all_equal(axes)
    assert all_equal(strides)
    axis, stride = axes[0], strides[0]
    assert all_equal(p.shape[axis] for p in prefills)
    prefill_len = prefills[0].shape[axis]
    assert not prefill_len % stride
    prefills_map = dict(zip(argnums, prefills))
    prefills = [
        prefills_map[n] if n in prefills_map
        else lax.slice_in_dim(a, 0, prefill_len)
        for n, a in enumerate(in_avals)
    ]
    out_prefill = op.bind(*prefills, **bind_params)
    carry_init = None

    def body_fn(carry, *args):
        assert carry is None
        args = list(args)
        for argnum, axis in zip(argnums, axes):
            args[argnum] = jnp.expand_dims(args[argnum], axis)
        ans = op.bind(*args, **bind_params)
        return None, lax.squeeze(ans, [axis])
    return carry_init, body_fn, [(0, ScanInfo(axis, stride, out_prefill))], []

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
    argnums, axes, strides = unzip3(inscanvars)
    axis = axes[0]
    stride = strides[0]
    if not all(a == axis for a in axes[1:]):
        # TODO: more detail
        raise ScanConversionError(
            "All scanned inputs to nary op must be scanned along same axis"
        )
    if not all(s == stride for s in strides):
        raise ScanConversionError(
            "All scanned inputs to nary op must have same stride/scale"
        )
    if (any(avals[n].shape[axis] == 1 for n in argnums)
            and any(a.shape[axis] > 1 for a in avals)):
        # TODO: more detail
        raise ScanConversionError(
            "Broadcasting scanned variable along scanned axis is not "
            "supported"
        )
    if all(a.ndim == 0 or a.shape[axis] == 1
           for i, a in enumerate(avals) if i not in argnums):
        return batch_rule(op, inscanvars, *avals, **kwargs)
    init = 0
    def body_fn(counter, *args):
        args = [
            a if i in argnums else lax.dynamic_index_in_dim(
                a, counter // stride, axis, False
            ) for i, a in enumerate(args)
        ]
        ans = op.bind(*args, **kwargs)
        return counter + 1, ans
    return init, body_fn, [(0, axis, stride)], []

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
    _, [inscan_axis], _, _ = unzip_scanvars(inscanvars)
    if inscan_axis in set(axes):
        raise ScanConversionError(
            "Global scan operating along reduce axis is not supported."
        )
    return batch_rule(op, inscanvars, xs_aval, axes=axes)
for op in reduce_ops:
    register_rule(op, partial(reduce_rule, op))

def scan_rule(
    inscanvars, *xs_avals, _split_transpose, jaxpr, length, linear, num_carry,
    num_consts, reverse, unroll
):
    scanvar_argnums, scanvar_axes, scanvar_strides = unzip3(inscanvars)
    xs_argnums = set(range(num_consts + num_carry, len(xs_avals)))
    if not set(scanvar_argnums) <= xs_argnums:
        raise ScanConversionError(
            "Global scan along an axis of a constant or carry in a call to "
            "scan. This is not currently supported, but could be in future."
        )
    if not all(a == 0 for a in scanvar_axes):
        # TODO: Make this error more specific
        raise ScanConversionError(
            "Mismatch between global scan axis and scan axis"
        )
    if not all_equal(scanvar_strides):
        raise ScanConversionError("Scan input strides must match.")
    stride = scanvar_strides[0]
    consts, carry = (
        xs_avals[:num_consts],
        xs_avals[num_consts:num_consts + num_carry],
    )
    def body_fun(i_and_carry, *args):
        i, old_carry = i_and_carry
        args = tuple(
            a if n in scanvar_argnums else
            lax.dynamic_index_in_dim(a, i // stride, keepdims=False)
            for n, a in enumerate(args)
        )
        new_carry_and_x = jaxpr_as_fun(jaxpr)(*(
            consts + old_carry + args[num_consts + num_carry:]
        ))
        carry = tree.map(
            partial(lax.select, i % stride),
            old_carry,
            tuple(new_carry_and_x[:num_carry]),
        )
        return (i + 1, carry), new_carry_and_x

    out_scanvars = zip(
        range(num_carry, len(jaxpr.out_avals)),
        repeat(0, len(jaxpr.out_avals) - num_carry),
        repeat(stride, len(jaxpr.out_avals) - num_carry),
    )
    out_to_delete = list(range(num_carry))
    return (0, carry), body_fun, out_scanvars, out_to_delete
register_rule(lax.scan_p, scan_rule)

def broadcast_in_dim_rule(
    inscanvars, operand, shape, broadcast_dimensions, sharding
):
    [(_, inscan_axis, stride)] = inscanvars
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
    def body_fn(carry, x):
        assert carry is None
        return None, jnp.squeeze(lax.broadcast_in_dim_p.bind(
                jnp.expand_dims(x, inscan_axis), shape=shape,
                broadcast_dimensions=broadcast_dimensions, sharding=sharding
            ), out_axis)
    return None, body_fn, [(0, out_axis, stride)], []
register_rule(lax.broadcast_in_dim_p, broadcast_in_dim_rule)

def _perm_inverse(p):
    p = np.asarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(len(p))
    return s

def transpose_rule(inscanvars, operand, permutation):
    [(argnum, in_axis, in_stride)] = inscanvars
    assert argnum == 0
    out_axis = _perm_inverse(permutation)[in_axis]
    def body_fn(carry, x):
        return None, lax.squeeze(
            lax.transpose(
                jnp.expand_dims(x, in_axis), permutation
            ), [out_axis]
        )
    return None, body_fn, [(0, out_axis, in_stride)], []
register_rule(lax.transpose_p, transpose_rule)

def conv_general_dilated_rule(
    inscanvars, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, batch_group_count, precision,
    preferred_element_type
):
    inscan_argnums, inscan_axes, inscan_strides = unzip3(inscanvars)
    if 1 in inscan_argnums:
        raise ScanConversionError(
            "Global scan is not currently supported over rhs of "
            "conv_general_dilated."
        )
    [inscan_axis] = inscan_axes
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
    [inscan_stride] = inscan_strides
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
    length = lhs.shape[inscan_axis]
    if length % window_stride:
        # TODO: Consider relaxing this constraint
        raise ScanConversionError(
            "Output scanned axis size of strided conv must exactly divide "
            "input scanned axis size."
        )
    if inscan_stride % lhs_dilation:
        raise ScanConversionError(
            "Conv lhs_dilation must exactly divide input stride along scanned "
            "axis."
        )
    outscan_stride = (inscan_stride * window_stride) // lhs_dilation
    carry_shape = list(lhs.shape)
    carry_shape[inscan_axis] = rhs_dilation * (window_size - 1)
    carry_init = 0, jnp.zeros(carry_shape, lhs.dtype)
    def body_fn(i_and_carry, x, rhs):
        i, carry = i_and_carry
        lhs = lax.concatenate(
            [
                carry,
                jnp.expand_dims(lax.select(
                    i % inscan_stride,
                    jnp.zeros_like(x),
                    x,
                ), inscan_axis)
            ], inscan_axis
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
        carry_new = lax.select(
            i % (inscan_stride // lhs_dilation),
            carry,
            carry_new,
        )
        return (i + 1, carry_new), out
    return carry_init, body_fn, [(0, outscan_axis, outscan_stride)], []
register_rule(lax.conv_general_dilated_p, conv_general_dilated_rule)

def slice_rule(
   inscanvars, operand, start_indices, limit_indices, strides
):
    _, [in_axis], [in_stride], [prefill] = unzip_scanvars(inscanvars)
    if not (start_indices[in_axis] in {0, np.shape(prefill)[in_axis]}
            and limit_indices[in_axis] == operand.shape[in_axis]):
        raise ScanConversionError(
            "Slice along the scanned axis can only be used to remove prefill."
        )
    if strides is not None and operand.shape[in_axis] % strides[in_axis]:
        # TODO: Consider relaxing this constraint
        raise ScanConversionError(
            "Strided slice along scan axis must have a stride which exactly "
            "exactly divides the input axis size."
        )
    if strides is not None and prefill.shape[in_axis] % strides[in_axis]:
        raise ScanConversionError(
            "Strided slice along scan axis must have stride which exactly "
            "divides the prefill axis size."
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

    out_stride = in_stride * (strides[in_axis] if strides is not None else 1)
    return None, body_fn, [(0, ScanInfo(in_axis, in_stride * out_stride,
                                        out_prefill))], []
register_rule(lax.slice_p, slice_rule)

def pad_rule(
    inscanvars, operand, padding_value, padding_config
):
    assert len(inscanvars) == 1
    [(argnum, axis, in_stride)] = inscanvars
    assert argnum == 0  # Shouldn't be possible to scan over scalar
                        # padding_value
    scan_pad_start, scan_pad_end, scan_pad_interior = padding_config[axis]
    if not scan_pad_start == 0:
        raise ScanConversionError(
            "Padding at the beginning of a scanned axis is not yet "
            "supported"
        )
    if not scan_pad_end == scan_pad_interior:
        raise ScanConversionError(
            "End padding on scanned axis must be equal to interior padding"
        )
    dilation = scan_pad_interior + 1
    if in_stride % dilation:
        raise ScanConversionError(
            "Pad dilation must exactly divide the input stride along scanned "
            "axis"
        )
    out_stride = in_stride // dilation
    padding_config_ = list(padding_config)
    padding_config_.pop(axis)
    def body_fn(i, operand, padding_value):
        ans = lax.pad(operand, padding_value, padding_config_)
        return i + 1, lax.select(
            i % in_stride,
            jnp.full_like(ans, padding_value),
            ans,
        )
    return 0, body_fn, [(0, axis, out_stride)], []
register_rule(lax.pad_p, pad_rule)

def concatenate_rule(inscanvars, *operands, dimension):
    argnums, axes, instrides, prefills = unzip_scanvars(inscanvars)
    if not all_equal(axes):
        raise ScanConversionError(
            "All scanned arguments to concatenate must be scanned along the "
            "same axis."
        )
    axis = axes[0]
    if axis == dimension:
        if (len(operands) == 2 and tuple(argnums) == (1,)
                and np.size(prefills[0]) == 0 and instrides[0] == 1):
            out_prefill = operands[0]
            def body_fn(carry, prefill_, x):
                assert carry is None
                return None, x
            return None, body_fn, [(0, ScanInfo(axis, 1, out_prefill))], []
        else:
            raise ScanConversionError(
                "Global scan along concatenation dimension is only supported "
                "for setting up prefill."
            )
    assert all_equal(instrides) # This should be guaranteed now, because the
                                # shapes must match along axis
    instride = instrides[0]
    carry_init = 0
    def body_fn(i, *operands):
        operands = [
            jnp.expand_dims(
                o if n in argnums else lax.dynamic_index_in_dim(
                    o, i // instride, axis, False
                ), axis
            )
            for n, o in enumerate(operands)
        ]
        ans = jnp.squeeze(
            lax.concatenate_p.bind(*operands, dimension=dimension),
            axis,
        )
        return i + 1, ans
    return carry_init, body_fn, [(0, axis, instride)], []
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
    argnums, axes, strides = unzip3(inscanvars)
    if len(argnums) == 2:  # Option (1)
        assert argnums == (0, 1)
        lhs_axis, rhs_axis = axes
        stride, rhs_stride = strides
        if out_sharding is not None:
            # TODO: Check if it's ok to relax this
            raise ScanConversionError(
                "Out sharding is not yet supported in dot_general"
            )
        if rhs_stride != stride:
            raise ScanConversionError(
                "Strides for lhs and rhs inputs to dot_general must match."
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
        def body_fn(carry, lhs, rhs):
            assert carry is None
            ans = jnp.squeeze(lax.dot_general_p.bind(
                jnp.expand_dims(lhs, lhs_axis), jnp.expand_dims(rhs, rhs_axis),
                dimension_numbers=dimension_numbers, precision=precision,
                preferred_element_type=preferred_element_type,
                out_sharding=out_sharding,
            ), out_axis)
            return None, ans
        return None, body_fn, [(0, out_axis, stride)], []
    [argnum], [axis], [stride] = argnums, axes, strides
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
        def body_fn(i, lhs, rhs):
            if axis in lhs_batch_axes:
                rhs_axis = rhs_batch_axes[lhs_batch_axes.index(axis)]
                rhs = lax.dynamic_index_in_dim(
                    rhs, i // stride, rhs_axis, keepdims=True
                )
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
        def body_fn(i, lhs, rhs):
            if axis in rhs_batch_axes:
                lhs_axis = lhs_batch_axes[rhs_batch_axes.index(axis)]
                lhs = lax.dynamic_index_in_dim(
                    lhs, i // stride, lhs_axis, keepdims=True
                )
            ans = jnp.squeeze(lax.dot_general_p.bind(
                lhs, jnp.expand_dims(rhs, axis),
                dimension_numbers=dimension_numbers, precision=precision,
                preferred_element_type=preferred_element_type,
                out_sharding=out_sharding,
            ), out_axis)
            return i + 1, ans
    return 0, body_fn, [(0, out_axis, stride)], []
register_rule(lax.dot_general_p, dot_general_rule)

def reshape_rule(
    inscanvars, operand, *dyn_shape, new_sizes, dimensions, sharding
):
    if dyn_shape:
        raise ScanConversionError(
            "Dyanamic shapes in reshape are not currently supported in "
            "scan conversion."
        )
    [(argnum, axis, stride)] = inscanvars
    if sharding is not None:
        raise ScanConversionError(
            "Sharding is currently not supported in scan conversion of "
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
    new_sizes = list(new_sizes)
    new_sizes.pop(a)
    new_sizes = tuple(new_sizes)
    dimensions = list(dimensions)
    dimensions.remove(axis)
    dimensions = [d - 1 if d > axis else d for d in dimensions]
    dimensions = tuple(dimensions)
    def body_fn(carry, x):
        assert carry is None
        return None, lax.reshape_p.bind(
            x, new_sizes=new_sizes, dimensions=dimensions, sharding=sharding
        )
    return None, body_fn, [(0, a, stride)], []
register_rule(lax.reshape_p, reshape_rule)

def split_rule(inscanvars, operand, sizes, axis):
    [(_, scan_axis, stride)] = inscanvars
    if scan_axis == axis:
        raise ScanConversionError(
            "Applying split along scanned axis is not supported."
        )
    axis_ = axis if axis < scan_axis else axis - 1
    def body_fn(carry, x):
        assert carry is None
        return None, lax.split(x, sizes, axis_)
    outscanvars = [(n, scan_axis, stride) for n in range(len(sizes))]
    return None, body_fn, outscanvars, []
register_rule(lax.split_p, split_rule)

def squeeze_rule(inscanvars, array, dimensions):
    [(_, axis, stride)] = inscanvars
    if axis in dimensions:
        # TODO: Maybe this is actually fine?
        raise ScanConversionError("Cannot squeeze scanned axis.")
    return batch_rule(lax.squeeze_p, inscanvars, array, dimensions=dimensions)
register_rule(lax.squeeze_p, squeeze_rule)

def call_rule(inscanvars, jaxpr, *args):
    body_fns, scanvars, carry_init, outscanvars = core.make_carry_init(
        jaxpr, inscanvars
    )
    def body_fn(carry, *args):
        return core.body_fn(jaxpr, body_fns, scanvars, carry, args)
    return carry_init, body_fn, outscanvars, []

def pjit_rule(
    inscanvars, *args, jaxpr, in_shardings, out_shardings, in_layouts,
    out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline,
    compiler_options_kvs,
):
    # TODO: Figure out how to handle (non-default cases of) all the params
    return call_rule(inscanvars, jaxpr, *args)
register_rule(pjit_p, pjit_rule)

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
