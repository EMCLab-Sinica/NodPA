import functools
import logging
import sys

import onnx
import onnx.numpy_helper
import numpy as np
import tensorly.decomposition
import tensorly.tucker_tensor

from utils import (
    THIS_DIR,
    add_zeros,
    get_attr,
    find_initializer,
    find_node_and_idx_by_output,
    infer_auto_pad,
)
from decomposition.common import (
    Context,
    add_new_nodes,
    find_rank,
    try_cp_rank,
)
from fine_tune import fine_tune

logger = logging.getLogger('decomposition.conv')

def tucker2_decomposition(ctx):
    sys.path.append(str(THIS_DIR.parent / 'decomposition' / 'pytorch-tensor-decompositions'))
    from VBMF import VBMF

    unfold_0 = tensorly.unfold(ctx.orig_weights_data, 0)
    unfold_1 = tensorly.unfold(ctx.orig_weights_data, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    for idx, rank in enumerate(ranks):
        rank = max(1, rank)
        if rank < ctx.orig_weights_data.shape[idx]:
            rank = (rank + 1) // 2 * 2
        else:
            rank = rank // 2 * 2
        ranks[idx] = rank
    logger.info('ranks=%r', ranks)
    roots = tensorly.decomposition.partial_tucker(ctx.orig_weights_data, modes=[0, 1], rank=ranks)

    new_nodes = []

    decomposed_model = onnx.ModelProto()
    decomposed_model.ParseFromString(ctx.model.SerializeToString())

    add_tucker2_filters(decomposed_model, ctx.node, roots, new_nodes)
    add_new_nodes(ctx, decomposed_model, new_nodes)

    decomposed_model = fine_tune(ctx.model, ctx.train_data, ctx.test_data, ctx.epoch, qat=False, config=ctx.config)

    return decomposed_model

def add_cp_filters(ctx, model, weights, factors, orig_weights_shape, new_nodes):
    assert np.all(weights == [1] * len(weights)), weights

    orig_inputs = ctx.node.input
    outputs = ctx.node.output

    orig_pads = infer_auto_pad(model, ctx.node)
    orig_strides = get_attr(ctx.node, 'strides')

    EXPANDED_DIMS = [
        [2, 3],
        [1, 3],
        [1, 2],
        [2, 3],
    ]
    DIMS = [1, 2, 3, 0]

    n_nodes = len(factors)

    augmented_factors = []
    for dim in orig_weights_shape:
        if dim == 1:
            augmented_factors.append(None)
        else:
            augmented_factors.append(factors.pop(0))

    node_idx = 0
    for _ in range(4):
        inputs = list(orig_inputs)
        dim = DIMS.pop(0)
        expand_dims = EXPANDED_DIMS.pop(0)
        if orig_weights_shape[dim] == 1:
            continue
        if node_idx != 0:
            inputs[0] = outputs[0] + f'_cp{node_idx-1}'
        inputs[1] = orig_inputs[1] + f'_cp{node_idx}'
        pads = [0] * 4
        strides = [1, 1]
        if dim in (2, 3):
            # pads are [x1_begin, x2_begin...x1_end, x2_end,...]
            for pad_dim in (dim-2, dim):
                pads[pad_dim] = orig_pads[pad_dim]
            strides[dim-2] = orig_strides[dim-2]
        data = augmented_factors[dim]
        if dim != 0:
            if len(inputs) == 3:
                inputs[2] = add_zeros(model, (np.shape(data)[1],))
            data = np.transpose(data)
        data = np.expand_dims(data, axis=expand_dims)
        new_nodes.append(onnx.helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[outputs[0] + f'_cp{node_idx}' if node_idx != n_nodes - 1 else outputs[0]],
            pads=pads,
            strides=strides,
            kernel_shape=data.shape[2:],
            group=len(weights) if dim in (2, 3) else 1,
        ))
        model.graph.initializer.append(onnx.helper.make_tensor(inputs[1], data_type=ctx.orig_weights.data_type, dims=data.shape, vals=data))
        node_idx += 1

def add_tucker2_filters(model, node, roots, new_nodes):
    core, factors = roots

    orig_inputs = node.input
    outputs = node.output

    orig_weights = find_initializer(model, orig_inputs[1])
    orig_pads = infer_auto_pad(model, node)
    orig_strides = get_attr(node, 'strides')

    EXPANDED_DIMS = [
        [2, 3],
        [],
        [2, 3],
    ]
    TRANSPOSE_AXES = [
        [1, 0],
        [0, 1, 2, 3],
        [0, 1],
    ]
    DIMS = [1, 2, 3, 0]

    weights = [factors[1], core, factors[0]]
    N_NODES = len(weights)
    for node_idx in range(N_NODES):
        inputs = list(orig_inputs)
        dim = DIMS[node_idx]
        if node_idx != 0:
            inputs[0] = outputs[0] + f'_tucker{node_idx-1}'
        inputs[1] = orig_inputs[1] + f'_tucker{node_idx}'
        pads = [0] * 4
        strides = [1, 1]
        if node_idx == 1:
            # pads are [x1_begin, x2_begin...x1_end, x2_end,...]
            pads = orig_pads
            strides = orig_strides
        if dim != 0:
            inputs = inputs[:2]
        data = weights[node_idx]
        data = np.transpose(data, axes=TRANSPOSE_AXES[node_idx])
        data = np.expand_dims(data, axis=EXPANDED_DIMS[node_idx])
        new_nodes.append(onnx.helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[outputs[0] + f'_tucker{node_idx}' if node_idx != N_NODES - 1 else outputs[0]],
            pads=pads,
            strides=strides,
            kernel_shape=data.shape[2:],
        ))
        model.graph.initializer.append(onnx.helper.make_tensor(inputs[1], data_type=orig_weights.data_type, dims=data.shape, vals=data))

def decompose_conv(model, orig_accuracy, node_output_name, train_data, test_data, epoch, config, decomposition_method):
    node, node_idx = find_node_and_idx_by_output(model.graph.node, node_output_name)

    orig_weights_name = node.input[1]
    orig_weights = find_initializer(model, orig_weights_name)
    assert orig_weights

    logger.info('Decomposing %s node %s', node.op_type, node.output[0])

    orig_weights_data = onnx.numpy_helper.to_array(orig_weights)
    orig_weights_data = np.reshape(orig_weights_data, orig_weights.dims)

    ctx = Context(
        model=model,
        node=node,
        node_idx=node_idx,
        orig_weights=orig_weights,
        orig_weights_data=orig_weights_data,
        train_data=train_data,
        test_data=test_data,
        orig_accuracy=orig_accuracy,
        epoch=epoch,
        config=config,
    )

    if decomposition_method == 'cp':
        shape = np.array(ctx.orig_weights_data.shape)
        shape = np.delete(shape, np.where(shape == 1))
        if len(shape) > 2:
            max_dim = max(shape) + 1
        else:
            max_dim = min(shape) + 1
        try_rank_func = functools.partial(try_cp_rank, ctx=ctx, add_cp_filters_func=add_cp_filters)
        decomposed_model = find_rank(ctx.orig_accuracy, max_dim, try_rank_func=try_rank_func)
    elif decomposition_method == 'tucker2':
        decomposed_model = tucker2_decomposition(ctx)

    return decomposed_model
