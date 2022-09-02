import functools
import logging

import onnx
import onnx.numpy_helper
import numpy as np

from utils import (
    add_zeros,
    find_initializer,
    find_node_and_idx_by_output,
)
from decomposition.common import (
    Context,
    find_rank,
    try_cp_rank,
)

logger = logging.getLogger('decomposition.fc')

def add_cp_filters(ctx, model, weights, factors, orig_weights_shape, new_nodes):
    orig_weights_name = ctx.node.input[1]
    orig_weights = find_initializer(model, orig_weights_name)

    for node_idx in range(2):
        data = factors[node_idx]
        if node_idx != 0:
            data = np.transpose(data)
        inputs = list(ctx.node.input)
        inputs[1] += f'_cp{node_idx}'
        dims = data.shape
        if node_idx == 1:
            inputs[0] = ctx.node.output[0] + f'_cp{node_idx - 1}'
            outputs = ctx.node.output
        if node_idx == 0 and len(inputs) == 3:
            inputs[2] = add_zeros(model, (dims[1],))
            outputs = [ctx.node.output[0] + f'_cp{node_idx}']
        new_nodes.append(onnx.helper.make_node(
            ctx.node.op_type, inputs=inputs, outputs=outputs,
        ))
        model.graph.initializer.append(onnx.helper.make_tensor(inputs[1], data_type=orig_weights.data_type, dims=dims, vals=data))

def decompose_gemm(model: onnx.ModelProto, orig_accuracy, node_output_name: str, train_data, test_data, epoch, config):
    node, node_idx = find_node_and_idx_by_output(model.graph.node, node_output_name)

    orig_weights_name = node.input[1]
    orig_weights = find_initializer(model, orig_weights_name)
    if not orig_weights:
        logger.warning('Skipping non-initializer weights {orig_weights_name}')
        return

    if orig_weights.dims[0] == 1 or orig_weights.dims[1] == 1:
        return

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

    max_dim = orig_weights.dims[1] + 1
    try_rank_func = functools.partial(try_cp_rank, ctx=ctx, add_cp_filters_func=add_cp_filters)
    decomposed_model = find_rank(ctx.orig_accuracy, max_dim, try_rank_func=try_rank_func)

    return decomposed_model
