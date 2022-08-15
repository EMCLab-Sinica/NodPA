import logging

import onnx
import onnx.numpy_helper
import numpy as np
import tensorly.decomposition

from utils import (
    add_zeros,
    find_initializer,
    find_node_and_idx_by_output,
)

logger = logging.getLogger('decomposition')

RECONSTRUCTION_ERROR_THRESHOLD = 0.5

def add_cp_filters(model, node, factors, new_nodes):
    orig_weights_name = node.input[1]
    orig_weights = find_initializer(model, orig_weights_name)

    for node_idx in range(2):
        data = factors[node_idx]
        if node_idx != 0:
            data = np.transpose(data)
        inputs = list(node.input)
        inputs[1] += f'_cp{node_idx}'
        dims = data.shape
        if node_idx == 1:
            inputs[0] = node.output[0] + f'_cp{node_idx - 1}'
            outputs = node.output
        if node_idx == 0 and len(inputs) == 3:
            inputs[2] = add_zeros(model, (dims[1],))
            outputs = [node.output[0] + f'_cp{node_idx}']
        new_nodes.append(onnx.helper.make_node(
            node.op_type, inputs=inputs, outputs=outputs,
        ))
        model.graph.initializer.append(onnx.helper.make_tensor(inputs[1], data_type=orig_weights.data_type, dims=dims, vals=data))

def decompose_gemm(model: onnx.ModelProto, node_output_name: str):
    node, node_idx = find_node_and_idx_by_output(model.graph.node, node_output_name)

    new_nodes = []
    orig_weights_name = node.input[1]
    orig_weights = find_initializer(model, orig_weights_name)
    if not orig_weights:
        logger.warning('Skipping non-initializer weights {orig_weights_name}')
        return

    if orig_weights.dims[0] == 1 or orig_weights.dims[1] == 1:
        return

    logger.info('Decomposing %s node %s', node.op_type, node.name)

    data = onnx.numpy_helper.to_array(orig_weights)
    data = np.reshape(data, orig_weights.dims)

    # Only try even ranks to avoid matrices with odd columns in GEMM
    for try_rank in range(2, orig_weights.dims[1]+1, 2):
        weights, factors = tensorly.decomposition.parafac(data, rank=try_rank)
        prod = tensorly.cp_to_tensor((weights, factors))
        rec_error = tensorly.norm(prod - data) / tensorly.norm(data)
        if rec_error < RECONSTRUCTION_ERROR_THRESHOLD:
            break

    add_cp_filters(model, node, factors, new_nodes)

    model.graph.node.remove(node)
    model.graph.initializer.remove(orig_weights)
    for new_node in new_nodes:
        model.graph.node.insert(node_idx, new_node)
        node_idx += 1
