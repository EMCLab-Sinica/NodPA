import logging

import onnx
import onnx.numpy_helper
import numpy as np
import tensorly.decomposition
import tensorly.tucker_tensor

from utils import add_zeros, get_attr, find_initializer, infer_auto_pad

logger = logging.getLogger('decomposition')

RECONSTRUCTION_ERROR_THRESHOLD = 0.5

class Tucker2DecomposedFilter:
    def __init__(self, data):
        from VBMF import VBMF

        unfold_0 = tensorly.unfold(data, 0)
        unfold_1 = tensorly.unfold(data, 1)
        sigma2 = 0.02 # XXX: sigma2 estimation does not work!?
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0, sigma2=sigma2)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1, sigma2=sigma2)
        ranks = [diag_0.shape[0], diag_1.shape[1]]
        for idx, rank in enumerate(ranks):
            rank = max(1, rank)
            rank = (rank + 1) // 2 * 2
            ranks[idx] = rank
        print(f'ranks={ranks}')
        self.roots = tensorly.decomposition.partial_tucker(data, modes=[0, 1], rank=ranks)

class CPDecomposedFilter:
    def __init__(self, data):
        rank_lower = 2
        shape = np.array(data.shape)
        max_dim = max(np.delete(shape, np.where(shape == 1))) + 1
        rank_upper = 1
        while rank_upper <= max_dim:
            (weights, factors), rec_error = tensorly.decomposition.parafac(data, rank=rank_upper, return_errors=True)
            print(f'rank={rank_upper}, error={rec_error[-1]}')
            if rec_error[-1] < RECONSTRUCTION_ERROR_THRESHOLD:
                break
            rank_upper *= 2

        # Only try even ranks to avoid matrices with odd columns in GEMM
        while rank_lower + 2 < rank_upper:
            try_rank = ((rank_upper + rank_lower) // 2 + 1) // 2 * 2
            (weights, factors), rec_error = tensorly.decomposition.parafac(data, rank=try_rank, return_errors=True)
            print(f'rank={try_rank}, error={rec_error[-1]}')
            if rec_error[-1] < RECONSTRUCTION_ERROR_THRESHOLD:
                rank_upper = try_rank
            else:
                rank_lower = try_rank
        self.roots = (weights, factors)

def add_cp_filters(model, node, roots, new_nodes):
    weights, factors = roots

    print([factor.shape for factor in factors])

    assert np.all(weights == [1] * len(weights)), weights

    orig_inputs = node.input
    outputs = node.output

    orig_weights = find_initializer(model, orig_inputs[1])
    orig_pads = infer_auto_pad(model, node)
    orig_strides = get_attr(node, 'strides')

    EXPANDED_DIMS = [
        [2, 3],
        [1, 3],
        [1, 2],
        [2, 3],
    ]
    DIMS = [1, 2, 3, 0]

    for node_idx in range(4):
        inputs = list(orig_inputs)
        dim = DIMS[node_idx]
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
        data = factors[dim]
        if dim != 0:
            if len(inputs) == 3:
                inputs[2] = add_zeros(model, (np.shape(data)[1],))
            data = np.transpose(data)
        print(f'max/mean={np.max(np.abs(data)) / np.mean(np.abs(data))}')
        data = np.expand_dims(data, axis=EXPANDED_DIMS[node_idx])
        new_nodes.append(onnx.helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[outputs[0] + f'_cp{node_idx}' if node_idx != 3 else outputs[0]],
            pads=pads,
            strides=strides,
            kernel_shape=data.shape[2:],
            group=len(weights) if dim in (2, 3) else 1,
        ))
        model.graph.initializer.append(onnx.helper.make_tensor(inputs[1], data_type=orig_weights.data_type, dims=data.shape, vals=data))

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
        print(np.shape(data))
        data = np.transpose(data, axes=TRANSPOSE_AXES[node_idx])
        # print(f'max/mean={np.max(np.abs(data)) / np.mean(np.abs(data))}')
        data = np.expand_dims(data, axis=EXPANDED_DIMS[node_idx])
        new_nodes.append(onnx.helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[outputs[0] + f'_tucker{node_idx}' if node_idx != N_NODES else outputs[0]],
            pads=pads,
            strides=strides,
            kernel_shape=data.shape[2:],
        ))
        model.graph.initializer.append(onnx.helper.make_tensor(inputs[1], data_type=orig_weights.data_type, dims=data.shape, vals=data))

def decompose_conv(model: onnx.ModelProto, node_idx):
    graph = model.graph
    node = graph.node[node_idx]

    new_nodes = []
    orig_weights_name = node.input[1]
    orig_weights = find_initializer(model, orig_weights_name)
    assert orig_weights

    one_dims = 0
    for dim in orig_weights.dims:
        if dim == 1:
            one_dims += 1
    if one_dims >= 2:
        return

    logger.info('Decomposing %s node %s', node.op_type, node.name)

    data = onnx.numpy_helper.to_array(orig_weights)
    data = np.reshape(data, orig_weights.dims)

    if True:
        roots = CPDecomposedFilter(data).roots
        add_cp_filters(model, node, roots, new_nodes)
    elif False:
        roots = Tucker2DecomposedFilter(data).roots
        add_tucker2_filters(model, node, roots, new_nodes)

    graph.node.remove(node)
    for new_node in new_nodes:
        graph.node.insert(node_idx, new_node)
        node_idx += 1
