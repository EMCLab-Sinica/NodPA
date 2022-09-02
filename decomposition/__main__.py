import argparse
import logging
import pathlib
import sys

import onnx
import onnx.version_converter
import numpy as np

TOPDIR = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(TOPDIR / 'dnn-models'))

from configs import configs
from utils import (
    ONNXNodeWrapper,
    find_initializer,
    find_tensor_value_info,
    load_model,
    onnx_optimize,
    run_model,
)
from fine_tune import (
    fine_tune,
)
from layer_utils import (
    determine_conv_tile_c,
)
from decomposition.conv import decompose_conv
from decomposition.fc import decompose_gemm

logger = logging.getLogger('decomposition')

def fix_onnx_ir_version(model):
    # https://github.com/onnx/onnx/pull/1718
    # IR version 4 allows initializers to be not part of inputs. Actually onnxruntime is unhappy if that's not the case
    model.ir_version = 4
    initializer_names = {n.name for n in model.graph.initializer}
    new_inputs = []
    for inp in model.graph.input:
        if inp.name not in initializer_names:
            new_inputs.append(inp)
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

def find_nodes_to_decompose(model, config, fractal):
    nodes_to_decompose = []

    for idx, node in enumerate(model.graph.node):
        if node.op_type not in ('Conv', 'Gemm'):
            continue

        input_value_info = find_tensor_value_info(model, node.input[0])
        input_dims = list(filter(None, [dim.dim_value for dim in input_value_info.type.tensor_type.shape.dim]))
        weights = find_initializer(model, node.input[1])
        output_value_info = find_tensor_value_info(model, node.output[0])
        output_dims = [dim.dim_value for dim in output_value_info.type.tensor_type.shape.dim]
        if fractal:
            node = ONNXNodeWrapper(node)
            if node.op_type == 'Conv':
                _, CHANNEL, kH, kW = weights.dims
                _, _, OUTPUT_H, OUTPUT_W = output_dims
                determine_conv_tile_c(model, config, is_japari=False, target='msp430', node=node)
                node_flags = node.flags.conv
                inputs_usage_span = 1
                weights_usage_span = node_flags.input_tile_c * kH * kW * node_flags.output_tile_c * OUTPUT_H * OUTPUT_W
            else:
                inputs_usage_span = weights_usage_span = 1
            nodes_to_decompose.append((idx, inputs_usage_span * np.prod(input_dims) + weights_usage_span * np.prod(weights.dims)))
        else:
            nodes_to_decompose.append((idx, np.prod(weights.dims)))

    nodes_to_decompose.sort(key=lambda item: (item[1], item[0]), reverse=True)

    for node_idx, priority in nodes_to_decompose:
        logger.debug('nodes_to_decompose: %s, %d', model.graph.node[node_idx].output[0], priority)

    return nodes_to_decompose

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=configs.keys())
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--fractal', action='store_true')
    parser.add_argument('--debug', action='store_true')
    decomposition_method = parser.add_mutually_exclusive_group(required=True)
    decomposition_method.add_argument('--cp', dest='decomposition_method', action='store_const', const='cp')
    decomposition_method.add_argument('--tucker2', dest='decomposition_method', action='store_const', const='tucker2')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger('decomposition').setLevel(logging.DEBUG)
        logging.getLogger('intermittent-cnn').setLevel(logging.DEBUG)
    else:
        logging.getLogger('decomposition').setLevel(logging.INFO)
        logging.getLogger('intermittent-cnn').setLevel(logging.INFO)

    config = configs[args.config]
    model = load_model(config, model_variant='')

    orig_model = onnx.ModelProto()
    orig_model.ParseFromString(model.SerializeToString())

    train_data = config['data_loader'](train=True)
    test_data = config['data_loader'](train=False)

    orig_accuracy = run_model(model, test_data, limit=None, verbose=False)

    nodes_to_decompose = find_nodes_to_decompose(model, config, args.fractal)

    for idx, _ in nodes_to_decompose:
        node = orig_model.graph.node[idx]
        if node.op_type == 'Conv':
            new_model = decompose_conv(model, orig_accuracy, node.output[0], train_data, test_data, args.epoch, config, args.decomposition_method)
        elif node.op_type == 'Gemm':
            new_model = decompose_gemm(model, orig_accuracy, node.output[0], train_data, test_data, args.epoch, config)
        if new_model:
            model = new_model

    model = fine_tune(model, train_data, test_data, args.epoch, qat=True, config=config)

    suffix = args.decomposition_method
    if args.fractal:
        suffix += '-fractal'
    output_file = TOPDIR / 'dnn-models' / (config['onnx_model'] + f'-{suffix}.onnx')

    model = onnx_optimize(model, [
        'eliminate_identity',
    ])

    fix_onnx_ir_version(model)

    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save_model(model, output_file)
    accuracy = run_model(model, test_data, limit=None, verbose=False)
    logger.info('Accuracy: %.4f', accuracy)

if __name__ == '__main__':
    main()
