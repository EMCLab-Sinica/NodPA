import argparse
import logging
import pathlib
import sys

import onnx
import onnx.version_converter

TOPDIR = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(TOPDIR / 'dnn-models'))

from configs import configs
from utils import (
    load_model,
    onnx_optimize,
)
from fine_tune import (
    copy_quantized_parameters,
    fine_tune,
)
from decomposition.conv import decompose_conv
from decomposition.fc import decompose_gemm
from exp.original_model_run import run_model

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

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=configs.keys())
    parser.add_argument('--epoch', type=int, required=True)
    decomposition_method = parser.add_mutually_exclusive_group(required=True)
    decomposition_method.add_argument('--cp', dest='decomposition_method', action='store_const', const='cp')
    decomposition_method.add_argument('--tucker2', dest='decomposition_method', action='store_const', const='tucker2')
    args = parser.parse_args()

    config = configs[args.config]
    model = load_model(config, model_variant='')

    orig_model = onnx.ModelProto()
    orig_model.ParseFromString(model.SerializeToString())

    model_data = config['data_loader'](train=False)

    nodes_to_decompose = list(range(len(model.graph.node)-1, -1, -1))
    for idx in nodes_to_decompose:
        need_fine_tune = False
        node = orig_model.graph.node[idx]
        if node.op_type == 'Conv':
            decompose_conv(model, node.output[0], args.decomposition_method)
            need_fine_tune = True
        elif node.op_type == 'Gemm':
            decompose_gemm(model, node.output[0])
            need_fine_tune = True
        if need_fine_tune:
            model, net_int8 = fine_tune(model, epoch=args.epoch, config=config)

    copy_quantized_parameters(net_int8, model, add_tensor_annotations=True)

    output_file = TOPDIR / 'dnn-models' / (config['onnx_model'] + f'-{args.decomposition_method}.onnx')

    model = onnx_optimize(model, [
        'eliminate_identity',
    ])

    fix_onnx_ir_version(model)

    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save_model(model, output_file)
    accuracy = run_model(model, model_data, limit=None, verbose=False)
    print(accuracy)

if __name__ == '__main__':
    main()
