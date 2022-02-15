import logging
import pathlib
import sys

import onnx
import onnx.version_converter
import onnxoptimizer

TOPDIR = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(TOPDIR / 'dnn-models'))

from configs import configs
from utils import load_model
from decomposition.conv import decompose_conv
from decomposition.fc import decompose_gemm
from exp.original_model_run import run_model

def fix_onnx_ir_version(model):
    # https://github.com/onnx/onnx/pull/1718
    # IR version 4 allows initializers to be not part of inputs. Actually onnxruntime is unhappy if that's not the case
    # MXNet does not like IR version >= 4...
    model.ir_version = 4
    #type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.DataType.FLOAT, shape=zero_dim)
    #if model.ir_version < 4:
    #    model.graph.input.append(onnx.helper.make_value_info(name=new_node_name, type_proto=type_proto))
    initializer_names = {n.name for n in model.graph.initializer}
    print(initializer_names)
    new_inputs = []
    for inp in model.graph.input:
        print(inp.name)
        if inp.name not in initializer_names:
            new_inputs.append(inp)
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

def main():
    logging.basicConfig(level=logging.DEBUG)

    config = configs[sys.argv[1]]
    model = load_model(config, for_deployment=False, model_variant='')

    model_data = config['data_loader'](train=False)

    for idx in range(len(model.graph.node)-1, -1, -1):
        node = model.graph.node[idx]
        if node.op_type == 'Conv':
            decompose_conv(model, idx)
        elif node.op_type == 'Gemm':
            decompose_gemm(model, idx)

    input_names = set(inp.name for inp in model.graph.input)
    for initializer in model.graph.initializer:
        if initializer.name in input_names:
            break
        model.graph.input.append(onnx.helper.make_value_info(
            name=initializer.name,
            type_proto=onnx.helper.make_tensor_type_proto(elem_type=initializer.data_type, shape=initializer.dims),
        ))

    output_file = TOPDIR / 'dnn-models' / (config['onnx_model'] + '-decomposed.onnx')
    onnx.save_model(model, output_file)

    model = onnxoptimizer.optimize(model, [
        'eliminate_identity',
        'eliminate_unused_initializer',
    ])

    fix_onnx_ir_version(model)

    onnx.checker.check_model(model)
    onnx.save_model(model, output_file)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save_model(model, output_file)
    accuracy = run_model(model, model_data, limit=None, verbose=False)
    print(accuracy)

if __name__ == '__main__':
    main()
