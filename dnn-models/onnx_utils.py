import logging

import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np

from utils import (
    find_initializer,
)

logger = logging.getLogger(__name__)

def add_tensor_annotation(onnx_model, key, tensor_name, data_type, vals):
    vals = np.array(vals)
    dims = np.shape(vals)

    existing_tensor = find_tensor_annotation_initializer(onnx_model, key, tensor_name)
    if existing_tensor:
        logger.debug(f'Update existing tensor annotation {existing_tensor.name} = {vals}')
        new_tensor = onnx.helper.make_tensor(existing_tensor.name, data_type, dims, vals.flatten())
        existing_tensor.ParseFromString(new_tensor.SerializeToString())

    mapping = onnx.StringStringEntryProto()
    mapping.key = key
    mapping.value = f'{tensor_name}.{key}'

    annotation = onnx.TensorAnnotation()
    annotation.tensor_name = tensor_name
    annotation.quant_parameter_tensor_names.append(mapping)

    onnx_model.graph.quantization_annotation.append(annotation)

    tensor = onnx.helper.make_tensor(name=mapping.value, data_type=data_type,
                                     dims=dims, vals=vals.flatten())
    onnx_model.graph.initializer.append(tensor)

def find_tensor_annotation_initializer(onnx_model: onnx.ModelProto, key: str, tensor_name: str):
    for tensor_annotation in onnx_model.graph.quantization_annotation:
        if tensor_annotation.tensor_name != tensor_name:
            continue
        for mapping in tensor_annotation.quant_parameter_tensor_names:
            if key != mapping.key:
                continue
            return find_initializer(onnx_model, mapping.value)

def find_tensor_annotation(onnx_model: onnx.ModelProto, key: str, tensor_name: str):
    initializer = find_tensor_annotation_initializer(onnx_model, key, tensor_name)
    if initializer:
        return onnx.numpy_helper.to_array(initializer)

def list_tensors_for_annotations(onnx_model: onnx.ModelProto):
    referenced_tensors = []
    for tensor_annotation in onnx_model.graph.quantization_annotation:
        for mapping in tensor_annotation.quant_parameter_tensor_names:
            referenced_tensors.append(mapping.value)
    return referenced_tensors

def dims_from_value_info(value_info: onnx.ValueInfoProto):
    shape = value_info.type.tensor_type.shape
    dims = []
    for dim in shape.dim[1:]:  # The first dimension is the batch size
        assert dim.WhichOneof('value') == 'dim_value'
        dims.append(dim.dim_value)
    return dims

def get_param_limit(model: onnx.ModelProto, node: onnx.NodeProto):
    param_limit = 0
    for input_idx, input_ in enumerate(node.input[1:]):  # weights & possibly biases
        param_limit = max(param_limit, np.max(np.abs(onnx.numpy_helper.to_array(find_initializer(model, input_)))))
    return param_limit

def compute_parameter_scales(onnx_model: onnx.ModelProto):
    if not onnx_model.graph.quantization_annotation:
        # Use hand-configured scales if there are no scale information in the model
        return

    for node in onnx_model.graph.node:
        if node.op_type not in ('Add', 'Conv', 'Gemm'):
            continue
        add_tensor_annotation(onnx_model, key='Q15_SCLAE_PARAMS', tensor_name=node.output[0],
                              data_type=onnx.TensorProto.DataType.FLOAT, vals=get_param_limit(onnx_model, node))

    for idx in range(len(onnx_model.graph.node) - 1, 0, -1):
        node = onnx_model.graph.node[idx]
        if node.op_type not in ('ConvMerge', 'GemmMerge'):
            continue
        # * 2 for asymmetric quantization (?)
        scale = find_tensor_annotation(onnx_model, key='SCALE_TENSOR', tensor_name=node.output[0]) * 2
        param_scale = find_tensor_annotation(onnx_model, key='Q15_SCLAE_PARAMS', tensor_name=node.input[0])
        new_prev_scale = (scale / param_scale)
        for idx2 in range(idx - 2, 0, -1):
            if onnx_model.graph.node[idx2].op_type in ('Add', 'ConvMerge', 'GemmMerge'):
                break
        else:
            continue
        prev_node = onnx_model.graph.node[idx2]
        cur_prev_scale = find_tensor_annotation(onnx_model, key='SCALE_TENSOR', tensor_name=prev_node.output[0])
        if new_prev_scale > cur_prev_scale:
            add_tensor_annotation(onnx_model, key='SCALE_TENSOR', tensor_name=prev_node.output[0],
                                  data_type=onnx.TensorProto.DataType.FLOAT, vals=new_prev_scale)
