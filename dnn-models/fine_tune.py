# QAT: see https://pytorch.org/docs/stable/quantization.html, https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
# Export int8 model to ONNX (failed, given up) https://discuss.pytorch.org/t/onnx-export-of-quantized-model/76884/21
# https://zhuanlan.zhihu.com/p/349019936

from __future__ import annotations

import argparse
import logging
from typing import Any

from configs import configs
from utils import (
    THIS_DIR,
    dynamic_shape_inference,
    find_initializer,
    find_node_by_output,
    infer_auto_pad,
    load_model,
)
from onnx_utils import (
    add_tensor_annotation,
)

import torch
import torch.quantization
import torch.quantization.quantize_fx as quantize_fx
import onnx
import onnx.helper
import onnx.version_converter
import onnx2torch
import onnxoptimizer

logger = logging.getLogger('intermittent-cnn.fine_tune')

def measure_accuracy(net, testloader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info('Accuracy of the network on the 10000 test images: %.4f', correct / total)

def train_model(net, trainloader, testloader, device, lr, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                logger.info('[%d, %5d] loss: %.3f', epoch + 1, i + 1, running_loss / 200)
                running_loss = 0.0

        # measure_accuracy(net, testloader, device)

# Seems PyTorch Conv2D does not support asymmetric padding out-of-box, and
# onnx2torch does not support converting to combined operators, either [1]
# https://github.com/ENOT-AutoDL/onnx2torch/issues/52
def onnx_split_asymmetric_padding(onnx_model):
    # Return a new model as I don't want Pad in the merged model
    new_model = onnx.ModelProto()
    new_model.ParseFromString(onnx_model.SerializeToString())
    new_nodes = []
    g = new_model.graph
    for node in g.node:
        if node.op_type != 'Conv':
            new_nodes.append(node)
            continue

        pads = infer_auto_pad(new_model, node)
        # pads are [x1_begin, x2_begin...x1_end, x2_end,...]
        if pads[0] == pads[2] and pads[1] == pads[3]:
            new_nodes.append(node)
            continue

        name = node.input[0]
        g.initializer.append(onnx.helper.make_tensor(f'{name}_pads',
                                                     data_type=onnx.TensorProto.DataType.INT64,
                                                     dims=(8,), vals=[0, 0] + pads[0:2] + [0, 0] + pads[2:4]))
        new_nodes.append(onnx.helper.make_node('Pad',
                                               inputs=[name, f'{name}_pads'],
                                               outputs=[f'{name}_padded'],
                                               name=f'{name}_pre_padding'))

        # Remove original padding attributes
        new_attrs = [attr for attr in node.attribute
                     if attr.name not in ('pads', 'auto_pad')]
        del node.attribute[:]
        node.attribute.extend(new_attrs)

        node.input[0] = f'{name}_padded'
        new_nodes.append(node)

    del g.node[:]
    g.node.extend(new_nodes)

    return new_model

# See https://pytorch.org/docs/stable/fx.html for how to trace an FX graph
class FXGraphWalker:
    def __init__(self, net, onnx_model, add_tensor_annotations: bool):
        self.net = net
        self.onnx_model = onnx_model
        self.onnx_node_idx = 0
        self.add_tensor_annotations = add_tensor_annotations

    def _go_to_onnx_node(self, op_type):
        while self.onnx_model.graph.node[self.onnx_node_idx].op_type != op_type:
            self.onnx_node_idx += 1
        return self.onnx_model.graph.node[self.onnx_node_idx]

    def _handle_add(self, node):
        onnx_node = self._go_to_onnx_node('Add')
        _, _, scale_node, zero_point_node = node.args
        scale = self._handle_node(scale_node)
        zero_point = self._handle_node(zero_point_node)
        assert zero_point == 0

        if self.add_tensor_annotations:
            add_tensor_annotation(self.onnx_model, key='SCALE_TENSOR', tensor_name=onnx_node.output[0],
                                  data_type=onnx.TensorProto.DataType.FLOAT, vals=scale * 128)

        self.onnx_node_idx += 1

    def _handle_linear(self, op, onnx_op_type):
        params = [op.weight(), op.bias()]
        if onnx_op_type == 'Gemm':
            params[0] = torch.transpose(params[0], 0, 1)
        onnx_node = self._go_to_onnx_node(onnx_op_type)
        for idx in range(len(onnx_node.input[1:])):
            onnx_param = find_initializer(self.onnx_model, onnx_node.input[1+idx])
            assert onnx_param.dims == list(params[idx].size())
            new_param = onnx.helper.make_tensor(onnx_param.name, onnx_param.data_type, onnx_param.dims, vals=torch.flatten(torch.dequantize(params[idx])))
            onnx_param.ParseFromString(new_param.SerializeToString())
        if self.add_tensor_annotations:
            add_tensor_annotation(self.onnx_model, key='SCALE_TENSOR', tensor_name=onnx_node.output[0],
                                  data_type=onnx.TensorProto.DataType.FLOAT, vals=op.scale * 128)
            add_tensor_annotation(self.onnx_model, key='ZERO_POINT_TENSOR', tensor_name=onnx_node.output[0],
                                  data_type=onnx.TensorProto.DataType.INT64, vals=op.zero_point)
        self.onnx_node_idx += 1

    def _handle_call_module(self, node):
        op = getattr(self.net, node.target)
        if isinstance(op, torch.nn.quantized.Conv2d):
            self._handle_linear(op, 'Conv')
        elif isinstance(op, torch.nn.quantized.Linear):
            self._handle_linear(op, 'Gemm')

    def _handle_call_function(self, node):
        if node.target == torch.ops.quantized.add_relu:
            self._handle_add(node)

    def _handle_getattr(self, node):
        parts = node.target.split('.')
        ret = self.net
        while parts:
            ret = getattr(ret, parts.pop(0))
        return ret

    def _handle_node(self, node):
        if node.op == 'call_module':
            self._handle_call_module(node)
        elif node.op == 'call_function':
            self._handle_call_function(node)
        elif node.op == 'get_attr':
            return self._handle_getattr(node)

    def run(self):
        for node in self.net.graph.nodes:
            self._handle_node(node)

def copy_quantized_parameters(net_int8, onnx_model: onnx.ModelProto, add_tensor_annotations: bool):
    FXGraphWalker(net_int8, onnx_model, add_tensor_annotations).run()

def strip_softmax(onnx_model: onnx.ModelProto):
    # PyTorch CrossEntropyLoss already includes Softmax - strip it from the model
    output_node = find_node_by_output(onnx_model.graph.node, onnx_model.graph.output[0].name)
    if output_node.op_type == 'Softmax':
        onnx_model.graph.output[0].name = output_node.input[0]
        del onnx_model.graph.node[-1]
        new_value_infos = [value_info for value_info in onnx_model.graph.value_info
                           if value_info.name != output_node.output[0]]
        del onnx_model.graph.value_info[:]
        onnx_model.graph.value_info.extend(new_value_infos)
        logger.info('Removed %s', output_node.name)

def fine_tune(onnx_model: onnx.ModelProto, epoch: int, config: dict[str, Any]):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device %s', device)

    strip_softmax(onnx_model)

    onnx_model = onnx.version_converter.convert_version(onnx_model, 11)

    onnx_model = onnxoptimizer.optimize(onnx_model, [
        'extract_constant_to_initializer',
    ])

    net = onnx2torch.convert(onnx_split_asymmetric_padding(onnx_model))
    logger.debug('%r', net)

    # Make sure forward() can pass
    example_inputs = torch.randn([1] + config['sample_size'])
    net(example_inputs)

    net.to(device)

    net.train()
    # fbgemm uses per-channel quantization, which is not supported yet
    # https://github.com/pytorch/pytorch/issues/75785
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

    # Skip fuse_modules - requires manually listing modules & seems working without that

    qconfig_dict = {
        '': net.qconfig,
    }
    try:
        # example_inputs available and needed since pytorch 1.13
        # https://github.com/pytorch/pytorch/pull/77608
        net_prepared = quantize_fx.prepare_qat_fx(net, qconfig_dict, example_inputs=(example_inputs,))
    except TypeError:
        net_prepared = quantize_fx.prepare_qat_fx(net, qconfig_dict)

    batch_size = config['fine_tune_batch_size']

    if epoch:
        trainset = config['data_loader'](train=True).dataset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, pin_memory=True)

        testset = config['data_loader'](train=False).dataset
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True, pin_memory=True)

        train_model(net_prepared, trainloader, testloader, device, lr=config['learning_rate'], epoch=epoch)

    # https://discuss.pytorch.org/t/could-not-run-quantized-conv2d-new-with-arguments-from-the-quantizedcuda-backend/133516/4
    device = torch.device('cpu')
    net_prepared.to(device)
    net_prepared.eval()

    net_int8 = quantize_fx.convert_fx(net_prepared)
    if epoch:
        measure_accuracy(net_int8, testloader, device)

    copy_quantized_parameters(net_int8, onnx_model, add_tensor_annotations=False)

    dynamic_shape_inference(onnx_model, config['sample_size'])

    return onnx_model, net_int8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=configs.keys())
    parser.add_argument('--model-variant', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger('intermittent-cnn').setLevel(logging.DEBUG)

    config = configs[args.config]

    onnx_model = load_model(config, args.model_variant)

    onnx_model, net_int8 = fine_tune(onnx_model, args.epoch, config)

    copy_quantized_parameters(net_int8, onnx_model, add_tensor_annotations=True)

    onnx.save_model(onnx_model, THIS_DIR / f'{config["onnx_model"]}-{args.model_variant}_qat.onnx')

if __name__ == '__main__':
    main()
