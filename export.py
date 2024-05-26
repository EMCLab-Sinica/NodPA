import io
import os.path
from typing import IO

import torch
import torch.onnx
import onnx
import onnxoptimizer
import onnxsim

import misc
from decision import (
    apply_func,
    collect_params,
    decision_basicblock_forward,
    init_decision_basicblock,
    normalize_head_weights,
    replace_func,
    set_deterministic_value,
    set_pruning_threshold,
)
import models

def optimize_model(pytorch_exported_model: IO[bytes], model_name: str):
    onnx_model = onnx.load_model(pytorch_exported_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check
    onnx.save_model(onnx_model, model_name)

def main():
    # No training involved - use dummy values to lr and wd
    parser = misc.get_basic_argument_parser(default_lr=0, default_wd=0)
    args = parser.parse_args()

    args.num_classes = 10 if args.dataset == 'cifar10' else 100
    args.logdir = 'decision-%d/%s-%s/sparsity-%.2f' % (
        args.action_num, args.dataset, args.arch, args.sparsity_level
    )
    misc.prepare_logging(args)

    print('==> Initializing model...')
    model = models.__dict__['cifar_' + args.arch](args.num_classes)

    print('==> Loading pretrained model...')
    checkpoint = torch.load(
        os.path.join(args.logdir, 'checkpoint.pth.tar'),
        map_location=torch.device('cpu'),
    )

    init_func = init_decision_basicblock
    new_forward = decision_basicblock_forward
    module_type = 'BasicBlock'

    print('==> Transforming model...')

    apply_func(model, module_type, init_func, action_num=args.action_num)
    apply_func(model, 'DecisionHead', collect_params)
    replace_func(model, module_type, new_forward)
    apply_func(model, 'DecisionHead', normalize_head_weights)

    model.eval()
    apply_func(model, 'DecisionHead', set_deterministic_value, deterministic=True)

    model.load_state_dict(checkpoint['state_dict'])

    pytorch_exported_model_single = io.BytesIO()
    pytorch_exported_model_batched = io.BytesIO()
    dummy_input = torch.zeros((1, 3, 32, 32))

    onnx_opset = 11

    torch.onnx.export(
        model,
        dummy_input,
        pytorch_exported_model_single,
        opset_version=onnx_opset,
    )

    apply_func(model, 'DecisionHead', set_pruning_threshold, pruning_threshold=args.pruning_threshold)

    torch.onnx.export(
        model,
        dummy_input,
        pytorch_exported_model_batched,
        opset_version=onnx_opset,
        input_names=["input.1"],
        dynamic_axes={
            "input.1": {0: "N"},
        }
    )

    pytorch_exported_model_single.seek(0)
    pytorch_exported_model_batched.seek(0)

    optimize_model(pytorch_exported_model_single, f'cifar_{args.arch}-single.onnx')
    optimize_model(pytorch_exported_model_batched, f'cifar_{args.arch}-batched.onnx')

if __name__ == '__main__':
    main()
