# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

import io

import torch
import torch.onnx
from torch import nn
import onnx
import onnx.shape_inference
import onnxoptimizer
import onnxsim

class Transformer(nn.Module):
    def __init__(self, H, S, P, E):
        super().__init__()

        self.H = H  # heads
        self.S = S  # input length
        self.P = P  # projection dimension
        self.E = E  # embedding dimension

        # With batch_first=True, self-attention uses torch._native_multi_head_attention, which
        # cannot be exported as ONNX yet
        # https://github.com/pytorch/pytorch/blob/v2.1.2/torch/nn/modules/activation.py#L1196
        self.attn = nn.MultiheadAttention(embed_dim=P, num_heads=H, batch_first=True)
        self.w_q = nn.Linear(in_features=E, out_features=P)
        self.w_k = nn.Linear(in_features=E, out_features=P)
        self.w_v = nn.Linear(in_features=E, out_features=P)

    def forward(self, x):
        query = self.w_q(x)
        key   = self.w_k(x)
        value = self.w_v(x)

        out, _ = self.attn(query=query, key=key, value=value, need_weights=False)

        out = out.view(-1, self.S * self.P)

        return out

def main():
    torch.manual_seed(123)

    H = 3
    S = 80
    P = 12
    E = 60

    model = Transformer(H=H, S=S, P=P, E=E)

    model.eval()

    dummy_input = torch.randn([1, S, E])

    pytorch_exported_model = io.BytesIO()

    with torch.no_grad():
        # https://pytorch.org/docs/stable/onnx_torchscript.html
        torch.onnx.export(
            model,
            dummy_input,
            pytorch_exported_model,
            opset_version=14,
            input_names=['input'],
            output_names=['output'],
            # https://stackoverflow.com/questions/76980330/how-to-get-dynamic-batch-size-in-onnx-model-from-pytorch
            # Disabled as somehow onnx shape inference does not work with the generated model, and thus some
            # preprocessing steps in transform.py fail
            # dynamic_axes={
            #     'input': [0],
            #     'output': [0],
            # },
        )

    pytorch_exported_model.seek(0)

    onnx_model = onnx.load_model(pytorch_exported_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx_model = onnxoptimizer.optimize(onnx_model, passes=[
        'eliminate_identity',
        'fuse_matmul_add_bias_into_gemm',
    ])
    onnx_model, check = onnxsim.simplify(
        onnx_model,
        # Skip fuse_qkv to workaround https://github.com/daquexian/onnx-simplifier/issues/284
        skipped_optimizers=['fuse_qkv'],
    )
    assert check
    onnx.save_model(onnx_model, "dnn-models/transformers_single.onnx")

if __name__ == '__main__':
    main()
