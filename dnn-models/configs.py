from typing import Protocol, TypedDict
import sys

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from datasets import (
    load_data_cifar10,
    load_data_google_speech,
    load_har,
)
from utils import ModelData

ARM_PSTATE_LEN = 8704
# Acceleration output buffer size
# TODO: make these adjustable on runtime
OUTPUT_LEN = 256

vm_size = {
    # (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)
    'msp430': 1884,
    # determined by trial and error
    'msp432': 26704,  # includes space for pState
}

class DataLoader(Protocol):
    def __call__(self, train: bool, target_size: tuple[int, int]) ->  ModelData:
        ...

class ConfigType(TypedDict):
    onnx_model: str
    onnx_model_single: NotRequired[str]
    scale: float
    input_scale: float
    num_slots: int
    data_loader: DataLoader
    n_all_samples: int
    op_filters: int
    # Filled by transform.py
    total_sample_size: NotRequired[int]
    gemm_tile_length: NotRequired[int]

configs: dict[str, ConfigType] = {
    'cifar10': {
        'onnx_model': 'squeezenet_cifar10',
        'scale': 2,
        'input_scale': 10,
        'num_slots': 3,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'op_filters': 2,
    },
    'kws': {
        'onnx_model': 'KWS-DNN_S',
        'scale': 1,
        'input_scale': 120,
        'num_slots': 2,
        'data_loader': load_data_google_speech,
        'n_all_samples': 4890,
        'op_filters': 4,
    },
    'har': {
        'onnx_model': 'HAR-CNN',
        'scale': 2,
        'input_scale': 16,
        'num_slots': 2,
        'data_loader': load_har,
        'n_all_samples': 2947,
        'op_filters': 4,
    },
}

