from utils import (
    load_data_cifar10,
    load_data_google_speech,
    load_har,
    load_data_tomato,
)

# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'tomato': {
        'onnx_model': 'dnn-models/tomato.onnx',
        'scale': 4,
        'input_scale': 2,
        'num_slots': 2,
        'intermediate_values_size': 16000000,
        'data_loader': load_data_tomato,
        'n_all_samples': 1,
        'sample_size': [3, 300, 300],
        'op_filters': 2,
        'first_sample_outputs': [ -4.368355, -5.787876, -1.454861, 0.323135, -12.185640, -5.198028, -18.445097, 12.515130, -3.513454, 1.321224 ],
        'fp32_accuracy': 0.9769,
    },
    'cifar10': {
        'onnx_model': 'dnn-models/squeezenet_cifar10.onnx',
        'scale': 2,
        'input_scale': 4,
        'num_slots': 3,
        'intermediate_values_size': 65000,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'sample_size': [32, 32, 3],
        'op_filters': 2,
        'first_sample_outputs': [ 4.895500, 4.331344, 4.631835, 11.602396, 4.454658, 10.819544, 5.423588, 6.451203, 5.806091, 5.272837 ],
        'fp32_accuracy': 0.7704,
    },
    'kws': {
        'onnx_model': 'dnn-models/KWS-DNN_S.onnx',
        'scale': 1,
        'input_scale': 120,
        'num_slots': 2,
        'intermediate_values_size': 20000,
        'data_loader': load_data_google_speech,
        'n_all_samples': 4890,
        'sample_size': [25, 10],  # MFCC gives 25x10 tensors
        'op_filters': 4,
        'first_sample_outputs': [ -29.228327, 5.429047, 22.146973, 3.142066, -10.448060, -9.513299, 15.832925, -4.655487, -14.588447, -1.577156, -5.864228, -6.609077 ],
        'fp32_accuracy': 0.7983,
    },
    'har': {
        'onnx_model': 'dnn-models/HAR-CNN.onnx',
        'scale': 2,
        'input_scale': 16,
        'num_slots': 2,
        'intermediate_values_size': 20000,
        'data_loader': load_har,
        'n_all_samples': 2947,
        'sample_size': [9, 128],
        'op_filters': 4,
        'first_sample_outputs': [ -6.194588, 2.2284777, -13.659239, -1.4972568, 13.473643, -10.446839 ],
        'fp32_accuracy': 0.9121,
    },
}

