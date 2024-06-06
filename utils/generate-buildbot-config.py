import dataclasses
import itertools
import json
import pathlib
from typing import Any

@dataclasses.dataclass
class ModelConfig:
    model: str
    targets: list[str]
    approaches: list[str]
    batch_sizes: list[int]

def config_builder(model: str, batch_size: int, approach: str, target: str) -> dict[str, Any]:
    config = f'--target {target} --{approach} --batch-size {batch_size} {model}'
    if approach == 'ideal' or approach == 'stateful' and batch_size == 1:
        config += ' --all-samples'

    suffix = f'{model}_{approach}_b{batch_size}'

    if target == 'msp432':
        suffix += '_cmsis'

    return {
        'builder_name': f'stateful-cnn-{suffix}',
        'command_env': {
            'CONFIG': config,
            'LOG_SUFFIX': suffix.replace('.', '_'),
        },
    }

def main() -> None:
    all_targets = ['msp430', 'msp432']
    all_approaches = ['hawaii', 'japari', 'stateful']
    ideal_config = {'targets': all_targets, 'approaches': ['ideal'], 'batch_sizes': [1]}
    complete_config = {'targets': all_targets, 'approaches': all_approaches, 'batch_sizes': [1]}
    model_configs = [
        ModelConfig(model='cifar10', **ideal_config),
        ModelConfig(model='cifar10', **complete_config),
        ModelConfig(model='kws', **ideal_config),
        ModelConfig(model='kws', targets=all_targets, approaches=all_approaches, batch_sizes=[1, 2]),
        ModelConfig(model='har', **ideal_config),
        ModelConfig(model='har', **complete_config),
        ModelConfig(model='transformers', **ideal_config),
        ModelConfig(model='transformers', targets=all_targets, approaches=['hawaii', 'stateful'], batch_sizes=[1]),
        ModelConfig(model='cifar10-dnp', **ideal_config),
        ModelConfig(model='cifar10-dnp', targets=all_targets, approaches=['hawaii'], batch_sizes=[1]),
        ModelConfig(model='har-dnp', **ideal_config),
        ModelConfig(model='har-dnp', targets=all_targets, approaches=['hawaii'], batch_sizes=[1]),
    ]

    buildbot_configurations = []
    for m in model_configs:
        for target, approach, batch_size in itertools.product(m.targets, m.approaches, m.batch_sizes):
            buildbot_configurations.append(config_builder(m.model, batch_size, approach, target))

    output_path = pathlib.Path(__file__).resolve().parent / 'buildbot-config.json'
    with open(output_path, 'w') as f:
        json.dump(buildbot_configurations, f, indent=4)

if __name__ == '__main__':
    main()
