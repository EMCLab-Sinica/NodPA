# dynamic-pruning

This is a customized tool for dynamic pruning of various models.
The tool is based on https://github.com/frankwang345/dynamic-pruning, which is in turn codes for [Dynamic Network Pruning with Interpretable Layerwise Channel Selection](https://aaai.org/ojs/index.php/AAAI/article/view/6098)

## Usage

- train_baseline.py: pretrain baseline models
- main.py: train dynamic models with decision units
- finetune.py: pruning models based on dynamic pruning method

All evaluation networks were trained from scratch for 10 epochs using the SGD optimizer, with a learning rate of 0.1 for ResNet and HAR, and 0.01 for KWS.

Commands used to train and prune tested models can be found in commands.txt
