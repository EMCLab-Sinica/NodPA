# Capture Non-determinism If You Can: Intermittent Inference on Dynamic Neural Networks

<!-- ABOUT THE PROJECT -->
## Overview

This project develops a middleware module (referred to as NodPA) that accumulates non-deterministic inference progress to enable correct and efficient dynamic neural network inference on intermittent systems. 
NodPA strategically selects additional progress information to capture the non-determinism of the power-interrupted computation while preserving only the changed portions of the progress information to maintain low runtime overhead.

We implemented NodPA on the Texas Instruments device MSP-EXP432P401R. It is an ARM-based 32-bit MCU with 64KB SRAM and single instruction multiple data (SIMD) instructions for accelerated computation. An external NVM module (Cypress CY15B116QN serial FRAM) was integrated to the platform. 

NodPA was integrated with the [HAWAII](https://ieeexplore.ieee.org/document/9211553) intermittent inference engine for evalution purposes, although it is compatible with other engines. 

NodPA contains two main components which interacts with the inference engine at runtime:

* Non-determinism accumulator: determines the sufficient set of progress indicators to track, ensuring non-deterministic progress information capture of the target dynamic network.
* Preservation minimizer: ensures low progress preservation overhead, by reducing both the amount of data and number of transfers when preserving progress to NVM. 

We evaluate NodPA on three networks, ResNet, HAR, and KWS, using both static and dynamic variants, trained on the CIFAR-10 dataset, an accelerometer sensor dataset, and the Google Speech Commands dataset, respectively.
We compare NodPA with two existing baselines, CD and FD, in terms of model accuracy, inference latency, and runtime overhead, which includes both preservation and recovery overhead.


Demo video: [https://youtu.be/_1qVoG4aCxY](https://youtu.be/_1qVoG4aCxY)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)

## Directory/File Structure

Below is an explanation of the directories/files found in this repo.

* `common/conv.cpp`, `common/fc.cpp`, `common/pooling.cpp`, `common/op_handlers.cpp`, `common/op_utils.*`: functions implementing various neural network layers and auxiliary functions shared among different layers.
* `common/cnn_common.*`, `common/intermittent-cnn.*`: main components of the HAWAII intermittent inference engine.
* `common/platform.*`, `common/plat-mcu.*` and `common/plat-pc.*`: high-level wrappers for handling platform-specific peripherals.
* `common/my_dsplib.*`: high-level wrappers for accessing different vendor-specific library calls performing accelerated computations.
* `common/counters.*` : helper functions for measuring runtime overhead.
* `dnn-models/`: pre-trained models and python scripts for model training, converting different model formats to ONNX and converting a model into a custom format recognized by the lightweight inference engine.
* `msp432/`: platform-speicific hardware initialization functions.
* `tools/`: helper functions for various system peripherals (e.g., UART, system clocks and external FRAM).
* `train/`: codes for training neural networks with dynamic pruning.

## Getting Started

### Prerequisites

Here are basic software and hardware requirements to build NodPA along with the HAWAII intermittent inference engine:

* Python 3.11
* Several deep learning Python libraries defined in `requirements-base.txt`. Those libraries can be installed with `pip3 install -r requirements-base.txt`.
* [Code composer studio](https://www.ti.com/tool/CCSTUDIO) 12.8
* [MSP-EXP432P401R LaunchPad](https://www.ti.com/tool/MSP-EXP432P401R)
* [MSP432 driverlib](https://www.ti.com/tool/MSPDRIVERLIB) 3.21.00.05

### Setup and Build

1. Prepare vendor-supplied libraries for hardware-accelerated computation. `git submodule update --init --recursive` will download them all.
1. Convert the provided pre-trained models with the command `python3 dnn-models/transform.py --target msp432 --hawaii (cifar10-dnp|har-dnp)` to specify the model to deploy from one of `cifar10-dnp`, or `har-dnp`.
1. Download and extract MSP432 driverlib, and copy `driverlib/MSP432P4xx` folder into the `msp432/` folder.
1. Import the folder `msp432/` as a project in CCStudio.
