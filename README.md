## Description

The implementation of "AACP: Model Compression by Accurate and Automatic Channel Pruning"(accepted by ICPR2022), by Lanbo Lin, Shengjie Chen, Yujiu Yang and Zhenhua Guo.

## Table of Contents

1. [Installation](#install)
2. [Usage](#usage)
3. [Pretrained Models](#pretrained)

<a name="install"/>
## Installation

Before running the code, installing the packages in requirements.txt. Note that the **apex** is required to be install by compiling the source code to avoid errors occur. You can install apex by:

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir ./
```
<a name="usage"/>
## Usage

We provide the code of pruning VGG16/Resnet56/Resnet110 on CIFAR10 and Resnet50/Mobilenetv2 on ImageNet (ILSVRC2012). All commands are run in the directory of " your_path/AACP"

```bash
cd your_path/AACP
```

### CIFAR10

#### Pruning VGG16 on CIFAR10

1. Train an unpruned VGG16 model. Change the configurations and run

```bash
bash scripts/train_cifar10_vgg.sh
```

2. Search the optimal pruned architecture and finetune the searched result. Change the configurations and run

```bash
bash scripts/prune_cifar10_vgg.sh
```

Logs are saved to ./results by default.

#### Pruning Resnet56 on CIFAR10

1. Train an unpruned Resnet56 model. Change the configurations and run

```bash
bash scripts/train_cifar10_resnet56.sh
```

2. Search the optimal pruned architecture and finetune the searched result. Change the configurations and run

```bash
bash scripts/prune_cifar10_resnet56.sh
```

#### Pruning Resnet110 on CIFAR10

1. Train an unpruned Resnet56 model. Change the configurations and run

```bash
bash scripts/train_cifar10_resnet110.sh
```

2. Search the optimal pruned architecture and finetune the searched result. Change the configurations and run

```bash
bash scripts/prune_cifar10_resnet110.sh
```

### ImageNet

#### Pruning ResNet50 on ImageNet

1. Train an unpruned Resnet50 model. Change the configurations and run

```bash
bash scripts/train_imagenet_resnet50.sh
```

2. Search the optimal pruned architecture. Change the configurations and run

```bash
bash scripts/prune_imagenet_resnet50.sh
```

3. Finetune the searched result. Change the configurations and run

```bash
bash scripts/finetune_imagenet_resnet50.sh
```
<a name="pretrained"/>
#### Pruning Mobilenetv2 on ImageNet

1. Train an unpruned mobilenetv2 model. Change the configurations and run

```bash
bash scripts/train_imagenet_mobilenetv2.sh
```

2. Search the optimal pruned architecture. Change the configurations and run

```bash
bash scripts/prune_imagenet_mobilenetv2.sh
```

3. Finetune the searched result. Change the configurations and run

```bash
bash scripts/finetune_imagenet_mobilenetv2.sh
```

## Pretrained Models

Coming soon.
