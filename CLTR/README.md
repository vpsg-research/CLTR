# CLTR  

This is the code for the paper:
[CLTR: Continual Learning Time-varying Regularization for Robust Classification of Noisy Label Images]    
Authors: Yanhong Li, Zhiqing Guo, Liejun Wang.

## Abstract
On datasets with noisy labels, the deep neural network will overfit the noisy labels, which will weaken the generalization of the model. Related studies show that deep neural networks have the characteristic of memorizing clean data first and then noisy data. Regularization and small loss algorithms exploit the memory characteristics of the network to improve its noise immunity, but they lack control over the updating direction of network parameters, which will inevitably overfit noisy label data. In this paper, we propose a continual learning time-varying regularization (CLTR) based robust classification method for noisy label images. Specifically, learning with noisy label is considered as a continual learning process. By introducing the parameter processing method in catastrophic forgetting, the training parameters of the network are decomposed into clean and noisy parameters. In CLTR, the update direction of clean and noisy parameters is controlled by regularization terms with time-varying coefficients. The time-varying coefficient comes from the initial prediction of the network, which truly reflects the dynamic influence of noisy labels on the network and avoids the setting of hyperparameters. It greatly reduces the complexity of network training and improves its applicability in practice. Extensive experiments on synthetic and real-world benchmarks confirm the superior performance of the proposed method. The code will be released upon acceptance.


## Dependencies
We implement our methods by PyTorch. The environment is as bellow:
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 0.4.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 9.0
- [Anaconda3](https://www.anaconda.com/)

Install PyTorch and Torchvision (Conda):
```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install PyTorch and Torchvision (Pip3):
```bash
pip3 install torch torchvision
```
## Experiments      
Here is an example: 
```bash
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2
```



