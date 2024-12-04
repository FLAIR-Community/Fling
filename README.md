<img src="./README.assets/fling.png" style="zoom: 25%;" />

## Fling

FL-Launching (**Fling**) is a research platform for Federated Learning using PyTorch as backend. 

Its goal is to **simulate the distributed learning process of Federated Learning on single or multiple machines**, providing a **fair testing platform** for the performance of various federated learning algorithms on different datasets. It is mainly based on the Python language and uses the PyTorch framework as the backend module for deep learning, supporting a variety of federated learning algorithms and commonly used federated learning datasets.

It mainly supports:

- Generic Federated Learning methods, such as FedAvg.
- Personalized Federated Learning methods, such as FedPer.
- Attacking methods, such as DLG.

## Installation

Firstly, it is recommended to install PyTorch manually with a suitable version (specifically 1.1.0 or higher). However, using PyTorch version 2.0.0 or later is preferred due to its better computational efficiency. Instructions for installation can be found at this [link](https://pytorch.org/get-started/locally/).

After the first step, you can simply install the latest version of Fling with the following command by using Git:

```bash
git clone https://github.com/FLAIR-Community/Fling
cd Fling
pip install -e .
```

Finally, you can use

```bash
fling -v
```

to check whether Fling is successfully installed.

## Quick Start

After successfully install Fling, users can start the first Fling experiment by using the following command. An example for generic federated learning:

```bash
python flzoo/mnist/mnist_fedavg_cnn_toy_config.py
```

Or using our cli util by:

```shell
fling run -c flzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline
```

This config is a simplified version for conducting FedAvg on the dataset MNIST and iterate for 4 communication rounds.

For other algorithms and datasets, users can refer to `argzoo/` or customize your own configuration files.

For visualization utilities, please refer to [README for visualization](https://github.com/FLAIR-Community/Fling/tree/main/fling/utils/visualize_utils/README.md).

For attacking methods, please refer to our examples in: [demo for attack](https://github.com/FLAIR-Community/Fling/blob/main/fling/utils/attack_utils/demo)

**Tutorials:**

[Overall Framework of Fling](https://github.com/FLAIR-Community/Fling/blob/main/docs/framework_for_fling_en.md) | [Fling 整体框架](https://github.com/FLAIR-Community/Fling/blob/main/docs/framework_for_fling_zh.md)

[Meaning for Each Configuration Key](https://github.com/FLAIR-Community/Fling/blob/main/docs/meaning_for_configurations_en.md) | [配置文件各字段含义](https://github.com/FLAIR-Community/Fling/blob/main/docs/meaning_for_configurations_zh.md)

[How to Add a New FL Algorithm](https://github.com/FLAIR-Community/Fling/blob/main/docs/how_to_add_new_algorithm_en.md) | [如何自定义联邦学习算法](https://github.com/FLAIR-Community/Fling/blob/main/docs/how_to_add_new_algorithm_zh.md)

[How to Add a New Dataset](https://github.com/FLAIR-Community/Fling/blob/main/docs/how_to_add_new_dataset_en.md) | [如何添加新数据集](https://github.com/FLAIR-Community/Fling/blob/main/docs/how_to_add_new_dataset_zh.md)

[Cli Usage in Fling](https://github.com/FLAIR-Community/Fling/blob/main/docs/cli_en.md) | [Fling 的 CLI 使用](https://github.com/FLAIR-Community/Fling/blob/main/docs/cli_zh.md)

## Feature

- Support for a variety of algorithms and datasets.
- Support multiprocessing training on each client for better efficiency.
- Using single GPU to simulate Federated Learning process.
- Strong visualization utilities. See this [README](https://github.com/FLAIR-Community/Fling/tree/main/fling/utils/visualize_utils/README.md) file for detailed information. There are also [demos](https://github.com/FLAIR-Community/Fling/blob/main/fling/utils/visualize_utils/demo) for reference.

## Supported Algorithms

![generic](https://img.shields.io/badge/-generic-brightgreen) &nbsp; Generic federated learning, which finally trains a single global model for all clients.

![personalized](https://img.shields.io/badge/-personalized-green) &nbsp; Personalized federated learning, which finally trains a personalized model for each client.

![attacking](https://img.shields.io/badge/-attacking-darkgreen) &nbsp; Attacking methods, which simulate the attacking process of adversaries and test the robustness of federated learning algorithms.

![visualization](https://img.shields.io/badge/-visualization-yellow) &nbsp; Visualization utilities for federated learning.

![cross-domain](https://img.shields.io/badge/-crossdomain-blue) &nbsp; Cross-domain scenarios represent feature shift heterogeneity, where data from different clients originate from different domains, resulting in distinct feature distributions. 

Fling currently supports [DomainNet](https://ai.bu.edu/M3SDA/), with plans to extend support to additional cross-domain datasets in the future. The **flzoo/domainnet/** folder provides demos of various algorithms on DomainNet.


| Algorithm | Reference Link                                               | Categories                                                   | Demo                                                   |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| FedAvg    | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) | ![generic](https://img.shields.io/badge/-generic-brightgreen)![cross-domain](https://img.shields.io/badge/-crossdomain-blue) | python flzoo/cifar10/cifar10_fedavg_resnet_config.py   |
| FedProx   | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf) | ![generic](https://img.shields.io/badge/-generic-brightgreen) | python flzoo/cifar10/cifar10_fedprox_resnet_config.py  |
| FedMOON   | [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf) | ![generic](https://img.shields.io/badge/-generic-brightgreen) | python flzoo/cifar10/cifar10_fedmoon_cnn_config.py     |
| SCAFFOLD  | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning ](https://arxiv.org/abs/1910.06378) | ![generic](https://img.shields.io/badge/-generic-brightgreen) | python flzoo/cifar10/cifar10_scaffold_resnet_config.py |
| FedPart   | [Why Go Full? Elevating Federated Learning Through Partial Network Updates](https://arxiv.org/abs/2410.11559) | ![generic](https://img.shields.io/badge/-generic-brightgreen)![personalized](https://img.shields.io/badge/-personalized-green) | python flzoo/cifar10/cifar10_fedpart_resnet8_config.py |
| FedPer    | [Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818v1.pdf) | ![personalized](https://img.shields.io/badge/-personalized-green)![cross-domain](https://img.shields.io/badge/-crossdomain-blue) | python flzoo/cifar10/cifar10_fedper_resnet_config.py   |
| FedBN     | [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/pdf/2102.07623.pdf) | ![personalized](https://img.shields.io/badge/-personalized-green)![cross-domain](https://img.shields.io/badge/-crossdomain-blue) | python flzoo/cifar10/cifar10_fedbn_resnet_config.py    |
| FedRoD    | [On Bridging Generic and Personalized Federated Learning for Image Classification](https://openreview.net/pdf?id=I1hQbx10Kxn) | ![personalized](https://img.shields.io/badge/-personalized-green) | python flzoo/cifar10/cifar10_fedrod_resnet_config.py   |
| pFedSD    | [Personalized Edge Intelligence via Federated Self-Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/9964434) | ![personalized](https://img.shields.io/badge/-personalized-green) | python flzoo/cifar10/cifar10_pfedsd_resnet_config.py   |
| FedCAC    | [Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration](https://arxiv.org/abs/2309.11103) | ![personalized](https://img.shields.io/badge/-personalized-green) | python flzoo/cifar10/cifar10_fedcac_resnet_config.py   |
| DLG       | [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935) | ![attacking](https://img.shields.io/badge/-attacking-darkgreen) | python fling/utils/attack_utils/demo/demo_dlg.py       |
| iDLG      | [Inverting Gradients -- How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053) | ![attacking](https://img.shields.io/badge/-attacking-darkgreen) | python fling/utils/attack_utils/demo/demo_idlg.py      |

## Feedback and Contribution

- For any bugs, questions, feature requirements, feel free to propose them in [issues](https://github.com/FLAIR-Community/Fling/issues)
- For any contributions that can improve Fling (more algorithms or better system design), we warmly welcome you to propose them in a [pull request](https://github.com/FLAIR-Community/Fling/pulls).

## Acknowledgments

Special thanks to [@kxzxvbk](https://github.com/kxzxvbk), [@chuchugloria](https://github.com/chuchugloria), [@KeyGuo](https://github.com/KyeGuo), [@XinHao-96](https://github.com/XinHao-96), [@Ando233](https://github.com/Ando233), [@shonnyx](https://github.com/shonnyx).


## Citation
```latex
@misc{Fling,
    title={Fling: Framework for Federated Learning},
    author={Fling Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/FLAIR-Community/Fling}},
    year={2023},
}
```

## License
Fling is released under the Apache 2.0 license.
