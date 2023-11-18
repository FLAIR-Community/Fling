<img src="https://github.com/kxzxvbk/Fling/blob/main/README.assets/fling.png" style="zoom: 35%;" />

## Fling

**Fling** is a research platform for Federated Learning using PyTorch as backend. 

Its goal is to **simulate the distributed learning process of Federated Learning on single or multiple machines**, providing a **fair testing platform** for the performance of various federated learning algorithms on different datasets. It is mainly based on the Python language and uses the PyTorch framework as the backend module for deep learning, supporting a variety of federated learning algorithms and commonly used federated learning datasets.

It mainly supports:

- Generic Federated Learning methods, such as FedAvg.
- Personalized Federated Learning methods, such as FedPer.

## Installation

Firstly, it is recommended to install PyTorch manually with a suitable version (specifically 1.1.0 or higher). However, using PyTorch version 2.0.0 or later is preferred due to its better computational efficiency. Instructions for installation can be found at this [link](https://pytorch.org/get-started/locally/).

After the first step, you can simply install the latest version of Fling with the following command by using Git:

```bash
git clone https://github.com/kxzxvbk/Fling
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

**Other tutorials:**

[Overall Framework of Fling](https://github.com/kxzxvbk/Fling/blob/main/docs/framework_for_fling_en.md) | [Fling 整体框架](https://github.com/kxzxvbk/Fling/blob/main/docs/framework_for_fling_zh.md)

[Meaning for Each Configuration Key](https://github.com/kxzxvbk/Fling/blob/main/docs/meaning_for_configurations_en.md) | [配置文件各字段含义](https://github.com/kxzxvbk/Fling/blob/main/docs/meaning_for_configurations_zh.md)

[How to Add a New FL Algorithm](https://github.com/kxzxvbk/Fling/blob/main/docs/how_to_add_new_algorithm_en.md) | [如何自定义联邦学习算法](https://github.com/kxzxvbk/Fling/blob/main/docs/how_to_add_new_algorithm_zh.md)

[How to Add a New Dataset](https://github.com/kxzxvbk/Fling/blob/main/docs/how_to_add_new_dataset_en.md) | [如何添加新数据集](https://github.com/kxzxvbk/Fling/blob/main/docs/how_to_add_new_dataset_zh.md)

[Cli Usage in Fling](https://github.com/kxzxvbk/Fling/blob/main/docs/cli_en.md) | [Fling 的 CLI 使用](https://github.com/kxzxvbk/Fling/blob/main/docs/cli_zh.md)

## Feature

- Support for a variety of algorithms and datasets.
- Support multiprocessing training on each client for better efficiency.
- Using single GPU to simulate Federated Learning process (multi-GPU version will be released soon).

## Supported Algorithms

### Generic Federated Learning

**FedAvg:** [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

**FedProx:** [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)

**FedMOON:** [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf)

### Personalized Federated Learning

**FedPer:** [Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818v1.pdf)

**FedBN:** [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/pdf/2102.07623.pdf)

**FedRoD:** [On Bridging Generic and Personalized Federated Learning for Image Classification](https://openreview.net/pdf?id=I1hQbx10Kxn)

**pFedSD:** [Personalized Edge Intelligence via Federated Self-Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/9964434)

**FedCAC:** [Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration]()

## Feedback and Contribution

- For any bugs, questions, feature requirements, feel free to propose them in [issues](https://github.com/kxzxvbk/Fling/issues)
- For any contributions that can improve Fling (more algorithms or better system design), we warmly welcome you to propose them in a [pull request](https://github.com/kxzxvbk/Fling/pulls).

## Acknowledgments

Special thanks to [@chuchugloria](https://github.com/chuchugloria), [@shonnyx](https://github.com/shonnyx), [@XinghaoWu](https://github.com/XinghaoWu), 


## Citation
```latex
@misc{Fling,
    title={Fling: Framework for Federated Learning},
    author={Fling Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/kxzxvbk/Fling}},
    year={2023},
}
```

## License
Fling is released under the Apache 2.0 license.
