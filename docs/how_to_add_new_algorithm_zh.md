# 联邦学习算法的流水线

在 Fling 中，我们已经支持一些常见的联邦学习算法（包括通用联邦学习和个性化联邦学习），对于大多数场景，用户只需要通过修改配置文件就可以在不同的场景下开启联邦学习实验。当然，我们也完全支持用户进行一些复杂的定制化算法实现。我们将一步步梳理在 Fling 框架中联邦学习的工作流水线，来帮助用户对其中的流程有更清晰的认识。

## 算法的运行流程：Pipeline（流水线）

无论是通用联邦学习算法还是个性化联邦学习算法，其运行的主体都遵照对应的 pipeline 进行。pipeline 的作用是围绕 Fling 框架中的三个主要组件：客户端（Client）、服务器（Server）和群组（Group）来组织算法的运行流程。主要流程可以分为**组件初始化**和**训练主体**两个部分。

### 组件初始化

接下来我们将分段对 pipeline 的主体进行详细解释。这里以个性化联邦学习的流水线 `fling/pipeline/personalized_model_pipeline.py ` 为例，如下所示：

```python
def personalized_model_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for personalized federated learning. Under this setting, models of each client is different.
       The final performance of is calculated by averaging the local model in each client.
       Typically, each local model is tested using local test dataset.
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed=seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset.
    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)
    # Split dataset into clients.
    train_sets = data_sampling(train_set, args, seed, train=True)
    test_sets = data_sampling(test_set, args, seed, train=False)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)
    for i in range(args.client.client_num):
        group.append(get_client(args=args, client_id=i, train_dataset=train_sets[i], test_dataset=test_sets[i]))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.optimizer.lr, args=args.learn.scheduler)

    # Setup launcher.
    launcher = get_launcher(args)
```

在初始化部分中，主要涉及**数据集的划分**、**主要组件**（客户端、服务器、群组）**的初始化**、**学习率调度器的设置**、**启动器`launcher`的设置**。接下来是对各个组件的简单介绍：

- 数据集：上述代码块包含了构造数据集、划分数据集（采用 non-IID 方式）等操作。如果需要新定义数据集，可以参考[此教程](https://github.com/FLAIR-Community/Fling/blob/main/docs/how_to_add_new_dataset_zh.md)。
- 学习率调度器 ``lr_scheduler``：这一组件的作用是在每个训练轮次开始时，为每个客户端决定学习率。具体的使用和修改方式，可参考[此文档](https://github.com/FLAIR-Community/Fling/blob/main/docs/meaning_for_configurations_zh.md)中的对应部分。
- 启动器 `launcher`：这一组件的作用是可以并行化地安排所有客户端进行训练、测试、微调的组件，提高执行效率。具体的使用方式和实现可参考[此处](https://github.com/FLAIR-Community/Fling/blob/main/fling/utils/launcher_utils.py)。
- 客户端：客户端包含了联邦学习中边缘设备所需要进行的所有操作，包括本地训练、测试、微调，上传参数等。常见的客户端定义在了 ``fling/component/client`` 中。
- 服务器：服务器包含了联邦学习中参数服务器的操作，包括参数聚合、全局操作等。常见的服务器定义在了 ``fling/component/server`` 中。
- 群组：一个群组逻辑上包含了一个服务器和若干个客户端。其设计的目的是能够更好地组织客户端和服务器交互和执行的逻辑关系，便于代码的编写。常见的群组定义在了 ``fling/component/group`` 中

### 训练主体

承接上文，在这小节中我们对个性化联邦学习的 `pipeline` 的**训练主体**部分进行介绍：

```python
    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        # Initialize variable monitor.
        train_monitor = VariableMonitor()
        # Random sample participated clients in each communication round.
        participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)
        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)
        # Local training for each participated client and add results to the monitor.
        # Use multiprocessing for acceleration.
        train_results = launcher.launch(
            clients=[group.clients[j] for j in participated_clients], lr=cur_lr, task_name='train'
        )
        for item in train_results:
            train_monitor.append(item)
        # Testing
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()
            # Testing for each client and add results to the monitor
            # Use multiprocessing for acceleration.
            test_results = launcher.launch(
                clients=[group.clients[j] for j in range(args.client.client_num)], task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)
            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()
            # Logging test variables.
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=mean_test_variables, rnd=i)
        # Aggregate parameters in each client.
        trans_cost = group.aggregate(i)
        # Logging for train variables.
        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update({'trans_cost(MB)': trans_cost / 1e6, 'lr': cur_lr})
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)
        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()
            # Testing for each client and add results to the monitor
            # Use multiprocessing for acceleration.
            test_results = launcher.launch(
                clients=[group.clients[j] for j in range(args.client.client_num)], task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)
            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()
            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=mean_test_variables, rnd=i)
            # Saving model checkpoints.
            torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))

    # Fine-tuning
    # Fine-tune model on each client and collect all the results.
    finetune_results = launcher.launch(
        clients=[group.clients[j] for j in range(args.client.client_num)],
        lr=cur_lr,
        finetune_args=args.learn.finetune_parameters,
        task_name='finetune'
    )
    # Logging fine-tune results
    for key in finetune_results[0][0].keys():
        for eid in range(len(finetune_results[0])):
            tmp_mean = sum([finetune_results[cid][eid][key]
                            for cid in range(len(finetune_results))]) / len(finetune_results)
            logger.add_scalar(f'finetune/{key}', tmp_mean, eid)
```

在训练主体部分中，每个通信轮次内，主要涉及**客户端采样**（选取部分客户端参与本轮训练）、使用 `launcher` 对各个客户端进行**本地训练**、**聚合前/聚合后测试**，以及使用 `group.aggregate()` 实现服务器组织客户端的**聚合**。由于是个性化联邦学习，在最后还使用 `launcher` 对各个客户端进行本地微调。

接下来，我们介绍如何对相应的组件部分进行修改/添加：

- 关于启动器组件 `launcher` ，我们使用 `launcher.launch()` 来组织各个客户端串行/并行地执行 `task_name` 对应的操作：
  1. 我们目前共有三种可调用的模式，对应参数 `task_name` 的 `'train'` 、`'test'` 、`'finetune'` 值，分别表示执行客户端的本地训练、测试、微调操作。例如，如果使用  [`base_client`](https://github.com/FLAIR-Community/Fling/blob/main/fling/component/client/base_client.py) 作为客户端组件，那么上述三种模式则分别实际调用了相应类的 `train` 、`test` 、`finetune` 函数。
  3. 我们引入 `launcher` 的目的是实现对各个客户端执行相应操作的并行化。相关配置参数的定义可参照 [Fling/flzoo/default_config.py](https://github.com/FLAIR-Community/Fling/blob/main/flzoo/default_config.py) 中的 `launcher.name` 字段。
- 如果涉及到自定义 `logger` 中结果的呈现，可以针对 `train_monitor`、`test_monitor` 以及 `logger.add_scalars_dict()` 部分的操作进行修改。

## 自定义算法组件并使用

在详细了解了 pipeline 之后，我们知道要自定义联邦学习算法，可以对客户端（Client）、服务器（Server）和群组（Group）进行自定义，即覆写原组件的方法、添加新属性甚至新方法等。如果需要，甚至可以对流水线（Pipeline）进行自定义。

这里结合 **MOON 算法为例**进行说明：

### 步骤 1：分析需要新定义的组件

在最开始的一步中，我们建议您可以先分析自定义算法与基线算法（FedAvg/FedPer），甚至其他 Fling 框架内已有算法之间的差异。具体而言，就是要先确定新算法相对已有算法而言，需要在哪些部分进行自定义。

**以 MOON 算法为例：**它与 FedAvg 的主要不同之处在于，它在客户端本地训练的部分引入了对比学习。而具体的实现，需要用到本客户端在上一个全局通信轮次本地训练后的局部模型 $$w_i^{t-1}$$、本全局通信轮次初始的全局模型 $$w_{global}^{t}$$、本全局通信轮次正在进行本地训练的局部模型 $$w_i^t$$，来计算本地训练的 model-contrastive loss 。因此，我们可以在每个客户端（Client）中对 $$w_i^{t-1}$$ 和 $$w_{global}^{t}$$ 进行暂存，并且在训练（train）的部分利用它们来计算新的 model-contrastive loss 。总结来看，MOON 算法需要新定义一个 MOON-Client 组件。

### 步骤 2：自定义新组件

在这一步中，您可以在 `fling/component` 中自定义新组件，其中包括客户端（client）、服务器（server）和群组（group）三种组件。

**以 MOON 算法为例：**我们在 `fling/component/client` 中为 MOON 算法定义了新的客户端文件 `fling/component/client/fedmoon_client.py`，它对该文件夹中定义的基本客户端组件类 `BaseClient` 进行继承，对基本类中定义的方法进行覆写，并自定义了新属性和新方法，如下所示：

```python
import copy
import torch
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedmoon_client')
class FedMOONClient(BaseClient):
    r"""
    Overview:
        This class is the base implementation of client of FedMOON introduced in: Model-Contrastive Federated Learning
     <link https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf link>.
    """

    def __init__(self, *args, **kwargs):
        super(FedMOONClient, self).__init__(*args, **kwargs)
        # The weight of fedmoon loss.
        self.mu = self.args.learn.mu
        # The temperature parameter of fedmoon.
        self.t = self.args.learn.temperature
        # The variable to store the global model.
        self.glob_model = None
        # The variable to store the previous models.
        self.prev_models = []
        # The max length of prev_models
        self.queue_len = self.args.learn.queue_len

    def _store_prev_model(self, model: nn.Module) -> None:
        r"""
        Overview:
            Store the prev model for fedmoon loss calculation.
        """
        if len(self.prev_models) >= self.queue_len:
            self.prev_models.pop(0)
        self.prev_models.append(copy.deepcopy(model))

    def _store_global_model(self, model: nn.Module) -> None:
        r"""
        Overview:
            Store the global model for fedmoon loss calculation.
        """
        self.glob_model = copy.deepcopy(model)

    def train_step(self, batch_data, criterion, monitor, optimizer):
        r"""
        Overview:
            Training step. The loss of fedmoon should be added to the original loss.
        """
        batch_x, batch_y = batch_data['x'], batch_data['y']
        z, o = self.model(batch_x, mode='compute-feature-logit')
        main_loss = criterion(o, batch_y)
        # Calculate fedmoon loss.
        cos = nn.CosineSimilarity(dim=-1)
        self.glob_model.to(self.device)
        with torch.no_grad():
            z_glob, _ = self.glob_model(batch_x, mode='compute-feature-logit')
        z_i = cos(z, z_glob)
        logits = z_i.reshape(-1, 1)
        for prev_model in self.prev_models:
            prev_model.to(self.device)
            with torch.no_grad():
                z_prev, _ = prev_model(batch_x, mode='compute-feature-logit')
            nega = cos(z, z_prev)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.t
        labels = torch.zeros(batch_x.size(0)).to(self.device).long()
        fedmoon_loss = criterion(logits, labels)
        # Add the main loss and fedmoon loss together.
        loss = main_loss + self.mu * fedmoon_loss
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                'train_acc': torch.mean((y_pred == batch_y).float()).item(),
                'main_loss': main_loss.item(),
                'fedmoon_loss': self.mu * fedmoon_loss.item(),
                'total_loss': loss.item(),
            },
            weight=batch_y.shape[0]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for prev_model in self.prev_models:
            prev_model.to('cpu')

    def train(self, lr, device=None):
        r"""
        Overview:
            Training function. The global model and prev model should be stored.
        """
        self._store_global_model(self.model)
        mean_monitor_variables = super(FedMOONClient, self).train(lr=lr, device=device)
        # Reset the global model to save memory.
        del self.glob_model
        # Store the current model as prev model in next round
        self._store_prev_model(self.model)
        return mean_monitor_variables
```

在上面的代码示例中，关于新定义组件有几点值得注意：

- 注册新组件：在最开始使用代码 `@CLIENT_REGISTRY.register('fedmoon_client')` 对新定义的 `fedmoon_client` 进行注册，以便在后续的配置文件中使用它。
- 继承原组件：您自定义的组件应继承原有的基础组件。如此处新注册的 `fedmoon_client` 应为 `fling.component.client.BaseClient` 的子类。
- 添加新属性：在 `__init__()` 中对 MOON 算法所需要的新属性如 `self.mu` 、`self.glob_model` 、`self.prev_models` 等进行了定义，以便在后续新定义的方法中使用。
- 定义新方法：根据前文分析，MOON 算法在客户端本地训练中需要对 $$w_i^{t-1}$$ 和 $$w_{global}^{t}$$ 进行暂存，因此上述代码中新定义了 `_store_prev_model()` 和 `_store_global_model()` 方法来满足此需求。
- 覆写原方法：根据前文分析，MOON 算法在客户端本地训练的部分引入了对比学习的 model-contrastive loss ，因此上述代码中针对 `BaseClient` 中已经定义的 `train()` 和 `train_step()` 方法进行了覆写，调用了新定义的方法，也使用 `super()` 调用了父类中的方法。

上述例子以 MOON 算法为例，展示了如何自定义新的 `client` 组件。同理，如果您需要的话，也可以对 [`fling/component`](https://github.com/FLAIR-Community/Fling/tree/main/fling/component) 中的 `server` 或者 `group` 组件进行自定义。更进一步地，您也可以在 [`fling/pipeline`](https://github.com/FLAIR-Community/Fling/tree/main/fling/pipeline) 中定义自己的新 `pipeline` 。

### 步骤 3：导入自定义文件

当您添加一个新的组件时，别忘了在相应目录下的 `__init__.py` 文件中导入它。

**以 MOON 算法为例：**我们在 `fling/component/client` 中为 MOON 算法定义了新的客户端文件 `fling/component/client/fedmoon_client.py` 后，需要在 `fling.component.client.__init__.py` 中导入它：

```python
from .fedmoon_client import FedMOONClient
```

同理，如果您在 `fling/component/server`  或是 `fling/component/group` 中定义了新组件，也请在对应目录下的 `__init__.py` 文件中导入它。如果您在 `fling/pipeline` 目录中定义了新的 `pipeline` 同样需要对其进行导入。

### 步骤 4：准备配置文件

在完成前面的步骤之后，您现在可以编写配置文件，调用您自定义的新组件，来执行对应的新算法了！

**以 MOON 算法为例：** 我们在配置文件 `flzoo/cifar100/cifar100_fedmoon_resnet_config.py` 中编写了在 CIFAR-100 数据集上执行 MOON 算法的配置，如下所示：

```python
from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='cifar100',
        data_path='./data/CIFAR100',
        sample_method=dict(name='dirichlet', alpha=0.5, train_num=500, test_num=100)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=10,
        global_eps=40,
        batch_size=64,
        optimizer=dict(name='sgd', lr=0.01, momentum=0.9),
        # The weight of fedmoon loss.
        mu=1,
        temperature=0.5,
        queue_len=1,
    ),
    model=dict(name='resnet8', input_channel=3, class_number=100),
    client=dict(name='fedmoon_client', client_num=10),
    server=dict(name='base_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=3, logging_path='./logging/cifar100_fedmoon_resnet_dirichlet_05')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=0)
```

其中：

- `client.name` 的键值是您在 **步骤 2** 中注册的新组件的名称 `fedmoon_client` 。同理，如果您新定义了 `server` 或者 `group` 组件需要使用，也应该修改对应的 `server.name` 或者 `group.name` 的键值。
- 字典 `learn` 中可以传递新定义的参数。比如此处设置了新属性 `mu` 、`temperature` 、`queue_len` 的键值，其在 **步骤 2** 中定义的 `fedmoon_client.__init__()` 中被取用。
- 如果您定义并注册了新的 `pipeline` ，可以在最后的 `__main__` 函数中将其导入并执行。

至此，整个添加自定义联邦学习算法的流程就完成了，您可以使用 `python` 运行您自定义的配置文件来执行新算法。
