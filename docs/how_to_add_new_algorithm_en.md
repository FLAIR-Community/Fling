# The Pipeline of Federated Learning Algorithms

In Fling, we have already supported some common federated learning algorithms (including generic federated learning and personalized federated learning). For most scenarios, users can start federated learning experiments in different settings simply by modifying the configuration file. Of course, we also fully support users in implementing more complex, customized algorithms. We will walk through the federated learning pipeline in the Fling framework step by step to help users gain a clearer understanding of the process.

## The Algorithm Workflow: Pipeline

Whether it's a generic federated learning algorithm or a personalized federated learning algorithm, the main process follows the corresponding pipeline files. A pipeline is responsible for organizing the algorithm's workflow around three main components in the Fling framework: Client, Server, and Group. The main workflow can be divided into **Component Initialization** and **Training Process**.

### Component Initialization

We explain the main steps of the pipeline in detail. Here we take the personalized federated learning pipeline `fling/pipeline/personalized_model_pipeline.py` as an example, as shown below:

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

In the initialization phase, the main tasks involve **dataset partitioning**, **initialization of major components** (Client, Server, Group), **learning rate scheduler setup**, and **launcher setup**. Below is a brief introduction to each component:

- **Dataset**: The code above includes operations for constructing and partitioning datasets (using a non-IID method). If you need to define a new dataset, you can refer to [this tutorial](https://github.com/FLAIR-Community/Fling/blob/main/docs/how_to_add_new_dataset_en.md).
- **Learning Rate Scheduler `lr_scheduler`**: This component is responsible for deciding the learning rate for each client at the beginning of each training round. For specific usage and modifications, you can refer to the corresponding section in [this documentation](https://github.com/FLAIR-Community/Fling/blob/main/docs/meaning_for_configurations_en.md).
- **Launcher `launcher`**: This component helps to parallelize the training, testing, and fine-tuning of all clients, improving execution efficiency. You can find more details about its usage and implementation [here](https://github.com/FLAIR-Community/Fling/blob/main/fling/utils/launcher_utils.py).
- **Client**: The client contains all the operations required for edge devices in federated learning, including local training, testing, fine-tuning, and uploading parameters. Common client definitions can be found in `fling/component/client`.
- **Server**: The server contains operations for the parameter server in federated learning, including parameter aggregation and global operations. Common server definitions can be found in `fling/component/server`.
- **Group**: A group logically contains a server and several clients. Its purpose is to better organize the logical relationship between clients and servers during interactions and execution, making the code easier to write. Common group definitions can be found in `fling/component/group`.

### Training Process

Following the previous section, in this subsection, we will introduce the **Training Process** part of the personalized federated learning `pipeline`:

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

In the training process section, for each communication round, the main tasks involve **client sampling** (selecting some clients to participate in the current round of training), using the `launcher` to perform **local training** on each client, **pre-aggregation/post-aggregation testing**, and using `group.aggregate()` to allow the server to organize **aggregation** among the clients. Since this is personalized federated learning, the `launcher` is also used to perform local **fine-tuning** on each client at the end.

Next, we introduce how to modify or add to the relevant components:

- For the launcher component `launcher`, we use `launcher.launch()` to organize the execution of the operation corresponding to `task_name` by each client either serially or in parallel:
  1. Currently, there are three callable modes corresponding to the values of the parameter `task_name`: `'train'`, `'test'`, and `'finetune'`, which represent the local training, testing, and fine-tuning operations of the clients, respectively. For example, if [`base_client`](https://github.com/FLAIR-Community/Fling/blob/main/fling/component/client/base_client.py) is used as the client component, the actual function that executes are the `train`, `test`, and `finetune` functions defined in  corresponding  class file.
  1. Currently, there are three callable modes corresponding to the values of the parameter `task_name`: `'train'`, `'test'`, and `'finetune'`, which represent the local training, testing, and fine-tuning operations of the clients, respectively.
  2. Specifically, in the [code](https://github.com/FLAIR-Community/Fling/blob/main/fling/utils/launcher_utils.py), the calls to the three built-in functions `train`, `test`, and `finetune` within the `client` component are defined. For example, if `base_client` is used as the client component, the actual definitions of the `train`, `test`, and `finetune` functions are found in the [`base_client`](https://github.com/FLAIR-Community/Fling/blob/main/fling/component/client/base_client.py) class file.

## Customizing and Using Algorithm Components

After gaining a detailed understanding of the pipeline, we know that to customize a federated learning algorithm, you can customize the Client, Server, and Group components, which may involve overriding original component methods, adding new attributes, or even introducing new methods. If needed, you can even customize the pipeline itself.

Here, we will explain this process using the **MOON algorithm as an example**:

### Step 1: Analyze the Components to be Defined

In the first step, we recommend analyzing the differences between your custom algorithm and the baseline algorithms (FedAvg/FedPer), or even other existing algorithms in the Fling framework. Specifically, you need to identify which parts of the new algorithm require customization compared to the existing ones.

**Using the MOON algorithm as an example:** Its main difference from FedAvg is that it introduces contrastive learning in the client’s local training phase. The implementation involves using the client's locally trained model from the previous global communication round $$w_i^{t-1}$$, the initial global model from the current round $$w_{global}^{t}$$, and the locally trained model from the current round $$w_i^t$$ to calculate the model-contrastive loss during local training. Therefore, we need to store $$w_i^{t-1}$$ and $$w_{global}^{t}$$ on each client (Client) and use them to calculate the new model-contrastive loss during training. In summary, the MOON algorithm requires a newly defined MOON-Client component.

### Step 2: Customize New Components

In this step, you can customize new components in `fling/component`, including Client, Server, and Group components.

**Using the MOON algorithm as an example:** We define a new client file for the MOON algorithm in `fling/component/client/fedmoon_client.py`. This file inherits from the basic client component class `BaseClient` defined in the same folder. It overrides methods from the base class and customizes new attributes and methods, as shown below:

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

### Step 3: Import Custom Components

Ensure you import your new components in the appropriate `__init__.py` file, like this:

```python
from .fedmoon_client import FedMOONClient
```

### Step 4: Prepare the Configuration File

Finally, prepare a configuration file to call your new component and execute the algorithm. Here's an example for MOON on CIFAR-100:

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

This is how you can customize a federated learning algorithm in Fling.