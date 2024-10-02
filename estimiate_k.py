import torch.nn
import fling.dataset
import argparse
import gc
import random
import tqdm
from easydict import EasyDict
from fling.utils.registry_utils import MODEL_REGISTRY, DATASET_REGISTRY
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--train-epo', type=int, required=True)
parser.add_argument('--number', type=int, required=True)
args = parser.parse_args()


def vectorize_grad():
    params = dict(model.named_parameters())
    total_grad = []
    for k in keys:
        total_grad.append(torch.flatten(params[k].grad).detach().cpu().clone())
    return torch.cat(total_grad, dim=0)


dataset_config = EasyDict(dict(data=dict(data_path='C:\\Users\\23207\\Desktop\\fling\\Fling\\data\\CIFAR10',
                                         transforms=dict())))
dataset = DATASET_REGISTRY.build('cifar10', dataset_config, train=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model_arg = EasyDict(dict(
        name=args.model,
        input_channel=3,
        class_number=10,
    ))
model_name = model_arg.pop('name')
model = MODEL_REGISTRY.build(model_name, **model_arg).cuda()
keys = list(dict(model.named_parameters()).keys())

criterion = torch.nn.CrossEntropyLoss()
grad_matrix = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(args.train_epo):
    for batch_data in dataloader:
        batch_x, batch_y = batch_data['input'].cuda(), batch_data['class_id'].cuda()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

gc.collect()

for batch_data in dataloader:
    batch_x, batch_y = batch_data['input'].cuda(), batch_data['class_id'].cuda()
    pred = model(batch_x)
    loss = criterion(pred, batch_y)
    model.zero_grad()
    loss.backward()
    grad_matrix.append(vectorize_grad())

model = model.to('cpu')
del model
gc.collect()

grad_matrix = torch.stack(grad_matrix, dim=0).cuda()
n_dim = grad_matrix.shape[1]

M_frac = 0.125


def get_rand_var():
    n_frac = int(M_frac * n_dim)
    choices = random.sample(list(range(n_dim)), k=n_frac)
    sub_matrix = grad_matrix[:, choices]
    mean = torch.mean(sub_matrix, dim=0)
    return torch.norm(sub_matrix - mean).item()


min_var = 1e30
max_var = 0

for _ in tqdm.tqdm(range(args.number)):
    tmp = get_rand_var()
    min_var = min(min_var, tmp)
    max_var = max(max_var, tmp)
print(f'Final result: {max_var / min_var}')
