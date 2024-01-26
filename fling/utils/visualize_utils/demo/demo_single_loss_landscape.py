from easydict import EasyDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from fling import dataset
from fling.utils.visualize_utils import plot_2d_loss_landscape
from fling.utils.registry_utils import DATASET_REGISTRY


class ToyModel(nn.Module):
    """
    Overview:
        A toy model for demonstrating attacking results.
    """

    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flat(self.pool(x))
        return self.fc(x)


if __name__ == '__main__':
    # Step 1: prepare the dataset.
    dataset_config = EasyDict(dict(data=dict(data_path='./data/cifar10', transforms=dict())))
    dataset = DATASET_REGISTRY.build('cifar10', dataset_config, train=False)

    # Test dataset is for generating loss landscape.
    test_dataset = [dataset[i] for i in range(100)]
    test_dataloader = DataLoader(test_dataset, batch_size=100)

    # Step 2: prepare the model.
    model = resnet18(pretrained=False, num_classes=10)

    # Step 3: train the randomly initialized model.
    dataloader = DataLoader(dataset, batch_size=100)
    device = 'cuda'
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(10):
        for _, (data) in enumerate(dataloader):
            data_x, data_y = data['input'], data['class_id']
            data_x, data_y = data_x.to(device), data_y.to(device)
            pred_y = model(data_x)
            loss = criterion(pred_y, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.to('cpu')

    # Step 4: plot the loss landscape after training the model.
    # Only one line of code for visualization!
    plot_2d_loss_landscape(
        model=model,
        dataloader=test_dataloader,
        device='cuda',
        caption='Loss Landscape Trained',
        save_path='./landscape.pdf',
        noise_range=(-1, 1),
        resolution=30,
        log_scale=True,
        max_val=20,
    )
