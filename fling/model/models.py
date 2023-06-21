import torch
from torch import nn

from fling.utils.registry_utils import MODEL_REGISTRY


@MODEL_REGISTRY.register('cnn')
class CNNModel(nn.Module):

    def __init__(self, class_number, input_channel=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(180, class_number)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x


@MODEL_REGISTRY.register('mlp')
class MLPModel(nn.Module):

    def __init__(self, input_dim, class_number, hidden_units=1024):
        super(MLPModel, self).__init__()
        self.layer_input = nn.Linear(input_dim, hidden_units)
        self.layer_hidden = nn.Linear(hidden_units, class_number)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer_input(x)
        x = torch.dropout(x, 0.5, train=self.training)
        x = torch.relu(x)
        x = self.layer_hidden(x)
        return x
