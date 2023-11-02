from torch import nn
import torch.nn.functional as F

from fling.utils import get_activation
from fling.utils.registry_utils import MODEL_REGISTRY


class CNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(CNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

@MODEL_REGISTRY.register('moon_models')
class FedMOONModel(nn.Module):

    def __init__(self, base_model, out_dim, class_number, input_channel, net_configs=None):
        super(FedMOONModel, self).__init__()

        if base_model == 'cnn':
            self.features = CNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=class_number)
            num_ftrs = 84

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, class_number)

    def forward(self, x):
        h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        #print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return x, y