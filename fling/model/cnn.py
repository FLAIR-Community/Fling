from torch import nn

from fling.utils import get_activation
from fling.utils.registry_utils import MODEL_REGISTRY


@MODEL_REGISTRY.register('cnn')
class CNNModel(nn.Module):

    def __init__(
        self,
        class_number,
        input_channel,
        dropout=0.1,
        kernel_sizes=[5, 3, 3],
        paddings=[2, 1, 1],
        hidden_dims=[32, 32, 32],
        linear_hidden_dims=[],
        activation='relu'
    ):
        super(CNNModel, self).__init__()

        self.layers = []
        self.layers.append(nn.Conv2d(input_channel, hidden_dims[0], kernel_size=kernel_sizes[0], padding=paddings[0]))
        self.layers.append(get_activation(name=activation))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=kernel_sizes[i + 1], padding=paddings[i + 1])
            )
            self.layers.append(get_activation(name=activation))
        self.layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*self.layers)

        self.glp = nn.AdaptiveAvgPool2d((1, 1))

        if len(linear_hidden_dims):
            self.mlp = []
            self.mlp.append(nn.Flatten())
            self.mlp.append(nn.Linear(hidden_dims[-1], linear_hidden_dims[0]))
            for i in range(len(linear_hidden_dims) - 1):
                self.mlp.append(get_activation(name=activation))
                self.mlp.append(nn.Linear(linear_hidden_dims[i], linear_hidden_dims[i + 1]))
            self.mlp = nn.Sequential(*self.mlp)
            self.fc = nn.Linear(linear_hidden_dims[-1], class_number)
        else:
            self.mlp = nn.Identity()
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(hidden_dims[-1], class_number))

    def forward(self, x, mode='compute-logit'):
        x = self.layers(x)
        x = self.glp(x)
        x = self.mlp(x)
        y = self.fc(x)
        if mode == 'compute-logit':
            return y
        elif mode == 'compute-feature-logit':
            return x, y
        else:
            return y
