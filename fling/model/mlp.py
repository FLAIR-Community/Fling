from torch import nn

from fling.utils.registry_utils import MODEL_REGISTRY
from fling.utils import get_activation


@MODEL_REGISTRY.register('mlp')
class MLPModel(nn.Module):

    def __init__(
        self,
        input_dim,
        class_number,
        dropout=0.1,
        hidden_units=[64, 128, 256],
        activation='relu',
    ):
        super(MLPModel, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_units[0]))
        self.layers.append(get_activation(name=activation))
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.layers.append(get_activation(name=activation))
        self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(hidden_units[-1], class_number))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
