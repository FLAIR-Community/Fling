from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fling.utils import VariableMonitor
from fling.utils.registry_utils import SERVER_REGISTRY
from .server_template import ServerTemplate


@SERVER_REGISTRY.register('base_server')
class BaseServer(ServerTemplate):

    def __init__(self, args: Dict, test_dataset: Dataset):
        super(BaseServer, self).__init__(args, test_dataset)
        self.test_loader = DataLoader(test_dataset, batch_size=args.learn.batch_size, shuffle=True)

    def apply_grad(self, grad: Dict, lr: float = 1.) -> None:
        state_dict = self.glob_dict
        for k in grad:
            state_dict[k] = state_dict[k] + lr * grad[k]

    def test_step(self, model, batch_data, criterion, monitor):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        o = model(batch_x)
        loss = criterion(o, batch_y)
        y_pred = torch.argmax(o, dim=-1)

        monitor.append(
            {
                'test_acc': torch.mean((y_pred == batch_y).float()).item(),
                'test_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )

    def preprocess_data(self, data):
        return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}

    def test(self, model, test_loader=None):
        if test_loader is not None:
            old_loader = self.test_loader
            self.test_loader = test_loader

        model.eval()
        model.to(self.device)

        monitor = VariableMonitor()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.test_step(model=model, batch_data=preprocessed_data, criterion=criterion, monitor=monitor)
        model.to('cpu')

        if test_loader is not None:
            self.test_loader = old_loader

        return monitor.variable_mean()
