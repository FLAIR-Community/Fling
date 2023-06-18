import torch
from torch.utils.data import DataLoader

from utils import get_loss, VariableMonitor


class Server:

    def __init__(self, args, device, test_dataset):
        self.args = args
        self.glob_dict = None

        self.device = device if device >= 0 else 'cpu'
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    def apply_grad(self, grad, lr=1.):
        state_dict = self.glob_dict
        for k in grad:
            state_dict[k] = state_dict[k] + lr * grad[k]

    def test(self, model, loss, test_loader=None):
        if test_loader is not None:
            old_loader = self.test_loader
            self.test_loader = test_loader

        model.eval()
        model.to(self.device)

        monitor = VariableMonitor(['acc', 'loss'])
        criterion = get_loss(loss)

        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(self.test_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                o = self.model(batch_x)
                loss = criterion(o, batch_y)
                y_pred = torch.argmax(o, dim=-1)

                monitor.append(
                    {
                        'acc': torch.mean((y_pred == batch_y).float).item(),
                        'loss': loss.item()
                    }, weight=batch_y.shape[0]
                )
        model.to('cpu')

        if test_loader is not None:
            self.test_loader = old_loader

        return monitor.variable_mean()
