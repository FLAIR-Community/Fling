import torch

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.utils.torch_utils import balanced_softmax_loss
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedrod_client')
class FedRoDClient(BaseClient):
    """
    Overview:
        This class is the implementation FedRoD introduced in ON BRIDGING GENERIC AND PERSONALIZED FEDERATED LEARNING
        FOR IMAGE CLASSIFICATION <link https://openreview.net/pdf?id=I1hQbx10Kxn link>.
    """

    def __init__(self, *args, **kwargs):
        super(FedRoDClient, self).__init__(*args, **kwargs)

        # Calculate the ``sample_per_client``
        self.spc = [0] * self.args.model.class_number
        for data in self.train_dataloader.dataset:
            self.spc[data['class_id']] += 1
        self.spc = torch.LongTensor(self.spc)

    def train_step(self, batch_data, criterion, monitor, optimizer):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        g_head, p_head = self.model(batch_x)

        # Calculate loss for p_head and g_head respectively.
        g_loss = balanced_softmax_loss(batch_y, g_head, self.spc.to(self.device))
        p_loss = criterion(p_head, batch_y)
        loss = g_loss + p_loss

        # Prediction should use p_head.
        y_pred = torch.argmax(p_head, dim=-1)

        monitor.append(
            {
                'train_acc': torch.mean((y_pred == batch_y).float()).item(),
                'train_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def finetune_step(self, batch_data, criterion, monitor, optimizer):
        return self.train_step(batch_data, criterion, monitor, optimizer)

    def test_step(self, batch_data, criterion, monitor):
        batch_x, batch_y = batch_data['x'], batch_data['y']
        g_head, p_head = self.model(batch_x)

        # Calculate loss for p_head and g_head respectively.
        p_loss = criterion(p_head, batch_y)
        loss = p_loss

        # Prediction should use p_head.
        y_pred = torch.argmax(p_head, dim=-1)

        monitor.append(
            {
                'test_acc': torch.mean((y_pred == batch_y).float()).item(),
                'test_loss': loss.item()
            },
            weight=batch_y.shape[0]
        )

    def preprocess_data(self, data):
        return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}
