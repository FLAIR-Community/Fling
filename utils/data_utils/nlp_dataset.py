import torch
import numpy as np


class NLPDataset:

    def __init__(self, data, batch_size, block_size, device, batch_num=20000):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.batch_num = batch_num

    def __len__(self):
        return self.batch_num

    def __getitem__(self, item):
        ix = torch.randint(len(self.data) - self.block_size, size=(1, ))
        x = torch.from_numpy((self.data[ix[0]:ix[0] + self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[ix[0] + 1:ix[0] + 1 + self.block_size]).astype(np.int64))
        return x, y
