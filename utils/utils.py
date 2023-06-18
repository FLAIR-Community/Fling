import numpy as np
import os
import time

from torch.utils.tensorboard import SummaryWriter


def client_sampling(client_ids, sample_rate):
    participated_clients = np.array(client_ids)
    participated_clients = sorted(
        list(np.random.choice(participated_clients, int(sample_rate * participated_clients.shape[0]), replace=False))
    )
    return participated_clients


class Logger(SummaryWriter):

    def __init__(self, path):
        super(Logger, self).__init__(path)
        self.txt_logger_path = os.path.join(path, 'base_logger_output.txt')

    def logging(self, s):
        print(s)
        with open(self.txt_logger_path, mode='a') as f:
            f.write('[' + time.asctime(time.localtime(time.time())) + ']    ' + s + '\n')


class VariableMonitor:

    def __init__(self, keys):
        self.length = 0
        self.dic = {k: [] for k in keys}

    def append(self, item, weight=1):
        self.length += weight
        for k in item.keys():
            self.dic[k].append(weight * item[k])

    def variable_mean(self):
        return {k: sum(self.dic[k]) / self.length for k in self.dic.keys()}
