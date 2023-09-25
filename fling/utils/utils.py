import numpy as np
import os
import time
from typing import Iterable, Dict, List
from prettytable import PrettyTable

from torch.utils.tensorboard import SummaryWriter


def client_sampling(client_ids: Iterable, sample_rate: float) -> List:
    participated_clients = np.array(client_ids)
    participated_clients = sorted(
        list(np.random.choice(participated_clients, int(sample_rate * participated_clients.shape[0]), replace=False))
    )
    return participated_clients


class Logger(SummaryWriter):

    def __init__(self, path: str):
        super(Logger, self).__init__(path)
        self.txt_logger_path = os.path.join(path, 'txt_logger_output.txt')

    def logging(self, s: str) -> None:
        print(s)
        with open(self.txt_logger_path, mode='a') as f:
            f.write('[' + time.asctime(time.localtime(time.time())) + ']    ' + s + '\n')

    def round(self, num: object, length: int = 5) -> object:
        if isinstance(num, float):
            return round(num, length)
        else:
            return num

    def add_scalars_dict(self, prefix: str, dic: dict, rnd: int) -> None:
        # Log in the tensorboard.
        for k in dic.keys():
            self.add_scalar(f'{prefix}/{k}', dic[k], rnd)

        # Log in the command line.
        tupled_dict = [(k, v) for k, v in dic.items()]
        tb = PrettyTable(["Phase", "Round"] + [ii[0] for ii in tupled_dict])
        tb.add_row([prefix, rnd] + [self.round(ii[1]) for ii in tupled_dict])
        txt_info = str(tb)
        self.logging(txt_info)


class VariableMonitor:

    def __init__(self):
        self.length = {}
        self.dic = {}

    def append(self, item: dict, weight: float = 1) -> None:
        for k in item.keys():
            if k not in self.dic.keys():
                self.dic[k] = []
                self.length[k] = 0
            self.dic[k].append(weight * item[k])
            self.length[k] += weight

    def variable_mean(self) -> Dict:
        return {k: sum(self.dic[k]) / self.length[k] for k in self.dic.keys()}
