"""
Run this file to download Mini-Imagenet dataset and prepare it for training / testing.
"""
import os
from typing import Dict, List
import pickle
import copy
import random
import numpy as np


def dict2list(dic: Dict) -> List:
    """
    Convert the dict data to list form.
    """
    res = []
    for k, v in dic.items():
        for img in v:
            res.append((img, k))
    random.shuffle(res)
    return res


if __name__ == '__main__':
    os.system('mkdir mini-imagenet')
    os.system('cd mini-imagenet')
    os.system('wget https://data.deepai.org/miniimagenet.zip')
    os.system('unzip miniimagenet.zip')

    with open("mini-imagenet-cache-train.pkl", "rb") as f:
        train_data = pickle.load(f)

    with open("mini-imagenet-cache-val.pkl", "rb") as f:
        val_data = pickle.load(f)

    with open("mini-imagenet-cache-test.pkl", "rb") as f:
        test_data = pickle.load(f)

    data_load_all = {}
    data_load_all['image_data'] = np.concatenate(
        (train_data['image_data'], val_data['image_data'], test_data['image_data'])
    )
    data_load_all['class_dict'] = copy.deepcopy(train_data['class_dict'])
    data_load_all['class_dict'].update(val_data['class_dict'])
    data_load_all['class_dict'].update(test_data['class_dict'])

    # 100 classes and 60000 images
    assert len(data_load_all['class_dict'].keys()) == 100
    assert sum([len(data_load_all['class_dict'][k]) for k in data_load_all['class_dict'].keys()]) == 60000

    total_res = {i: [] for i in range(100)}
    class_name2idx = {name: idx for idx, name in enumerate(list(data_load_all['class_dict'].keys()))}
    for k, img_idxes in data_load_all['class_dict'].items():
        random.shuffle(img_idxes)
        # Each class has 600 images.
        assert len(img_idxes) == 600
        for img_idx in img_idxes:
            total_res[class_name2idx[k]].append(copy.deepcopy(data_load_all['image_data'][img_idx]))

    train_res = dict2list({k: v[:500] for k, v in total_res.items()})
    val_res = dict2list({k: v[500:] for k, v in total_res.items()})

    assert len(train_res) == 50000 and len(val_res) == 10000

    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_res, f)

    with open('val_dataset.pkl', 'wb') as f:
        pickle.dump(val_res, f)

    os.system('rm mini-imagenet-cache-train.pkl')
    os.system('rm mini-imagenet-cache-val.pkl')
    os.system('rm mini-imagenet-cache-test.pkl')
