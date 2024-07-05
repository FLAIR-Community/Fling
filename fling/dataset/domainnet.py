import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('domainnet')
class DomainNetDataset(data.Dataset):
    r"""
    Implementation for DomainNet dataset. 
    """
    def __init__(self, cfg: dict, domain: str,  train: bool): # cfg can be loaded
        super(DomainNetDataset, self).__init__()
        self.train_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor()
    ])
        self.test_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])
        self.train = train
        self.cfg = cfg
        self.train_imgs, self.train_labels, self.test_imgs, self.test_labels = \
        load_domainnet(base_dir=self.cfg.data.base_dir, domain=domain, train_num=500, test_num=300000)

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else: 
            return len(self.test_labels)
        
    def __getitem__(self, index: int) -> dict:
        if self.train:
            img, label = self.train_imgs[index], self.train_labels[index]
            if len(img.split()) != 3:
                img = transforms.Grayscale(num_output_channels=3)(img)
            img = self.train_transform(img)
        else: 
            img, label = self.test_imgs[index], self.test_labels[index]
            if len(img.split()) != 3:
                img = transforms.Grayscale(num_output_channels=3)(img)
            img = self.test_transform(img)


        # return {'input': img, 'class_id': label}
        return img, label

def load_domainnet(base_dir, domain, train_num=105, test_num=-1):
    # load image paths and lables for *.pkl file
    train_paths, train_text_labels = np.load('{}DomainNet/split/{}_train.pkl'.format(base_dir, domain), allow_pickle=True)
    test_paths, test_text_labels = np.load('{}DomainNet/split/{}_test.pkl'.format(base_dir, domain), allow_pickle=True)

    label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}

    # transform text labels to digit labels
    train_labels = [label_dict[text] for text in train_text_labels]
    test_labels = [label_dict[text] for text in test_text_labels]

    train_imgs = []
    test_imgs = []

    # load images in train dataset
    for i in range(len(train_paths)):
        img_path = os.path.join(base_dir, train_paths[i])
        img = Image.open(img_path)
        train_imgs.append(img.copy())

    for i in range(len(test_paths)):
        img_path = os.path.join(base_dir, test_paths[i])
        img = Image.open(img_path)
        test_imgs.append(img.copy())

    if train_num <= len(train_imgs):
        train_imgs = train_imgs[:train_num]
        train_labels = train_labels[:train_num]

    if test_num <= len(test_imgs):
        test_imgs = test_imgs[:test_num]
        test_labels = test_labels[:test_num]

    print('Load {} Dataset...'.format(domain))
    print('Train Dataset Size:', len(train_imgs))
    print('Test Dataset Size:', len(test_imgs))

    return train_imgs, train_labels, test_imgs, test_labels


