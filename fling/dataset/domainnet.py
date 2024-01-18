from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('domainnet')
class DomainNetDataset(Dataset):

    def __init__(self, cfg: dict, domain, train: bool, train_num=105, test_num=-1):
        super(DomainNetDataset, self).__init__()

        # Load data using load_domainnet function
        train_imgs, train_labels, test_imgs, test_labels = load_domainnet(
            base_dir=cfg.data.base_dir, domain=domain, train_num=train_num, test_num=test_num
        )

        # Set class attributes
        self.domain = domain
        self.train = train
        self.train_data = list(zip(train_imgs, train_labels))
        self.test_data = list(zip(test_imgs, test_labels))

        scale = 256
        # user_defined_transform
        self.transform_train = transforms.Compose(
            [
                transforms.Resize([scale, scale]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30, 30)),
                transforms.ToTensor()
            ]
        )

        self.transform_test = transforms.Compose([transforms.Resize([scale, scale]), transforms.ToTensor()])

    def __len__(self) -> int:
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index: int) -> dict:
        # Return a specific sample from the dataset
        if self.train:
            img, label = self.train_data[index]
            img = self.transform_train(img)
        else:
            img, label = self.test_data[index]
            img = self.transform_test(img)

        return {'input': img, 'class_id': label}


def load_domainnet(base_dir, domain, train_num=105, test_num=-1):
    # load image paths and labels for *.pkl file
    train_paths, train_text_labels = np.load(
        '{}DomainNet/split/{}_train.pkl'.format(base_dir, domain), allow_pickle=True
    )
    test_paths, test_text_labels = np.load('{}DomainNet/split/{}_test.pkl'.format(base_dir, domain), allow_pickle=True)

    label_dict = {
        'bird': 0,
        'feather': 1,
        'headphones': 2,
        'ice_cream': 3,
        'teapot': 4,
        'tiger': 5,
        'whale': 6,
        'windmill': 7,
        'wine_glass': 8,
        'zebra': 9
    }

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
        img.close()

    for i in range(len(test_paths)):
        img_path = os.path.join(base_dir, test_paths[i])
        img = Image.open(img_path)
        test_imgs.append(img.copy())
        img.close()

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


if __name__ == '__main__':
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    base_dir = 'D:/VScode/FL/Data/DomainNet/'

    # Iterate over domains and create DomainNetDataset instances
    for domain in domains:
        dataset = DomainNetDataset(base_dir=base_dir, domain=domain, train=True)
        print(f"{domain} Dataset Size:", len(dataset))
        import matplotlib.pyplot as plt
        temp_show = dataset[10]
        image_for_display = temp_show['input'].permute(1, 2, 0).numpy()
        plt.imshow(image_for_display)
        plt.title(temp_show['class_id'])
        plt.show()
