"""
This file will convert the raw Imagenet dataset to lmdb files for faster data loading. \
Assume that you have already downloaded the ImageNet dataset from https://image-net.org/, the files is organized as:
.
├── imagenet
│   ├── train
│   │   ├── n1440764
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── val
│   │   ├── n1440764
│   │   │   ├── aaa.jpg
│   │   │   ├── bbb.jpg
│   │   │   ├── ...
│   │   ├── ...
"""
import os
import pickle

import lmdb
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def folder2lmdb(dpath: str, name: str, write_frequency: int = 5000, num_workers: int = 16) -> None:
    """
    Overview:
        Convert an image folder into lmdb file.
    Arguments:
        - dpath: The path of the image-folder.
        - name: The split of the dataset. Options: ["train", "val"].
        - write_frequency: The frequency of writing lmdb file, default as 5000.
        - num_workers: Number of workers for processing the data, default to be 16.
    """
    directory = os.path.expanduser(os.path.join(dpath, name))
    print("Loading dataset from %s" % directory)
    trans = transforms.Compose([transforms.Resize((256, 256))])
    dataset = ImageFolder(directory, transform=trans)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = os.path.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)

    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)

    for idx, data in tqdm(enumerate(data_loader)):
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    folder2lmdb('./imagenet', num_workers=16, name='val')
    folder2lmdb('./imagenet', num_workers=16, name='train')
