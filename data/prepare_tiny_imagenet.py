"""
Run this file to download Tiny-Imagenet dataset and prepare it for training / testing.
"""
import os
from shutil import move
from os import rmdir

if __name__ == '__main__':
    os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    os.system('unzip tiny-imagenet-200.zip')
    target_folder = './tiny-imagenet-200/val/'

    val_dict = {}
    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = os.listdir('./tiny-imagenet-200/val/images')
    for path in paths:
        path = './tiny-imagenet-200/val/images/' + path
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')

    for path in paths:
        path = './tiny-imagenet-200/val/images/' + path
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = target_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    rmdir('./tiny-imagenet-200/val/images')
