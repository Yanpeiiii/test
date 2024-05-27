import os
import torch.utils.data as data
from PIL import Image
import data.transforms as tr


def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not i.startswith('.')]
    valid_data.sort()

    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp]}

    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],}

    return train_dataset, val_dataset


def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp]}

    return test_dataset


def loader(img_path, aug):
    dir = img_path[0]
    name = img_path[1]

    index, _ = name.split('.')

    img1 = Image.open(dir + 'A/' + name)
    img2 = Image.open(dir + 'B/' + name)
    sample = {'image': (img1, img2)}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return index, sample['image'][0], sample['image'][1]


class Loader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = loader
        self.aug = aug

    def __getitem__(self, index):

        img_path = self.full_load[index]['image']

        return self.loader(img_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
