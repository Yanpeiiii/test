import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms


class Normalize(object):

    def __init__(self, op_mean=(0., 0., 0.), op_std=(1., 1., 1.), sar_mean=0., sar_std=0.):
        self.op_mean = op_mean
        self.op_std = op_std
        self.sar_mean = sar_mean
        self.sar_std = sar_std

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)
        if img1.shape[-1] == 3:
            img1 /= 255.0
            img1 -= self.op_mean
            img1 /= self.op_std
        else:
            img1 /= 255.0
            img1 -= self.sar_mean
            img1 /= self.sar_std

        if img2.shape[-1] == 3:
            img2 /= 255.0
            img2 -= self.op_mean
            img2 /= self.op_std
        else:
            img2 /= 255.0
            img2 -= self.sar_mean
            img2 /= self.sar_std

        return {'image': (img1, img2)}


class ToTensor(object):

    def __call__(self, sample):
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample['image'][0]
        img2 = sample['image'][1]

        img1 = np.array(img1)
        img2 = np.array(img2)
        if img1.shape[-1] > 3:
            img1 = np.expand_dims(img1, axis=-1)
            img1 = img1.astype(np.float32).transpose((2, 0, 1))
        else:
            img1 = img1.astype(np.float32).transpose((2, 0, 1))
        if img2.shape[-1] > 3:
            img2 = np.expand_dims(img2, axis=-1)
            img2 = img2.astype(np.float32).transpose((2, 0, 1))
        else:
            img2 = img2.astype(np.float32).transpose((2, 0, 1))

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()

        return {'image': (img1, img2)}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2)}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        # mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2)}


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)

        return {'image': (img1, img2)}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]

        if random.random() < 0.5:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img1 = img1.rotate(rotate_degree, Image.BILINEAR, expand=True)
            img2 = img2.rotate(rotate_degree, Image.BILINEAR, expand=True)

        return {'image': (img1, img2)}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]

        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': (img1, img2)}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        # assert img1.size == img2.size
        # assert img1.size == mask.size and img2.size == mask.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)

        return {'image': (img1, img2)}


train_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomFixRotate(),
            RandomRotate(30),
            FixedResize(256),
            RandomGaussianBlur(),
            Normalize(op_mean=(0.5, 0.5, 0.5), op_std=(0.5, 0.5, 0.5), sar_mean=(0.5,), sar_std=(0.5,)),
            ToTensor()])

test_transforms = transforms.Compose([
            FixedResize(256),
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomFixRotate(),
            # RandomGaussianBlur(),
            Normalize(op_mean=(0.5, 0.5, 0.5), op_std=(0.5, 0.5, 0.5), sar_mean=(0.5,), sar_std=(0.5,)),
            ToTensor()])
