import random
from PIL import Image
import numpy as np
import torch
from sympy.physics.units import degree
from sympy.printing.tests.test_repr import test_set
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, size=256):   #modify
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, [self.size, self.size])
        target = F.resize(target, [self.size, self.size], interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, image, mask):
        if random.random() < self.p: return F.rotate(image,self.angle), F.rotate(mask,self.angle)
        else: return image, mask

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ResizeKeepAspectRatio(object):
    def __init__(self, target_size=224):
        # target_size是最大边的大小，传入整数
        self.target_size = target_size

    def __call__(self, image, target):
        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        return image.resize((new_width, new_height), Image.LANCZOS),target.resize((new_width, new_height),Image.LANCZOS)


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
        elif data_name == 'kvasir':
            if train:
                self.mean = 89.4836
                self.std = 66.4157
            else:
                self.mean = 91.1939
                self.std = 64.4282
        elif data_name == 'kvasir_seg':
            if train:
                self.mean = 99.2775
                self.std = 60.8323
            else:
                self.mean = 97.6336
                self.std = 62.4160
        elif data_name == 'polyp':
            if train:
                self.mean = 86.17
                self.std = 69.08
            else:
                self.mean = 86.17
                self.std = 69.08
        elif data_name == 'ulcer':
            if train:
                self.mean = 120.00
                self.std = 58.56
            else:
                self.mean = 120.22
                self.std = 58.72
        elif data_name == 'lsil':
            if train:
                self.mean = 142.77
                self.std = 37.92
            else:
                self.mean = 142.81
                self.std = 38.25

    def __call__(self, img, mask):
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - img_normalized.min())
                          / (img_normalized.max()- img_normalized.min())) * 255.
        return img_normalized, mask

# Thyroid argumentation
class train_transforms(object):
    def __init__(self):
        trans = [
            Resize(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize()
        ]

        self.transforms = Compose(trans)

    def __call__(self, image, target):
        return self.transforms(image, target)

class val_transforms(object):
    def __init__(self):
        self.transforms = Compose([Resize(), ToTensor(), Normalize()])

    def __call__(self, image, target):
        return self.transforms(image, target)

def get_transform(train=True):
    if train:
        return train_transforms()
    else:
        return val_transforms()


# Kvasir argumentation

class myToTensor:
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return F.resize(image, [self.size_h, self.size_w]), F.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return F.vflip(image), F.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return F.rotate(image, self.angle), F.rotate(mask, self.angle)
        else:
            return image, mask

# train_transformer = T.Compose([
#     myNormalize(data_name='Kvasir-seg', train=True), #
#     myToTensor(),
#     myRandomHorizontalFlip(p=0.5),
#     myRandomVerticalFlip(p=0.5),
#     myRandomRotation(p=0.5, degree=[0, 360]),
#     myResize(256, 256)
#     ])
# test_transformer = T.Compose([
#     myNormalize(data_name='Kvasir-seg', train=False),
#     myToTensor(),
#     myResize(256, 256)
# ])

class KvaNormalize:
    def __init__(self, data_name='kvasir_seg', train=True):
        if data_name == 'kvasir':
            if train:
                self.mean = 89.4836
                self.std = 66.4157
            else:
                self.mean = 91.1939
                self.std = 64.4282
        elif data_name == 'kvasir_seg':
            if train:
                self.mean = 99.2775
                self.std = 60.8323
            else:
                self.mean = 97.6336
                self.std = 62.4160
        elif data_name == 'polyp':
            if train:
                self.mean = 86.17
                self.std = 69.08
            else:
                self.mean = 86.17
                self.std = 69.08

    def __call__(self, image, target):
        img_normalized = (image - self.mean) / self.std
        img_normalized = ((img_normalized - img_normalized.min())
                          / (img_normalized.max()- img_normalized.min())) * 255.
        return img_normalized, target

class train_transformer(object):
    def __init__(self):
        trans = [
            Resize(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(p=0.5, degree=[0, 360]),
            ToTensor(),
            KvaNormalize()
        ]

        self.transforms = Compose(trans)

    def __call__(self, image, target):
        return self.transforms(image, target)

class test_transformer(object):
    def __init__(self):
        self.transforms = Compose([Resize(), ToTensor(), KvaNormalize(train=False)])

    def __call__(self, image, target):
        return self.transforms(image, target)

def load_transform(train=True):
    if train:
        return train_transformer()
    else:
        return test_transformer()

