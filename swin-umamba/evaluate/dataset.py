import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),  # 随机水平翻转
    A.VerticalFlip(p=0.5),    # 随机垂直翻转
    A.RandomRotate90(p=0.5),  # 随机旋转 90 度
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # 平移、缩放和旋转
    A.RandomBrightnessContrast(p=0.5),  # 随机亮度对比度调整
    A.GaussianBlur(p=0.2),              # 高斯模糊
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ToTensorV2()  # 转换为 PyTorch 张量
])
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

# transformer 等比例缩放图片
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

# 在transform中使用自定义的resize转换器
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Dataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), transform=None, augmentation=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.augmentation = augmentation
        self.names = [i for i in os.listdir(os.path.join(self.root_dir, 'images'))]
        # 获取图像和标签的文件路径
        self.image_list = [os.path.join(root_dir, 'images', fname) for fname in self.names]
        self.mask_list = [os.path.join(root_dir, 'masks', fname.replace('.jpg', '.png')) for fname in self.names]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 读取图像和标签
        image = Image.open(self.image_list[idx]).convert('RGB')
        mask = Image.open(self.mask_list[idx]).convert('L')  # 灰度图

        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size)
        # 转换为 numpy 数组以进行增强
        image = np.array(image)
        # mask = np.array(mask)

        # 数据增强
        if self.augmentation:
            augmented = self.augmentation(image=image)
            # image = augmented['image'], augmented['mask']
            image = augmented['image']

        # 转换为 PIL 图像
        # image = Image.fromarray(image)
        # mask = Image.fromarray(mask)

        # 转换为张量并标准化
        # if self.transform:
        #     image = self.transform(image)
        # else:
        #     transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 适配预训练模型
        #     ])
        #     image = transform(image)

        # 处理 mask: 二值化处理
        mask = torch.tensor(np.array(mask) > 127, dtype=torch.float32).unsqueeze(0)  # Binarize and add channel dimension

        return image, mask

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

class ThyroidDataset(Dataset):
    def __init__(self, root_dir, transform=None,image_size=(224,224)):
        self.image_size = image_size
        self.root_dir = root_dir
        self.transforms = transform

        self.names = [i for i in os.listdir(os.path.join(self.root_dir, 'images'))]
        #self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.mask_list = [os.path.join(self.root_dir, 'masks', i.replace('.jpg','.png')) for i in self.names]

        # check files
        for i in self.image_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        # image = transform(image)
        mask = Image.open(self.mask_list[idx]).convert('L')
        # mask = transform(mask)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask / 255.0

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)          #图像填充，填充值为0
        batched_targets = cat_list(targets, fill_value=255)     #mask填充值为255
        return batched_imgs, batched_targets

class BinaryDataset(Dataset):# Transform for input images
    def __init__(self, root_dir, image_size=(224, 224),transform=None):
        self.image_size = image_size
        self.root_dir = root_dir
        self.transforms = transform

        self.names = [i for i in os.listdir(os.path.join(self.root_dir, 'images'))]
        self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.mask_list = [os.path.join(self.root_dir, 'masks', i.replace('.jpg', '.png')) for i in self.names]

        # Check if files exist
        for i in self.image_list:
            if not os.path.exists(i):
                raise FileNotFoundError(f"File {i} does not exist.")
        for i in self.mask_list:
            if not os.path.exists(i):
                raise FileNotFoundError(f"File {i} does not exist.")

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_list[idx]).convert('RGB')
        mask = Image.open(self.mask_list[idx]).convert('L')
        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size)

        if self.transforms:
            image = self.transforms(image)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 适配预训练模型
            ])
            image = transform(image)
        # Load mask
        mask = torch.tensor(np.array(mask) > 127, dtype=torch.float32).unsqueeze(0)  # Binarize and add channel dimension

        return image, mask

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0)
        batched_targets = torch.stack(targets, dim=0)
        return batched_imgs, batched_targets

