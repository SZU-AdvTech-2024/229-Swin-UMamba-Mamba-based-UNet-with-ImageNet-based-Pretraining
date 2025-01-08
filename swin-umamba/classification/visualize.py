import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from timm.models.resnet import resnet50
from torchvision.transforms import transforms
from my_dataset import MyDataSet
from model1 import MedMamba
data_transform = {
    "val": transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = "./data/mini-imagenet"
json_path = "./classes_name.json"
    # 实例化训练数据集

    # 实例化验证数据集
val_dataset = MyDataSet(root_dir=data_root,
                            csv_name="new_val.csv",
                            json_path=json_path,
                            transform=data_transform["val"])
val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=18,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)




net = MedMamba().cuda()
    # net = resnet50(num_class)
    # 加载模型参数
net.load_state_dict(torch.load("./weights/MedModel.pth"),False)
net.eval()
    # 加载验证集的数据进行验证
    # data = next(dataset_val.create_dict_iterator())
data_iter = iter(val_loader)
images, labels = next(data_iter)
img = images
images ,labels = images.cuda(), labels.cuda()
output = net(images)
predictions = torch.max(output,1)[1]

    # 图像分类
with open("./classes_name.json", "r") as f:  # 替换为实际路径
    raw_classes = json.load(f)  # 如果 JSON 是一个列表，直接返回
classes = [None] * len(raw_classes)  # 创建一个空列表，大小等于类别总数
for key, value in raw_classes.items():
    index = value[0]  # 获取类别索引
    name = value[1]  # 获取类别名称
    classes[index] = name  # 按索引填充类别名称
print(classes[:10])

def visualize_model(images,labels,predictions,classes):
    images = images.cpu()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    if len(images.shape) == 4:  # (N, C, H, W)
        batch_size = images.shape[0]
    elif len(images.shape) == 5:  # (1, N, C, H, W) -> 需要 squeeze
        images = images.squeeze(0)
        batch_size = images.shape[0]
    else:
        raise ValueError("Unsupported images tensor shape. Expected 4D or 5D tensor.")
    # 显示图像及图像的预测值

    plt.figure(figsize=(18, 9))
    for i in range(min(batch_size,18)):
        plt.subplot(3, 6, i + 1)
        image = images[i].permute(1, 2, 0).numpy()  # 转换为 HWC 格式
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反归一化
        image = np.clip(image, 0, 1)  # 保证像素值在 [0, 1] 范围内
        plt.imshow(image)
        pred_label = classes[predictions[i]]
        true_label = classes[labels[i]]
        color = "blue" if pred_label == true_label else "red"
        plt.title(f"predict: {pred_label}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_model(images,labels,predictions,classes)
# 使用测试数据集进行验证

