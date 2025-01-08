import glob
import os

import cv2
import numpy as np
from PIL import Image


def png_to_npz(input_dir, output_dir, image_dir='imagesTr', label_dir='labelsTr'):
    """
    将 PNG 图像和标签图像转换为 NPZ 格式。

    Args:
        input_dir (str): 原始数据集的根目录。
        output_dir (str): 转换后 .npz 文件的保存目录。
        image_dir (str): 输入图像所在的子目录名称。
        label_dir (str): 标签图像所在的子目录名称。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 图像和标签文件路径
    image_paths = sorted(os.listdir(os.path.join(input_dir, image_dir)))
    label_paths = sorted(os.listdir(os.path.join(input_dir, label_dir)))
    # print(image_paths)
    # print(label_paths)

    if len(image_paths) != len(label_paths):
        raise ValueError("The number of images and labels must be the same.")

    for image_name, label_name in zip(image_paths, label_paths):
        # 确保文件名匹配
        if os.path.splitext(image_name)[0] != os.path.splitext(label_name)[0]:
            raise ValueError(f"Image {image_name} and label {label_name} do not match.")
        if image_name.endswith('.jpg'):
        # 读取图像
            image_path = os.path.join(input_dir, image_dir, image_name)
            label_path = os.path.join(input_dir, label_dir, label_name)

            image = np.array(Image.open(image_path).convert('L'))  # 转换为 RGB 格式
            label = cv2.imread(label_path,flags=0)    # 转换为单通道灰度图
            # label[label!=255]=0
            # label[label==255] = 1
            # 保存为 .npz 格式
            output_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.npz")
            np.savez_compressed(output_file, image=image, label=label)
        # print(f"Saved: {output_file}")
    print("Done")

# 使用示例
input_dir = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid" # 数据集根目录，包含 'images' 和 'labels' 子目录
output_dir1 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/imagesTr"  # .npz 保存目录
output_dir2 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/test_vol_h5"  # .npz 保存目录
tx1 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/lists/Thyroid/train.txt"
tx2 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/lists/Thyroid/test_vol.txt"
# png_to_npz(input_dir, output_dir)

def write_name(output_dir,tx):
    #npz文件路径
    files = os.listdir(output_dir)
    #txt文件路径
    f = open(tx,'w')
    for i in files:
        name = i[:-4]+'\n'
        f.write(name)
    print("Done")
# convert name in txt

def conver_label(labels_folder,output_folder):
    for filename in os.listdir(labels_folder):
        if filename.endswith(".png"):  # 只处理 PNG 文件
            label_path = os.path.join(labels_folder, filename)
            label_image = Image.open(label_path)
            label_array = np.array(label_image)

            # 修正标签值，将所有非 0 和 1 的标签映射为 0
            label_array[label_array > 1] = 0

            # 保存修正后的标签图像
            corrected_label_image = Image.fromarray(label_array)
            corrected_label_image.save(os.path.join(output_folder, filename))

    print("Labels corrected successfully!")
if __name__ == '__main__':
    label_image = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/train_npz/L1-0001-1.npz"
    image_data = np.load(label_image)
    if 'image' in image_data:
        images = image_data['image']
        print(images.shape)

    if 'label' in image_data:
        labels = image_data['label']
        print(labels.shape)
    # label = cv2.imread(label_image,flags=0)
    # label_array = np.array(label)
    # print(label_array.shape) # 获取标签图像中的唯一值
    # unique_labels = np.unique(label_array)
    # print(unique_labels)
    # unique_labels[unique_labels!=255]=0
    # unique_labels[unique_labels==255]=1
    # print(unique_labels)
    write_name(output_dir1,tx1)
   #  png_to_npz(input_dir, output_dir1, image_dir='imagesTr', label_dir='labelsTr')
   #  png_to_npz(input_dir, output_dir2, image_dir='imagesTs', label_dir='labelsTs')
    #
    labels_folder1 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/labelsTr"
    labels_folder2 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/labelsTs"
    output_folder1 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/labelsTr"
    output_folder2 = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/labelsTs"

    # conver_label(labels_folder2,output_folder2)
    # 遍历所有标签图像




