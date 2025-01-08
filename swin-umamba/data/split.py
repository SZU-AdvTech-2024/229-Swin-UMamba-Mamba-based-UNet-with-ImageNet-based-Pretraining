import os
import shutil
import random

def split_dataset(images_dir, masks_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将 images 和 masks 数据集按照指定比例划分为 train, val, test 数据集
    并保持 images 和 masks 文件夹的对应关系。

    Args:
        images_dir (str): 原始 images 文件夹路径。
        masks_dir (str): 原始 masks 文件夹路径。
        output_dir (str): 输出数据集的根目录路径。
        train_ratio (float): 训练集比例，默认 0.8。
        val_ratio (float): 验证集比例，默认 0.1。
        test_ratio (float): 测试集比例，默认 0.1。
    """
    # 检查路径是否存在
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory '{images_dir}' not found.")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory '{masks_dir}' not found.")

    # 创建输出目录
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_masks_dir = os.path.join(output_dir, "train", "masks")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_masks_dir = os.path.join(output_dir, "val", "masks")
    test_images_dir = os.path.join(output_dir, "test", "images")
    test_masks_dir = os.path.join(output_dir, "test", "masks")

    for path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, test_images_dir, test_masks_dir]:
        os.makedirs(path, exist_ok=True)

    # 获取 images 和 masks 的文件列表
    image_files = sorted(os.listdir(images_dir))  # 确保排序一致
    mask_files = sorted(os.listdir(masks_dir))

    # 检查 images 和 masks 文件是否一一对应
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks does not match.")
    for img_file, mask_file in zip(image_files, mask_files):
        if os.path.splitext(img_file)[0] != os.path.splitext(mask_file)[0]:
            raise ValueError(f"Image and mask file names do not match: {img_file} and {mask_file}.")

    # 打乱文件列表
    data = list(zip(image_files, mask_files))
    random.shuffle(data)

    # 计算划分数量
    total = len(data)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_data = data[:train_count]
    val_data = data[train_count:train_count + val_count]
    test_data = data[train_count + val_count:]

    # 拷贝文件到对应目录
    def copy_files(data, images_output_dir, masks_output_dir):
        for img_file, mask_file in data:
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(images_output_dir, img_file))
            shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(masks_output_dir, mask_file))

    copy_files(train_data, train_images_dir, train_masks_dir)
    copy_files(val_data, val_images_dir, val_masks_dir)
    copy_files(test_data, test_images_dir, test_masks_dir)

    print(f"Dataset split completed. Total: {total}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

#使用示例
images_path = "./data/DDTI/images"  # 替换为 images 文件夹路径
masks_path = "./data/DDTI/masks"  # 替换为 masks 文件夹路径
output_path = "./data/DDTI"

#CVC-300 Train: 48, Val: 6, Test: 6
#ClinicDB Train: 489, Val: 61, Test: 62
#ColonDB Train: 304, Val: 38, Test: 38
#Kvasir-Ins Train: 472, Val: 59, Test: 59
#Kvasir-seg Train: 800, Val: 100, Test: 100

# images_path = "./data/CVC-ClinicDB/images"  # 替换为 images 文件夹路径
# masks_path = "./data/CVC-ClinicDB/masks"  # 替换为 masks 文件夹路径
# output_path = "./data/CVC-ClinicDB"  # 替换为输出目录路径

def rename(data_path, out_path):
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            newname = "cvc_colon_"+filename
            print(newname)
            old_dir = os.path.join(data_path,filename)
            new_dir = os.path.join(out_path,newname)
            os.rename(old_dir,new_dir)
    print("操作完成")


if __name__ == '__main__':
    # rename(data_path="./data/CVC-ColonDB/masks", out_path="./data/Polpy/test/masks")
    split_dataset(images_path, masks_path, output_path)

