import matplotlib.pyplot as plt
import numpy as np
import cv2
import os



def load_images(input_dir, gt_dir, pred_dirs):
    """
    加载原始图像、Ground Truth 和多个分割结果。
    """
    input_images = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')])
    gt_images = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
    pred_images = {method: sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith('.png')])
                   for method, pred_dir in pred_dirs.items()}
    return input_images, gt_images, pred_images


def plot_segmentation_results(input_images, gt_images, pred_images, methods, save_path=None):
    """
    绘制分割结果。
    """
    num_samples = len(input_images)               #5张图像
    num_methods = len(methods)                #5种方法

    fig, axes = plt.subplots(num_samples, num_methods + 2, figsize=(num_methods * 2, num_samples*2-3))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for i in range(num_samples):
        # 原始输入图像
        input_img = cv2.imread(input_images[i])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (224, 224))
        axes[i, 0].imshow(input_img)
        if i==0:
            axes[i, 0].set_title("Input",fontsize=10)
        axes[i, 0].axis('off')


        # Ground Truth
        gt_img = cv2.imread(gt_images[i], cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.resize(gt_img, (256,256))
        axes[i, 1].imshow(gt_img, cmap='gray')
        if i == 0:
            axes[i, 1].set_title("GT",fontsize=10)
        axes[i, 1].axis('off')

        # 分割结果
        for j, method in enumerate(methods):
            pred_img = cv2.imread(pred_images[method][i], cv2.IMREAD_GRAYSCALE)
            pred_img = cv2.resize(pred_img, (256,256))
            axes[i, j + 2].imshow(pred_img, cmap='gray')
            if i == 0:
                axes[i, j + 2].set_title(method,fontsize=10)
            axes[i, j + 2].axis('off')

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_imgs_from_path(img_path, msk_path, msk_pred_path, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # Load the image
    img = Image.open(img_path)
    img = np.array(img) / 255.0

    # Load the mask and prediction mask
    msk = Image.open(msk_path)
    msk = np.array(msk)

    msk_pred = Image.open(msk_pred_path)
    msk_pred = np.array(msk_pred)

    # Process masks based on dataset type
    if datasets == 'retinal':
        pass  # Retinal data assumed to be correctly loaded as 2D masks
    else:
        msk = np.where(msk > 0.5, 1, 0)
        msk_pred = np.where(msk_pred > threshold, 1, 0)

    # Plot and save the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Plot the original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the ground truth mask
    axes[1].imshow(msk, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    # Plot the predicted mask
    axes[2].imshow(msk_pred, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage
# save_imgs_from_path('path/to/img.png', 'path/to/msk.png', 'path/to/msk_pred.png', 0, 'output_directory', 'retinal')


# 示例使用
input_dir = './output/SwinUT_mask_pred/images'
gt_dir = './output/SwinUT_mask_pred/masks'
pred_dirs = {
    'nnUNet': './output/nnUNet',
    'SwinUNETR': './output/SwinUNETR',
    'UMamba': './output/UMamba',
    'VMUNetv2': './output/VMUNetv2',
    'SwinUMamba': './output/SwinUM_mask_pred',
    'SwinUMamba+': './output/SwinUMambaD'

}
def show():
    input_images = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/test/images/L4-0013-5.jpg'
    gt_images = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/test/masks/L4-0013-5.png'
    msk_pred='./output/VMUNetv2/ph1/L4-0013-5.png'
    # input_images = cv2.imread(input_images, cv2.COLOR_BGRA2BGR)
    # gt_images = cv2.imread(gt_images,cv2.IMREAD_GRAYSCALE)
    # msk_pred=cv2.imread(msk_pred,cv2.IMREAD_GRAYSCALE)
    save_imgs_from_path(img_path=input_images, msk_path=gt_images,msk_pred_path=msk_pred,i=0, datasets="None", threshold=0.5,
              test_data_name='test', save_path=None)
# 加载图像路径
input_images, gt_images, pred_images = load_images(input_dir, gt_dir, pred_dirs)
# 绘制和保存结果
methods = ['nnUNet',  'SwinUNETR', 'UMamba', 'VMUNetv2','SwinUMamba','SwinUMamba+']
if __name__ == '__main__':
    plot_segmentation_results(input_images, gt_images, pred_images, methods, save_path=None)
    # mask = cv2.imread('./output/SwinUT_mask_pred/visual/L4-0021-7.png', cv2.IMREAD_GRAYSCALE)
    # print(mask.shape)
    # mask_label = np.array(mask)
    # mask_unique = np.unique(mask_label)
    # print(mask_unique)
    # show()
