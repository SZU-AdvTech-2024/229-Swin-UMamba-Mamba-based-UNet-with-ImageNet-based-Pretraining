# import os
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Step 1: 加载真实标签和模型预测结果（假设图像为二值化图像）
# gt = cv2.imread("/home/tjc/pycharmprojects/data/busi/train_val_test2/test/msk_256/benign (124).png", cv2.IMREAD_GRAYSCALE)  # 真实标签
# prediction = cv2.imread("/home/tjc/pycharmprojects/aau-net/mask_pred/busi/cmunext2/benign (124).png", cv2.IMREAD_GRAYSCALE)  # 模型预测结果
#
# # 确保图像是二值化的，值为0或1
# gt = (gt > 128).astype(np.uint8)
# prediction = (prediction > 128).astype(np.uint8)
#
# # Step 2: 计算真阳性、假阳性和假阴性
# true_positive = np.logical_and(prediction == 1, gt == 1)  # 真阳性
# false_positive = np.logical_and(prediction == 1, gt == 0)  # 假阳性
# false_negative = np.logical_and(prediction == 0, gt == 1)  # 假阴性
#
# # Step 3: 创建一个彩色图像，用于显示结果
# result = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
#
# # Step 4: 设置颜色编码
# result[true_positive] = [255, 255, 0]    # 真阳性显示为黄色
# result[false_positive] = [255, 0, 0]     # 假阳性显示为红色
# result[false_negative] = [0, 255, 0]     # 假阴性显示为绿色
#
#
# output_folder = './output_results/busi1/'  # 指定保存文件夹
# output_filename = '2.png'  # 指定保存文件名
#
# # 如果文件夹不存在，创建文件夹
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 生成完整的保存路径
# output_path = os.path.join(output_folder, output_filename)
#
#
# cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # 保存为PNG文件
import cv2
import numpy as np
import os


# 定义处理单个图像的函数
def process_image(gt_path, pred_path, output_path):
    # Step 1: 加载真实标签和模型预测结果
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # 真实标签
    prediction = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 模型预测结果

    # 确保图像是二值化的，值为0或1
    gt = (gt > 128).astype(np.uint8)
    prediction = (prediction > 128).astype(np.uint8)

    # Step 2: 计算真阳性、假阳性和假阴性
    true_positive = np.logical_and(prediction == 1, gt == 1)  # 真阳性
    false_positive = np.logical_and(prediction == 1, gt == 0)  # 假阳性
    false_negative = np.logical_and(prediction == 0, gt == 1)  # 假阴性

    # Step 3: 创建一个彩色图像，用于显示结果
    result = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

    # Step 4: 设置颜色编码
    result[true_positive] = [255, 255, 0]    # 真阳性显示为黄色
    result[false_positive] = [255, 0, 0]     # 假阳性显示为红色
    result[false_negative] = [0, 255, 0]     # 假阴性显示为绿色

    # Step 5: 保存生成的对比图像
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # 保存为PNG文件

# 定义批处理的函数
def batch_process_folder(gt_folder, pred_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历真实标签文件夹中的每个文件
    for filename in os.listdir(gt_folder):
        # 构造对应的真实标签、预测结果和输出路径
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 检查预测结果是否存在
        if os.path.exists(pred_path):
            # 处理每一对图像
            process_image(gt_path, pred_path, output_path)
            print(f'Processed {filename}')
        else:
            print(f'Prediction file for {filename} not found!')

# 设置文件夹路径
gt_folder = "/home/tjc/pycharmprojects/data/ph2/train_val_test3/test/msk_256/"  # 真实标签文件夹
pred_folder = "/home/tjc/pycharmprojects/mamba/mask_pred/ph2/swin-umamba3/"  # 模型预测文件夹
output_folder = "/home/tjc/pycharmprojects/mamba/utils/output_results/ph2_3/swin-umamba3"  # 保存输出的文件夹

# 批处理文件夹中的所有图像
batch_process_folder(gt_folder, pred_folder, output_folder)
