from PIL import Image
import os

# 设置图片存放目录和目标目录
# source_dir = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset704_Endovis17/imagesTr'  # TIFF图片存放目录
# target_dir = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset704_Endovis17/imagesTr'    # PNG图片存放目录
# #
# # for filename in os.listdir(source_dir):
# #     if len(filename)==12 and filename.endswith('.png'):
# #         newname="17_2_"+filename[0:8]+"_0000.png"
# #         old_dir=os.path.join(target_dir, filename)
# #         new_dir=os.path.join(target_dir, newname)
# #         os.rename(old_dir, new_dir)
# # print('Done')
#
# # 确保目标目录存在
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)
#
# # 获取源目录下所有TIFF文件的路径
# tiff_images = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if file.endswith('.tiff')]
#
# # 遍历文件并转换为PNG
# for tiff_image in tiff_images:
#     img = Image.open(tiff_image)
#     img = img.convert('RGB')  # 转换色彩模式为RGB，防止图片颜色失真
#     png_name = os.path.splitext(os.path.basename(tiff_image))[0] + '.png'
#     png_path = os.path.join(target_dir, png_name)
#     img.save(png_path)
# print("Done")

# JPG->PNG格式
# from PIL import Image
# import os
#
# # 输入文件夹路径，包含所有需要转换的 JPG 文件
# input_folder = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Task705_Thyroid/imagesTs'
#
# # 输出文件夹路径，保存转换后的 PNG 文件
# output_folder = "/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Task705_Thyroid/imagesTs"
#
# # 如果输出文件夹不存在，创建它
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 遍历输入文件夹中的所有 JPG 文件
# for filename in os.listdir(input_folder):
#     if filename.endswith(".jpg") or filename.endswith(".jpeg"):
#         # 构建完整的文件路径
#         jpg_path = os.path.join(input_folder, filename)
#
#         # 打开 JPG 文件
#         with Image.open(jpg_path) as img:
#             # 构建 PNG 文件的保存路径
#             png_filename = os.path.splitext(filename)[0] + '.png'
#             png_path = os.path.join(output_folder, png_filename)
#
#             # 保存为 PNG 格式
#             img.save(png_path, 'PNG')
#             # print(f"Converted {filename} to {png_filename}")
#
# print("Batch conversion completed!")


from PIL import Image
import numpy as np

# 读取标签图像
source = "/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Task704_Endovis17/labelsTs"
out_dir = "/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset704_Endovis17/labelsTs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for filename in os.listdir(source):
    if filename.endswith('.png'):
        # 构建完整的文件路径
        input_path = os.path.join(source, filename)
        output_path = os.path.join(out_dir, filename)

        # 打开图片
        img = Image.open(input_path)

        # 转换为单通道（灰度）图像
        gray_img = img.convert('L')

        # 保存为新的PNG图片
        gray_img.save(output_path)

        print(f"Converted {filename} to grayscale and saved to {output_path}")


# label_array = np.array(label_image)
# print(label_array.shape)
# # # 获取标签图像中的唯一值
# unique_labels = np.unique(label_array)
# print("Unique labels in the image:", unique_labels)

# label_array[label_array > 1] = 0
#
# # 保存修正后的标签图像
# corrected_label_image = Image.fromarray(label_array)
# corrected_label_image.save("/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/labelsTr/L1-0001-1.png")
#
# import os
# from PIL import Image
# import numpy as np
#
labels_folder = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/labelsTr"
output_folder = "/home/cwq/MedicalDP/SwinUmamba/SwinUnet/evaluate/Thyroid/labelsTr"

# 遍历所有标签图像
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
