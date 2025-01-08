import shutil

import numpy as np
import os
import random

# data=[0,1,2,3,4,5,6,7,8,9,10]
# random.shuffle(data)
#
# n=len(data)
# train_data=data[:int(n*0.8)]
# test_data=data[int(n*0.8):int(n*0.9)]
# val_data=data[int(n*0.9):]
#
# print(train_data)
# print(test_data)
# print(val_data)
def data_split(data_dir,label_dir,source_dir,train_nums=0.8,test_nums=0.2):
    all_images =[f for f in os.listdir(data_dir)if os.path.isfile(os.path.join(data_dir,f))]
    random.shuffle(all_images)

    total=len(all_images)
    print("all images nums:",total)
    train_set=all_images[:int(train_nums*total)]
    test_set=all_images[int(train_nums*total):]

    train_dir=os.path.join(source_dir,"imagesTr")
    train_label_dir=os.path.join(source_dir,"labelsTr")
    test_dir=os.path.join(source_dir,"imagesTs")
    test_label_dir=os.path.join(source_dir,"labelsTs")


    for image in train_set:
        shutil.move(os.path.join(data_dir,image),train_dir)
        shutil.move(os.path.join(label_dir,image.split('.jpg')[0]+".png"),train_label_dir)
    for image in test_set:
        shutil.move(os.path.join(data_dir,image),test_dir)
        shutil.move(os.path.join(label_dir,image.split('.jpg')[0]+".png"),test_label_dir)
    print(f"Data split completed: {len(train_set)} train, {len(test_set)} test.")
    print("Done")

# data_dir="/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Task705_Thyroid/image"
# label_dir="/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Task705_Thyroid/label"
# source_dir="/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Task705_Thyroid"
# data_split(data_dir,label_dir,source_dir)

from pyproj import Proj, transform

# 定义UTM投影和WGS84投影


if __name__ == '__main__':
    # utm_proj = Proj(proj='utm', zone=50, datum='WGS84')  # 50N区
    # wgs84_proj = Proj(proj='latlong', datum='WGS84')
    #
    # # 提供的UTM坐标
    # utm_x = 168.587
    # utm_y = 2422.266
    #
    # # 将UTM坐标转换为经纬度
    # lon, lat = transform(utm_proj, wgs84_proj, utm_x, utm_y)
    # print(f"纬度: {lat}, 经度: {lon}")
    # 定义UTM投影和WGS84投影
    utm_proj = Proj(proj='utm', zone=50, datum='WGS84')  # 50N区
    wgs84_proj = Proj(proj='latlong', datum='WGS84')

    # 提供的UTM坐标
    utm_x = 168.587
    utm_y = 2422.266

    # 将UTM坐标转换为经纬度
    lon, lat = transform(utm_proj, wgs84_proj, utm_x, utm_y)
    print(f"纬度: {lat}, 经度: {lon}")