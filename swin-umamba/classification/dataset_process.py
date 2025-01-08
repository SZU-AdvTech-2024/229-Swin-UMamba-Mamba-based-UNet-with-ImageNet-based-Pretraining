import csv
import os
from PIL import Image
import json


train_csv_path = "./data/mini-imagenet/new_train.csv"
val_csv_path = "./data/mini-imagenet/new_val.csv"

json_path = "./classes_name.json"
label_dict = json.load(open(json_path, "r"))

train_label = {}
val_label = {}

with open(train_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        train_label[row[1]] = label_dict[row[2]][1]

with open(val_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        val_label[row[1]] = label_dict[row[2]][1]

img_path = "./data/mini-imagenet/images"
new_img_path = "./data/mini-imagenet"
for png in os.listdir(img_path):
    path = img_path + '/' + png
    im = Image.open(path)
    if (png in train_label.keys()):
        tmp = train_label[png]
        temp_path = new_img_path + '/train' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)

    elif (png in val_label.keys()):
        tmp = val_label[png]
        temp_path = new_img_path + '/val' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)
print("Done")
