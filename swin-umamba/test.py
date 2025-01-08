import os

import numpy as np
import pandas as pd

import cv2
import torch
from matplotlib import pyplot as plt
from pydantic.v1.utils import get_model
from tqdm import tqdm
from utils.config import  parse_args
from nets import get_dataset,net
from nets.vision_transformer import SwinUnet, Swin_model
from dataset import Dataset, ThyroidDataset, PolypDataset
from utils.transforms import get_transform,load_transform
from utils.metrics import iou_score
from utils.utils import AverageMeter
from utils.SWconfig import get_config,swin_config
def main():
    print("=>SwinUNet creating model")
    config = get_config(swin_config())
    model = SwinUnet(config).cuda()
    # 需要改
    model_path1= "/home/cwq/MedicalDP/SwinUmamba/swin-umamba/model_out/SwinUNet/checkpoint_best"
    model.load_state_dict(torch.load(model_path1),False)
    model.eval()

    # 需要改
    val_dataset = ThyroidDataset("/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/test",
                        get_transform(train=False))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24, shuffle=False,
                                             collate_fn=val_dataset.collate_fn, drop_last=False)
    print("The val_dataset numbers:", len(val_dataset))
    val_names = val_dataset.names
    count = 0

    # 需要改
    mask_pred = "/home/cwq/MedicalDP/SwinUmamba/swin-umamba/output/SwinUT_mask_pred"
    os.makedirs(os.path.join(mask_pred, 'ph1'), exist_ok=True)
    avg_meters = {'test_iou': AverageMeter(), 'test_dice': AverageMeter(),
                  'test_acc': AverageMeter(), 'test_pc': AverageMeter(), 'test_se': AverageMeter(),
                  'test_sp': AverageMeter()}

    with torch.no_grad():
        for input, target in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            print("Target range: min =", target.long().min().item(), "max =", target.long().max().item())
            output = model(input)
            print("Raw model output range: min =", output.min().item(), "max =", output.max().item(),"output shape:", output.size())
            mask = output.clone()
            mask = torch.sigmoid(mask).cpu().numpy() > 0.5
            print("Binary mask unique values:", np.unique(mask))
            sigmoid_output = torch.sigmoid(output).cpu().numpy()
            print("Sigmoid output range: min =", sigmoid_output.min(), "max =", sigmoid_output.max(), np.unique(sigmoid_output))

            # plt.imshow(input[0].cpu().numpy().transpose(1, 2, 0))
            # plt.title("Input Image")
            # plt.show()
            #
            # plt.imshow(target[0].cpu().numpy().squeeze(), cmap='gray')
            # plt.title("Target Label")
            # plt.show()
            # 需要改
            for i in range(len(mask)):
                single_mask =(mask[i,0]*255).astype('uint8')
                # plt.imshow(single_mask, cmap='gray')
                cv2.imwrite(
                    os.path.join(mask_pred, 'ph1', val_names[count].split('.')[0] + '.png'),
                    single_mask)
                count = count + 1


            target = target.unsqueeze(1)
            iou, dice, SE, PC, SP, ACC = iou_score(output, target)
            avg_meters['test_iou'].update(iou, input.size(0))
            avg_meters['test_dice'].update(dice, input.size(0))
            avg_meters['test_acc'].update(ACC, input.size(0))
            avg_meters['test_pc'].update(PC, input.size(0))
            avg_meters['test_se'].update(SE, input.size(0))
            avg_meters['test_sp'].update(SP, input.size(0))

    print('test_iou %.4f - test_dice %.4f' % (avg_meters['test_iou'].avg, avg_meters['test_dice'].avg))

    metrics = {
        'Metric': ['IOU', 'DICE', 'ACC', 'PC', 'SE', 'SP'],
        'Value': [avg_meters['test_iou'].avg, avg_meters['test_dice'].avg, avg_meters['test_acc'].avg,
                  avg_meters['test_pc'].avg, avg_meters['test_se'].avg, avg_meters['test_sp'].avg]
    }

    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(metrics)

    # 文件路径
    file_path = os.path.join(mask_pred,'ph1/Metric.xlsx')

    # 检查文件是否已存在
    if not os.path.exists(file_path):
        # 如果文件不存在，写入新的文件
        df.to_excel(file_path, index=False)
    else:
        # 如果文件已存在，读取现有文件并追加新数据
        existing_df = pd.read_excel(file_path)
        new_df = pd.concat([existing_df, df], ignore_index=True)
        new_df.to_excel(file_path, index=False)
    torch.cuda.empty_cache()


def Mamba_main():
    # print("=> creating model")
    config = vars(parse_args())
    model = net(config['model'])
    #需要改
    model_path = os.path.join("./output",config['model'],config['model_pth']+"1.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #需要改
    val_dataset = ThyroidDataset(os.path.join(get_dataset(config['dataset']),"val"), get_transform(train=False))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False,
                                             collate_fn=val_dataset.collate_fn, drop_last=False)
    # val_dataset = PolypDataset(os.path.join(get_dataset(config['dataset']),"val-seg"),load_transform(train=False))
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                           batch_size=24,
    #                                           shuffle=False,
    #                                           collate_fn=PolypDataset.collate_fn)
    val_names = val_dataset.names
    count = 0

    #需要改
    mask_pred = os.path.join("./output",config['model'],config['dataset'])
    os.makedirs(os.path.join(mask_pred, 'predict'), exist_ok=True)
    avg_meters = {'test_iou': AverageMeter(), 'test_dice': AverageMeter(),
                  'test_acc': AverageMeter(), 'test_pc': AverageMeter(), 'test_se': AverageMeter(),
                  'test_sp': AverageMeter()}
    with torch.no_grad():
        for input, target in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            output  = model(input)[0]
            mask = output.clone()
            mask = torch.sigmoid(mask).cpu().numpy() > 0.5
            #需要改
            for i in range(len(mask)):
                cv2.imwrite(
                    os.path.join(mask_pred, 'predict', val_names[count].split('.')[0] + '.png'),
                    (mask[i, 0] * 255).astype('uint8'))
                count = count + 1
            target = torch.unsqueeze(target,dim=1)
            iou, dice, SE, PC, SP, ACC = iou_score(output, target)
            avg_meters['test_iou'].update(iou, input.size(0))
            avg_meters['test_dice'].update(dice, input.size(0))
            avg_meters['test_acc'].update(ACC, input.size(0))
            avg_meters['test_pc'].update(PC, input.size(0))
            avg_meters['test_se'].update(SE, input.size(0))
            avg_meters['test_sp'].update(SP, input.size(0))

    print("Test result: test_mIoU:", avg_meters['test_iou'].avg,"test_Dice:",avg_meters['test_dice'].avg)
    metrics = {
        'Metric': ['IOU', 'DICE', 'ACC', 'PC', 'SE', 'SP'],
        'Value': [avg_meters['test_iou'].avg, avg_meters['test_dice'].avg, avg_meters['test_acc'].avg,
                  avg_meters['test_pc'].avg, avg_meters['test_se'].avg, avg_meters['test_sp'].avg]
    }

    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(metrics)

    # 文件路径
    file_path = os.path.join(mask_pred,'Metric.xlsx')

    # 检查文件是否已存在
    if not os.path.exists(file_path):
        # 如果文件不存在，写入新的文件
        df.to_excel(file_path, index=False)
    else:
        # 如果文件已存在，读取现有文件并追加新数据
        existing_df = pd.read_excel(file_path)
        new_df = pd.concat([existing_df, df], ignore_index=True)
        new_df.to_excel(file_path, index=False)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    Mamba_main()
