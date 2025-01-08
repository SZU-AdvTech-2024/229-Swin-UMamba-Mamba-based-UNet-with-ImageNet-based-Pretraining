from os.path import split
import argparse
import logging
import os
import na
import sys

import cv2
import imageio
import glob
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from SwinUnet.datasets.dataset_thyroid import ImageFolder
from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./dataset/Thyroid',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Thyroid', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Thyroid', help='list dir')
parser.add_argument('--output_dir',default='./model_out/test', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../model_out', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--split_name", default="test_vol", help="Directory of the input list")

args = parser.parse_args()

# if args.dataset == "Synapse":
#     args.root_path = os.path.join(args.root_path, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = ImageFolder(root_path=args.root_path,mode='test')
    testloader = DataLoader(db_test, batch_size=2, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # if args.dataset == "evaluate":
        #     case_name = split(case_name.split(",")[0])[-1]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def single_inference(model,model_path,test_path,save_path):
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    im_names = os.listdir(test_path)
    for name in im_names:
        full_path = os.path.join(test_path,name)
        img = cv2.imread(full_path)
        image = np.array(img,np.float32)/255.0*3.2-1.6
        image = np.array(image,np.float32).transpose(2,0,1)
        image = np.expand_dims(image,axis=0)
        image = torch.Tensor(image)
        image = image.cuda()
        output  = model(image).cpu().data.numpy()
        output[output<0.5] = 0
        output[output>=0.5] = 1
        output = np.squeeze(output)
        save_full = os.path.join(save_path,name)
        cv2.imwrite(save_full,output*255)


if __name__ == "__main__":
    args = parser.parse_args()
    config = get_config(args)
    net = ViT_seg(config, img_size=224, num_classes=args.num_classes).cuda()
    test_root = "./evaluate/Thyroid/imagesTs/"
    test_save_path = "./model_out/thyroid/pred_img/"
    model_path = "./model_out/thyroid/best_model"
    single_inference(net, model_path, test_root, test_save_path)
    # if not args.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    #
    # dataset_name = args.dataset
    # dataset_config = {
    #     args.dataset: {
    #         'root_path': args.root_path,
    #         'list_dir': f'./lists/{args.dataset}',
    #         'num_classes': 2,
    #         "z_spacing": 1
    #     },
    # }
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.root_path = dataset_config[dataset_name]['root_path']
    # # args.Dataset = dataset_config[dataset_name]['Dataset']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.z_spacing = dataset_config[dataset_name]['z_spacing']
    # args.is_pretrain = True
    #
    # net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    #
    # # snapshot = os.path.join(args.output_dir, 'best_model.pth')
    # snapshot = './model_out/best_model.pth'
    # if not os.path.exists(snapshot):
    #     snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    # msg = net.load_state_dict(torch.load(snapshot))
    # print("self trained swin unet", msg)
    # snapshot_name = snapshot.split('/')[-1]
    #
    # log_folder = './test_log/test_log_'
    # os.makedirs(log_folder, exist_ok=True)
    # logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # logging.info(snapshot_name)
    #
    # if args.is_savenii:
    #     args.test_save_dir = os.path.join(args.output_dir, "predictions")
    #     test_save_path = args.test_save_dir
    #     os.makedirs(test_save_path, exist_ok=True)
    # else:
    #     test_save_path = None
    # inference(args, net, test_save_path)

# python train.py --dataset Synapse --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE
# python train.py --output_dir './model_out/evaluate' --dataset evaluate --img_size 224 --batch_size 32 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/nnUNetPlans_2d_split
# python test.py --output_dir ./model_out/evaluate --dataset evaluate --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/test --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

# 数字标签路径
# path = r'output/*.png'
# # 设置标签颜色（这里是7种）
# colors = [[0, 0, 0], [255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 255, 255]]
#
# for file in glob.glob(path):
#     # print(file)
#
#     label = imageio.imread(file)
#     h, w = label.shape
#     label_rgb = np.zeros((h, w, 3))
#
#     for i, rgb in zip(range(7), colors):
#         # print(i,rgb) # 数字对应颜色
#         label_rgb[label == i] = rgb
#     # 保存图片
#     imageio.imsave(r'output1/' + file.split('\\')[-1], label_rgb)