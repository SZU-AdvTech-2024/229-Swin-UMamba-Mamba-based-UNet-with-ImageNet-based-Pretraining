import os
import math
import tempfile
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import functional as F
from model import tri_mamba,shufflenet_v2_x1_0
from my_dataset import MyDataSet
from multi_train_utils import train_one_epoch, evaluate, init_distributed_mode, dist, cleanup
from model1 import medmamba_t
from model import tri_mamba
from models.vmamba import vanilla_vmamba_tiny
from timm.models.resnet import resnet50
class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        return F.resize(data, [self.size_h, self.size_w])
def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    num_classes = args.num_classes
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir="./runs/medmamba")
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    data_transform = {
        "train": transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_path
    json_path = "./classes_name.json"
    # 实例化训练数据集
    train_dataset = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=data_transform["train"])

    # check num_classes
    if args.num_classes != len(train_dataset.labels):
        raise ValueError("dataset have {} classes, but input {}".format(len(train_dataset.labels),
                                                                        args.num_classes))

    # 实例化验证数据集
    val_dataset = MyDataSet(root_dir=data_root,
                            csv_name="new_val.csv",
                            json_path=json_path,
                            transform=data_transform["val"])
    print(len(train_dataset.__getitem__(0)[0]))
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 8 number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)
    # 实例化模型
    # model = EfficientNetBN(model_name="efficientnet-b0",num_classes=100).cuda()
    # model = vanilla_vmamba_tiny().cuda()
    # model  = tri_mamba()
    # model = resnet50(num_classes=100,pretrained=True).cuda()
    # model  = shufflenet_v2_x1_0(num_classes=args.num_classes)
    #改进结构
    model = medmamba_t()
    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        # for key in weights_dict.keys():
        #     print(key)
        # load_weights_dict = {k: v for k, v in weights_dict.items()
        #                      if model.state_dict()[k].numel() == v.numel()}
        # model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()
        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        if rank == 0:
            print("[epoch {}] val accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), "./weights/MedModel.pth")

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    print("Training finished")
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--gpus', type=str, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="./data/mini-imagenet/")

    parser.add_argument('--weights', type=str, default='./weights/MedMamba.pth',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')

    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
    # python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=3 --master_addr="127.0.0.1" --master_port=29501 train_launch.py