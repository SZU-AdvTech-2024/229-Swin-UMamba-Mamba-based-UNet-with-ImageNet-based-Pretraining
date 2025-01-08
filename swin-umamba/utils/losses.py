import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

deep_supervision_scales = [1.0, 0.5, 0.25, 0.125]  # 示例比例
weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
weights[-1] = 0  # 忽略最低分辨率
weights = weights / weights.sum()       #归一化


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        bce = F.binary_cross_entropy_with_logits(input, target)
        # ce = CrossEntropyLoss()
        # bce = ce(input, target)

        input = torch.sigmoid(input)
        num = target.size(0)               #return batch_size
        # input = input.view(num, -1)
        # target = target.view(num, -1)
        input = input.reshape(num, -1)
        target = target.reshape(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5 * bce + dice

class deep_supervision_loss():
    def __init__(self, base_loss, weights=weights):
        """
        初始化深度监督损失包装器
        Args:
            base_loss (callable): 基础损失函数（如 BCE、Dice）。
            weights (list or ndarray): 每个输出的权重。
        """
        self.base_loss = base_loss
        self.weights = weights

    def __call__(self, outputs, target):
        """
        计算深度监督损失
        Args:
            outputs (list of Tensors): 模型多分辨率输出。
            target (Tensor): Ground Truth 标签。

        Returns:
            Tensor: 加权综合损失。
        """
        if target.ndimension() == 3:
            target = target.unsqueeze(1)
        loss = 0
        for i, output in enumerate(outputs):
            # 将 target 下采样到当前分辨率
            target_resized = F.interpolate(target, size=output.shape[2:], mode="nearest")
            # print(output.shape,target_resized.shape,target.shape)
            # 计算并加权损失
            loss += self.weights[i] * self.base_loss(output,target_resized)
        return loss
