import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0, reduction='mean', accumulate_iters=1, label_smoothing=0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.accumulate_iters = accumulate_iters
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        # Generate noise vector n for each label in the batch and normalize it
        noise = torch.rand(target.size(0), input.size(1), device=input.device)
        noise = F.normalize(noise, p=1, dim=1)

        # Apply label smoothing to the original target if it is a class index
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=input.size(1)).float()

        # Apply label smoothing
        smoothed_target = target * (1 - self.label_smoothing) + self.label_smoothing / input.size(1)

        # Modify target according to the epsilon and noise
        smoothed_target = (1 - self.epsilon) * smoothed_target + self.epsilon * noise

        # Calculate the cross entropy loss manually
        log_softmax = F.log_softmax(input, dim=1)
        if self.reduction == 'mean':
            ce_loss = -torch.mean(log_softmax * smoothed_target)
        if self.reduction == 'sum':
            ce_loss = -torch.sum(log_softmax * smoothed_target) / input.size(0)
        if self.reduction == 'log':
            ce_loss = -torch.sum(log_softmax * smoothed_target) / input.size(0) / np.log(input.size(1))

        return ce_loss / self.accumulate_iters


class CustomMSELossWithL2Reg(nn.Module):
    def __init__(self, model, lambda_reg=0.01, reduction = 'mean', accumulate_iters=1):
        super(CustomMSELossWithL2Reg, self).__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.reduction = reduction
        self.accumulate_iters=accumulate_iters

    def custom_mse_loss(self, predicted, target):
        """
        自作のMSE損失関数

        Args:
        predicted (torch.Tensor): 予測値
        target (torch.Tensor): 真のターゲット

        Returns:
        torch.Tensor: MSE損失
        """
        squared_diff = (predicted - target) ** 2  # 予測値と真のターゲットの差の二乗を計算
        if self.reduction == 'sum':
            # 出力次元の方向に合計し、サンプル数の方向に平均する
            mse_loss = torch.sum(squared_diff, dim=1).mean()  # dim=1 は出力次元を指す
        elif self.reduction == 'mean':
            # 既定の動作: 二乗差の平均を計算
            mse_loss = torch.mean(squared_diff)
        return mse_loss

    def forward(self, input, target):
        # Mean squared error loss
        mse_loss = self.custom_mse_loss(input, target)
        
        # L2 regularization
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        
        total_loss = mse_loss + self.lambda_reg * l2_reg
        
        return total_loss / self.accumulate_iters