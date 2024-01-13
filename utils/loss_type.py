import torch
import torch.nn as nn

class CustomCrossEntropyLossWithL2Reg(nn.Module):
    def __init__(self, model, lambda_reg=0.01, label_smoothing=0, reduction = 'mean', accumulate_iters=1):
        super(CustomCrossEntropyLossWithL2Reg, self).__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.label_smoothing=label_smoothing
        self.reduction=reduction
        self.accumulate_iters=accumulate_iters
        
    def forward(self, input, target):
        # Cross entropy loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, reduction=self.reduction)
        ce_loss = self.cross_entropy_loss(input, target)
        
        # L2 regularization
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        
        total_loss = ce_loss + self.lambda_reg * l2_reg
        return total_loss / self.accumulate_iters

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
        mse_loss = torch.mean(squared_diff)  # 二乗差の平均を計算
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