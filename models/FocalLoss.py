import torchvision
from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x, target):
        one_hot = nn.functional.one_hot(target, num_classes=3).type(torch.float)
        return torchvision.ops.sigmoid_focal_loss(x, one_hot, gamma=self.gamma, alpha=self.alpha, reduction='mean')
