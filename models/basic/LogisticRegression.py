"""
说明：基本模块的手动实现
模型：Logistic回归
"""
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        return out


