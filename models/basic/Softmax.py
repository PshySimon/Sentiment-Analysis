"""
Softmax的实现：
"""
import torch
import torch.nn as nn

class Softmax(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        # 在实际使用中，一般不直接在网络中进行softmax归一化，而是在CrossEntropy中实现
        # 这样做的目的是为了保证数值的稳定性，避免指数运算过大或者过小而导致溢出
        # out = torch.softmax(out, dim=1)
        return out
