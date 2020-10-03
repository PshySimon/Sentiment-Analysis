"""
前馈神经网络
分三层：
        输入层
        隐藏层
        输出层
"""
import torch
import torch.nn as nn

class FNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.input(x)
        out = self.relu(out)
        out = self.hidden(out)
        out = self.relu(out)
        out = self.output(out)
        return out



