"""
简单循环神经网络
"""
import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size,bias=False)
        self.H = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, X, hidden_state = None):
        # 输入数据形状要求是[num_steps, batch_size, input_size]
        # hidden_state形状为[batch_size, hidden_size]
        if hidden_state is None:
            hidden_state = torch.zeros(X.shape[1], self.hidden_size)
        output = []
        out = None
        for x in X:
            out = torch.tanh(self.W(x) + self.H(hidden_state))
            output.append(out)
        return torch.stack(output, dim=0), out

if __name__ == "__main__":
    # 测试一下
    model = RNN(10,15)
    X = torch.ones(5, 6, 10)
    y,z = model(X)
    print(y.shape, z.shape)




