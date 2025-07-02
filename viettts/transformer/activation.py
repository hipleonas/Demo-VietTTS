import torch

import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor

class Swish(nn.Module):
    def forward(self, x: Tensor)-> Tensor:
        return x * torch.sigmoid(x)
class Snake(nn.Module):
    """
    The snake activation function is a 
    neural network activation function designed 
    to model and extrapolate periodic functions. 
    It's defined as snake(x) = x + (1/alpha) * sin^2(alpha * x), 
    where alpha is a trainable parameter that controls the frequency of the periodic component. This function was introduced as a solution to the inability of standard activation functions like ReLU and tanh to learn and extrapolate periodic data. 
    input đầu vào của forward
    Batchsize x Channel(Features) x Timestep
    x.shape = [4, 64, 100]
        + Có 4 đoạn audio (B = 4)

        + Mỗi đoạn có 64 đặc trưng (C = 64)

        + Mỗi đặc trưng kéo dài qua 100 bước thời gian (T = 100)
    """
    def __init__(
        self,
        in_features : int,
        alpha: float = 1.0,
        alpha_trainable:bool = True,
        alpha_logscale:bool = False,
    ):
        super(Snake, self).__init__()

        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: Tensor)-> Tensor:

        """
        Snake ∶= x + 1/a * sin^2 (xa)

        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x +(1.0 / (self.alpha + self.eps)) * torch.pow(torch.sin(x * alpha), 2)
        return x
if __name__  == "__main__":
    x = torch.randn(4, 64, 1)
    snake = Snake(in_features=64, alpha=1.0, alpha_trainable=True, alpha_logscale=False)
    output = snake(x)
    print(output.shape)  # Expected output shape: [4, 64, 64]