import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
"""
https://viblo.asia/p/separable-convolutions-toward-realtime-object-detection-applications-aWj534bpK6m
"""
"""
Mục đích xây dựng
Bổ sung thông tin cục bộ vào chuỗi đầu vào seq_input, 
ví dụ như các đặc trưng phát âm gần nhau

Là 1 trong 4 khối chính của Conformer Block
Feedforward -> MultiHead -> Convolution(*) -> Feedforward
"""
class ConvolutionalModule(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm : str = "batch_norm",
        causal: bool = False,
        bias : bool = True,
    ):
        super()._init__()
        self.activation = activation
        self.pointwise_conv = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size= 1,
            stride = 1,
            padding = 0,
            bias = bias
        )
        if causal: 
            padding = 0
            self.lorder =  kernel_size - 1
        else:
            assert kernel_size % 2 == 0, "kernel_size must be even for non-causal convolutions"
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = padding,
            groups = channels,
            bias = bias,
        )
        assert norm in ["layer_norm", "batch_norm"]
        if norm == "layer_norm":
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)
        else:
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        
        self.pointwise_conv2 = nn.Conv1d(
            2 * channels,
            channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = bias
        )
    def forward(
        self,
        x : Tensor,
        mask_pad: Tensor = torch.ones((0,0,0), dtype=torch.bool),
        cache: Tensor = torch.zeros((0,0,0)) 
    )-> Tuple[Tensor, Tensor]:
        """
        Input Tensor: x : batchsize x time x channels
        mask_pad  Tensor : sử dụng padding batch (batch , 1, time)
        cache : left contexxt cache used for causal convolutions batch  x channles x cahce_t
        
        """
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:
            x = x.masked_fill (~mask_pad, 0.0)
        if self.lorder > 0 :
            if cache.size(2) == 0:
                x = nn.functional.pad(x, (self.lorder, 0) , 'constant', 0.0 )

            else:
                assert cache.size(0) == x.size(0)
                assert cache.size(1) == x.size(1)

                x = torch.cat((cache, x), dim = 2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:,:,-self.lorder:]
        else:
            new_cache = torch.zeros((0,0,0), dtype = x.dtype, device = x.device)
        x = self.pointwise_conv(x)
        x = nn.functional.glu(x, dim = 1)

        x = self.deppthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1,2)
            x = self.norm(x)

        return
class MyConvolution(nn.Module):

    pass
if __name__ == "__main__":
    print()
    x = torch.randn(1, 64, 100) #Input tensor
    