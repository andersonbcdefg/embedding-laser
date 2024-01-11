import torch
import torch.nn as nn
from typing import Optional

class LaserLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        bottleneck_features: int,
        out_features: int,
        bias: bool,
        error: Optional[float] = None, # to store reconstruction error
        compression: Optional[int] = None # store # of params saved
    ):
        super().__init__()
        self.error = error
        self.compression = compression
        self.in_features = in_features
        self.bottleneck_features = bottleneck_features
        self.out_features = out_features
        self.bias = bias
        self.laser1 = nn.Linear(in_features, bottleneck_features, bias=False)
        self.laser2 = nn.Linear(bottleneck_features, out_features, bias=bias)
    
    def forward(self, x):
        return self.laser2(self.laser1(x))
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bottleneck_features: int,
    ):
        # perform SVD to approximate the original weight matrix
        u, s, vh = torch.linalg.svd(
            linear.weight.data,
            full_matrices=False
        )
        u_r = u[:, : bottleneck_features]
        s_r = s[: bottleneck_features]
        vh_r = vh[: bottleneck_features, :]

        # measure the reconstruction error (MSE)
        error = torch.mean((linear.weight.data - torch.mm(torch.mm(u_r, torch.diag(s_r)), vh_r)) ** 2)
        old_params = linear.weight.numel()
        new_params = bottleneck_features * (linear.in_features + linear.out_features)
        compression = old_params - new_params

        # create a new instance of LaserLinear
        new_linear = cls(
            linear.in_features,
            bottleneck_features,
            linear.out_features,
            linear.bias is not None,
            # sum the error, return normal float not tensor
            error=error.item(),
            compression=compression
        )

        # set the weights of the new instance
        new_linear.laser1.weight.data = torch.mm(torch.diag(s_r), vh_r)
        new_linear.laser2.weight.data = u_r

        # set the bias of the new instance
        if linear.bias is not None:
            new_linear.laser2.bias.data = linear.bias.data

        return new_linear
