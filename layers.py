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
        error: Optional[float] = None # to store reconstruction error
    ):
        super().__init__()
        self.error = error
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

        # measure the reconstruction error (frobenius norm)
        error = torch.linalg.norm(
            torch.mm(torch.mm(u_r, torch.diag(s_r)), vh_r) - linear.weight.data,
            ord="fro"
        )

        # create a new instance of LaserLinear
        new_linear = cls(
            linear.in_features,
            bottleneck_features,
            linear.out_features,
            linear.bias is not None,
            # sum the error, return normal float not tensor
            error=error.sum().item()
        )

        # set the weights of the new instance
        new_linear.laser1.weight.data = torch.mm(torch.diag(s_r), vh_r)
        new_linear.laser2.weight.data = u_r

        # set the bias of the new instance
        if linear.bias is not None:
            new_linear.laser2.bias.data = linear.bias.data

        return new_linear

def replace_module(model, module_name, new_module):
    """
    Replace a submodule within a model given the submodule's name and the new module.

    :param model: The model or submodule containing the module to replace.
    :param module_name: The name of the module to replace.
    :param new_module: The new module to insert in place of the old one.
    """
    # Split the module name into parts
    parts = module_name.split('.')
    # Access the submodule that is the parent of the target module
    parent = model
    for part in parts[:-1]:  # Go until the second last part
        parent = getattr(parent, part)
    
    # Replace the target module
    setattr(parent, parts[-1], new_module)

def apply_laser_single(model: nn.Module, target_module: str, bottleneck_features: int):
    # Find the target module
    for name, module in model.named_modules():
        if name == target_module:
            # Once the target module is found, create a new module to replace it
            new_module = LaserLinear.from_linear(module, bottleneck_features)
            # Replace the target module with the new module using the replace_module function
            replace_module(model, target_module, new_module)
            break
    else:
        raise AttributeError(f"No module named '{target_module}' found in the model.")
    return new_module.error

def apply_laser(model: nn.Module, target_modules: list[str], bottleneck_features: int):
    errors = []
    for target_module in target_modules:
        errors.append(apply_laser_single(model, target_module, bottleneck_features))
    return errors
