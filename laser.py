import copy
import random
import numpy as np
import torch.nn as nn
from layers import LaserLinear
from eval import OnlineEvaluator

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
    return new_module.error, new_module.compression

def apply_laser(model: nn.Module, target_modules: list[str], bottleneck_features: int):
    errors = []
    compression = []
    for target_module in target_modules:
        error, comp = apply_laser_single(model, target_module, bottleneck_features)
        errors.append(error)
        compression.append(comp)
    return sum(compression), np.mean(errors)

def generate_proposals(
    model: nn.Module,
    num_modules: int,
    num_proposals: int,
    bottleneck_features: int,
    target_patterns: list[str],
    exclude_patterns: list[str],
):  
    linear_layers = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear) and 
            any([pattern in name for pattern in target_patterns])
        ):
            linear_layers.append(name)
    linear_layers = [x for x in linear_layers if "laser" not in x]
    proposals = set()
    while len(proposals) < num_proposals:
        proposal = random.sample(linear_layers, k=num_modules)
        proposals.add(tuple(proposal))
    
    return proposals

def evaluate_proposal(
    model: nn.Module,
    target_modules: list[str],
    bottleneck_features: int,
    evaluator: OnlineEvaluator
):
    replica = copy.deepcopy(model)
    compression, error = apply_laser(replica, target_modules, bottleneck_features)
    avg, results = evaluator.run(replica)
    return {
        "layers": target_modules,
        "score": avg,
        "compression": compression,
        "error": error,
    }