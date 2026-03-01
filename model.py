import torch.nn as nn
from torchvision import models

# function: swap_norm_bn_gn
# swaps BatchNorm with GroupNorm to test different normalisation techniques
def swap_norm_bn_gn(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            groups = min(num_groups, num_channels)
            # GroupNorm requires channels to be evenly divisible by number of groups
            while num_channels % groups != 0:
                groups //= 2
            setattr(model, name, nn.GroupNorm(groups, num_channels))
        else:
            swap_norm_bn_gn(module, num_groups)
    return model

# function: swap_norm_bn_ln
# swaps BatchNorm with LayerNorm
# PyTorch doesn't have LayerNorm2d for convolutional layers so approximated using GroupNorm with num_groups=1
def swap_norm_bn_ln(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            setattr(model, name, nn.GroupNorm(1, num_channels))
        else:
            swap_norm_bn_ln(module)
    return model

# function: build_model
# builds the model depending on the architecture chosen
def build_model(architecture="resnet18", norm="batch", dropout=0.0, num_classes=200):
    # architectures: resnet18, resnet34, mobilenet
    # norms: batch, group, layer
    # dropout: applied before final FC

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)

    elif architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)

    elif architecture == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)

    else: 
        raise ValueError(f"unknown architecture: {architecture}")

    if norm == "group":
        swap_norm_bn_gn(model)
    elif norm == "layer":
        swap_norm_bn_ln(model)

    # used to compare dropout rates of 0.0, 0.3, and 0.5
    if (dropout > 0) and (hasattr(model, 'fc')):
        if isinstance(model.fc, nn.Linear):
            in_f = model.fc.in_features
        else:
            in_f = model.fc[-1].in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Linear(in_f, num_classes)
        )
        
    return model