import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

import math
from typing import Optional, List

from fling.utils.registry_utils import MODEL_REGISTRY

class LoRALayer():
    def __init__(
        self,
        r,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
            self.weight.requires_grad = True
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, mode='all'):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if mode == 'all':
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        elif mode == 'sigma':
            return F.linear(x, T(self.weight), bias=self.bias)
        elif mode == 'tau':
            result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else: raise NotImplementedError

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((int(r * out_channels) * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, int(r*out_channels)*kernel_size))
            )
            self.scaling = self.lora_alpha / int(self.r * out_channels)
            # Freezing the pre-trained weight matrix
            # self.conv.weight.requires_grad = False
            self.conv.weight.requires_grad = True
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x, mode='all'):
        if mode == 'all':
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        elif mode == 'sigma':
            return self.conv(x)
        elif mode == 'tau':
            return self.conv._conv_forward(
                x,
                (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        else: raise NotImplementedError

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, r: float = 0.5, lora_alpha: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, r=r, lora_alpha=lora_alpha)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, r: float = 0.5, lora_alpha: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, r=r, lora_alpha=lora_alpha)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            r: float = 0.5,
            lora_alpha: int = 1
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, r=r, lora_alpha=lora_alpha)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, r=r, lora_alpha=lora_alpha)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, mode='all') -> Tensor:
        identity = x

        out = self.conv1(x, mode=mode)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, mode=mode)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample[0](x, mode=mode)
            identity = self.downsample[1](identity)
            # identity = self.downsample(x, mode=mode)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock]],
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            input_channel: int = 3,
            class_number: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            Conv_r: float = 0.5,
            Linear_r: int = 4,
            lora_alpha: int = 1
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = Conv2d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, r=Conv_r, lora_alpha=lora_alpha)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = [self._make_layer(block, 64, layers[0], Conv_r=Conv_r, lora_alpha=lora_alpha)]
        for num in range(1, len(layers)):
            self.layers.append(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1], Conv_r=Conv_r, lora_alpha=lora_alpha))
        self.layers = nn.Sequential(*self.layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_labels)
        self.fc = Linear(features[len(layers) - 1] * block.expansion, class_number, r=Linear_r, lora_alpha=lora_alpha)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, Conv_r: float = 0.5, lora_alpha: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, r=Conv_r, lora_alpha=lora_alpha),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, r=Conv_r, lora_alpha=lora_alpha))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, r=Conv_r, lora_alpha=lora_alpha))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, mode='all') -> Tensor:
        x = self.conv1(x, mode=mode)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.layers(x, mode=mode)
        for layer in self.layers:
            for block in layer:
                x = block(x, mode=mode)
            # x = layer(x, mode=mode)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x, mode=mode)
        # return F.log_softmax(x, dim=1)
        return x

    def forward(self, x: Tensor, mode='all') -> Tensor:
        return self._forward_impl(x, mode=mode)

@MODEL_REGISTRY.register('lora_resnet18')
def lora_resnet18(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

@MODEL_REGISTRY.register('lora_resnet10')
def lora_resnet10(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)

@MODEL_REGISTRY.register('lora_resnet8')
def lora_resnet8(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1], **kwargs)

@MODEL_REGISTRY.register('lora_resnet6')
def lora_resnet6(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return ResNet(BasicBlock, [1, 1], **kwargs)

@MODEL_REGISTRY.register('lora_resnet4')
def lora_resnet4(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return ResNet(BasicBlock, [1], **kwargs)