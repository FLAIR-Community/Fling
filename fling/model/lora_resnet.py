import math

import torch.nn as nn

from fling.utils.registry_utils import MODEL_REGISTRY


class LoraConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, r, lora_alpha, **kwargs):
        super(LoraConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.lora_alpha = lora_alpha
        self.r = r
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((int(r * out_channels) * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros(
                    (out_channels // self.conv.groups * kernel_size, int(r * out_channels) * kernel_size)
                )
            )
            self.scaling = self.lora_alpha / int(self.r * out_channels)
            self.conv.weight.requires_grad = True
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x, mode='all'):
        if mode == 'all':
            return self.conv._conv_forward(
                x, self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        elif mode == 'sigma':
            return self.conv(x)
        elif mode == 'tau':
            return self.conv._conv_forward(
                x, (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling, self.conv.bias
            )
        else:
            raise ValueError(f'Unrecognized computation mode: {mode}')


class ResBlock(nn.Module):

    def __init__(self, inplane, outplane, r, lora_alpha):
        super().__init__()
        self.conv_bn_relu_1 = nn.Sequential(
            LoraConv2d(inplane, outplane, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha),
            nn.BatchNorm2d(outplane), nn.ReLU()
        )

        self.conv_bn_relu_2 = nn.Sequential(
            LoraConv2d(outplane, outplane, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha),
            nn.BatchNorm2d(outplane), nn.ReLU()
        )

        self.conv_bn_relu_3 = nn.Sequential(
            LoraConv2d(outplane, outplane, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha),
            nn.BatchNorm2d(outplane), nn.ReLU()
        )

        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_bn_relu_1(inputs)
        x = self.max_pool2d(x)
        y = self.conv_bn_relu_2(x)
        y = self.conv_bn_relu_3(y)
        x = x + y
        return x


@MODEL_REGISTRY.register('lora_resnet')
class LoraRes(nn.Module):

    def __init__(self, r, lora_alpha, input_channel=3, class_number=10):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res1 = ResBlock(inplane=64, outplane=128, r=r, lora_alpha=lora_alpha)
        self.conv1 = nn.Sequential(
            LoraConv2d(128, 256, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res2 = ResBlock(inplane=256, outplane=512, r=r, lora_alpha=lora_alpha)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512, class_number)

    def forward(self, x):
        y = self.pre(x)
        y = self.res1(y)
        y = self.conv1(y)
        y = self.res2(y)
        y = self.head(y)
        y = self.fc(y)
        return y
