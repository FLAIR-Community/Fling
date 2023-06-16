import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, inplane, outplane):
        super().__init__()
        self.conv_bn_relu_1 = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=3, padding=1), nn.BatchNorm2d(outplane), nn.ReLU()
        )

        self.conv_bn_relu_2 = nn.Sequential(
            nn.Conv2d(outplane, outplane, kernel_size=3, padding=1), nn.BatchNorm2d(outplane), nn.ReLU()
        )

        self.conv_bn_relu_3 = nn.Sequential(
            nn.Conv2d(outplane, outplane, kernel_size=3, padding=1), nn.BatchNorm2d(outplane), nn.ReLU()
        )

        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_bn_relu_1(inputs)
        x = self.max_pool2d(x)
        y = self.conv_bn_relu_2(x)
        y = self.conv_bn_relu_3(y)
        x = x + y
        return x


class CifarRes(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res1 = ResBlock(inplane=64, outplane=128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res2 = ResBlock(inplane=256, outplane=512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512, num_classes)

    def compute_feature(self, x):
        y = self.pre(x)
        y = self.res1(y)
        y = self.conv1(y)
        y = self.res2(y)
        y = self.head(y)
        return y

    def forward(self, x):
        y = self.pre(x)
        y = self.res1(y)
        y = self.conv1(y)
        y = self.res2(y)
        y = self.head(y)
        y = self.fc(y)
        return y

    def finetune_parameters(self, ftype):
        res = []
        if ftype == 'all':
            use_keys = self.state_dict().keys()
        elif ftype == 'fc':
            use_keys = []
            for k in self.state_dict().keys():
                if 'fc' in k:
                    use_keys.append(k)
        else:
            raise ValueError

        print('Finetune Keys: ' + str(use_keys))

        for key, param in self.named_parameters():
            if key in use_keys:
                res.append(param)
        return res
