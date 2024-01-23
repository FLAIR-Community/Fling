from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from fling.utils.attack_utils import DLGAttacker


class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.flat(self.pool(x))
        return self.fc(x)


if __name__ == '__main__':
    # Step 1: prepare the attack dataset.
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10('./data/cifar10', transform=transform)
    test_dataset = [dataset[i] for i in range(1)]

    # Step 2: prepare the model.
    model = ToyModel()

    # Step 3: initialize the attacker.
    attacker = DLGAttacker(iteration=300, working_dir='./dlg_attacker', iteration_per_save=10)

    # Step 4: attack.
    attacker.attack(model, test_dataset, device='cuda', class_number=10, save_img=True)
