import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import CIFAR10

from fling.utils.visualize_utils import plot_2d_loss_landscape

if __name__ == '__main__':
    # Step 1: prepare the dataset.
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10('./data/cifar10', transform=transform)

    # Test dataset is for generating loss landscape.
    test_dataset = [dataset[i] for i in range(100)]
    test_dataloader = DataLoader(test_dataset, batch_size=100)

    # Step 2: prepare the model.
    model = resnet18(pretrained=False, num_classes=10)

    # Step 3: train the randomly initialized model.
    dataloader = DataLoader(dataset, batch_size=100)
    device = 'cuda'
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(10):
        for _, (data_x, data_y) in enumerate(dataloader):
            data_x, data_y = data_x.to(device), data_y.to(device)
            pred_y = model(data_x)
            loss = criterion(pred_y, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.to('cpu')

    # Step 4: plot the loss landscape after training the model.
    # Only one line of code for visualization!
    plot_2d_loss_landscape(
        model=model,
        dataloader=test_dataloader,
        device='cuda',
        caption='Loss Landscape Trained',
        save_path='./landscape.pdf',
        noise_range=(-0.01, 0.01),
        resolution=30,
        log_scale=True
    )
