import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
from utils.device import get_torch_device

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train(config):
    device = get_torch_device()

    transform = transforms.ToTensor()
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()

    for epoch in range(config["epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    end = time.time()

    return {"time": end-start, "final_loss": loss.item()}
