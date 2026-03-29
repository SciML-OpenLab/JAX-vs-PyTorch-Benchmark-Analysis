import torch
import torch.nn as nn
import torch.optim as optim
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train(config):
    device = torch.device("cpu")

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    X = torch.randn(1000, 100).to(device)
    y = torch.randn(1000, 1).to(device)

    start = time.time()

    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    end = time.time()

    return {
        "time": end - start,
        "final_loss": loss.item()
    }
