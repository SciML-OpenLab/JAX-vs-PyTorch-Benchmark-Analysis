import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils.device import get_torch_device

class TinyTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=2, num_layers=2, num_classes=10):
        super().__init__()

        self.embedding = nn.Linear(32, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)
        x = self.transformer(x)

        # Take mean over sequence
        x = x.mean(dim=1)

        return self.classifier(x)

def generate_data(batch_size=64, seq_len=32, feature_dim=32):
    x = torch.randn(batch_size, seq_len, feature_dim)
    y = torch.randint(0, 10, (batch_size,))
    return x, y

def train(config):
    device = get_torch_device()

    model = TinyTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()

    for epoch in range(config["epochs"]):
        x, y = generate_data()

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

    end = time.time()

    return {
        "time": end - start,
        "final_loss": loss.item()
    }
