import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, optim

def train_model(model, dataset, epochs=10, batch_size=8, lr=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for img, clinical, target in dataloader:
            img, clinical, target = img.to(device), clinical.to(device), target.to(device)
            output = model(img, clinical).squeeze()
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")
