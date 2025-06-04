import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from uhrnet_architecture import UHRNet
from SSIM import SSIM
from tqdm import tqdm

# ==== CONFIGURATION ====
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "UHRNet_weight.pth"

# ==== LOAD DATA ====
print("Loading dataset...")

# Load and reshape fringe pattern
x = np.load("data_npy/f1_80.npy")
if x.ndim == 3:
    x = np.expand_dims(x, axis=1)  # (N, 1, H, W)
elif x.shape[-1] == 1:
    x = np.transpose(x, (0, 3, 1, 2))
x = torch.from_numpy(x).float()

# Load and reshape height map (Z)
y = np.load("data_npy/Z.npy")
if y.ndim == 3:
    y = np.expand_dims(y, axis=1)  # (N, 1, H, W)
elif y.shape[-1] == 1:
    y = np.transpose(y, (0, 3, 1, 2))
y = torch.from_numpy(y).float()

# Dataset
dataset = TensorDataset(x, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# ==== MODEL SETUP ====
model = UHRNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_mse = nn.MSELoss()
criterion_ssim = SSIM()

# ==== TRAINING LOOP ====
print("Starting training...")
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion_mse(outputs / 6, targets / 6)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_ssim = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion_mse(outputs / 6, targets / 6).item()
            val_ssim += criterion_ssim(outputs + 105, targets + 105).item()

    val_loss /= len(val_loader)
    val_ssim /= len(val_loader)
    print(f"Validation MSE: {val_loss:.4f} | SSIM: {val_ssim:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }, SAVE_PATH)
        print("✅ Model saved.")

print("✅ Training complete.")
