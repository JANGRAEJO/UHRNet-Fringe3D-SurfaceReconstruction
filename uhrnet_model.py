import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from SSIM import SSIM

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
print("Loading dataset...")
fringepath = 'data_npy/f1_80.npy'
gtpath = 'data_npy/Z.npy'

pic = np.load(fringepath)  # shape: (N, H, W) or (N, H, W, 1)
gt = np.load(gtpath)

# Ensure (N, 1, H, W)
if pic.ndim == 3:
    pic = np.expand_dims(pic, axis=1)  # (N, 1, H, W)
elif pic.shape[-1] == 1:
    pic = np.transpose(pic, (0, 3, 1, 2))  # (N, 1, H, W)

if gt.ndim == 3:
    gt = np.expand_dims(gt, axis=1)
elif gt.shape[-1] == 1:
    gt = np.transpose(gt, (0, 3, 1, 2))

pic = torch.from_numpy(pic).float()
gt = torch.from_numpy(gt).float()

# Train/val split
x_train, x_val, y_train, y_val = train_test_split(pic, gt, test_size=0.2, random_state=0)

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize model and loss
model = UHRNet().to(device)
criterion_mse = nn.MSELoss()
criterion_ssim = SSIM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
print("Starting training...")
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss_mse = criterion_mse(outputs / 6, targets / 6)
        loss_ssim = criterion_ssim(outputs + 105, targets + 105)
        loss = loss_mse + (1 - loss_ssim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

# Save the trained model
torch.save({'state_dict': model.state_dict()}, 'UHRNet_weight.pth')
print("Training complete. Model saved as 'UHRNet_weight.pth'")
