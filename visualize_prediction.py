import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from uhrnet_architecture import UHRNet

# === CONFIGURATION ===
FRINGE_PATH = 'data_npy/f1_80.npy'
GROUND_TRUTH_PATH = 'data_npy/Z.npy'
MODEL_PATH = 'UHRNet_weight.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'output_vis'
BATCH_SIZE = 4

# === LOAD MODEL ===
model = UHRNet().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# === LOAD DATA ===
x = np.load(FRINGE_PATH)
y = np.load(GROUND_TRUTH_PATH)

# Ensure shape is (N, 1, H, W)
if x.ndim == 3:
    x = np.expand_dims(x, axis=1)
elif x.shape[-1] == 1:
    x = np.transpose(x, (0, 3, 1, 2))
if y.ndim == 3:
    y = np.expand_dims(y, axis=1)
elif y.shape[-1] == 1:
    y = np.transpose(y, (0, 3, 1, 2))

# Convert to tensor
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# === PREDICTION IN BATCHES ===
model.to(DEVICE)
preds = []
gt = []

with torch.no_grad():
    for i in range(0, len(x_tensor), BATCH_SIZE):
        batch_x = x_tensor[i:i+BATCH_SIZE].to(DEVICE)
        batch_y = y_tensor[i:i+BATCH_SIZE]
        batch_pred = model(batch_x).cpu().numpy()
        preds.append(batch_pred)
        gt.append(batch_y.numpy())

preds = np.concatenate(preds, axis=0)
gt = np.concatenate(gt, axis=0)

# === VISUALIZATION ===
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for i in range(len(preds)):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[i, 0], cmap='gray')
    axes[0].set_title('Input Fringe Pattern')
    axes[1].imshow(gt[i, 0], cmap='jet')
    axes[1].set_title('Ground Truth Height Map')
    axes[2].imshow(preds[i, 0], cmap='jet')
    axes[2].set_title('Predicted Height Map')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'vis_{i:02d}.png'))
    plt.close()

print(f"✅ Visualization saved to '{OUTPUT_DIR}'")
