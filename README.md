# 🌊 UHRNet for Fringe Pattern to Height Map Conversion

UHRNet for Fringe Pattern to Height Map Conversion

This repository contains a PyTorch implementation and full training/inference pipeline for converting **structured-light fringe pattern images** into **surface height maps** using a deep learning model based on the UHRNet architecture.

Fringe pattern images—generated through **structured light projection techniques**—encode 3D surface geometry by projecting sinusoidal patterns onto an object. As the pattern deforms over the object's shape, it carries depth-related information that can be interpreted to reconstruct a 3D profile.

This approach is particularly useful in **optical metrology** and **non-contact surface inspection**, where traditional methods such as phase unwrapping or multi-pattern acquisition can be complex or computationally expensive.

Our model provides a direct, end-to-end solution for **interpreting single fringe images** and reconstructing corresponding 3D height maps, enabling fast and scalable inspection pipelines in **precision manufacturing**, **semiconductor metrology**, and **optical surface analysis**.

## 📌 Project Summary

This project aims to train a deep neural network to interpret interference fringe patterns and output corresponding surface height maps, enabling **non-contact, high-resolution surface measurement** at micrometer-level precision. This is a crucial step for building practical tools in optics and precision manufacturing systems.

## 📁 Dataset

The original dataset used in this project is sourced from the supplementary materials of the paper:

- **Title**: [A deep-learning method for fringe pattern analysis with UHRNet](https://doi.org/10.1364/OE.485239)
- **Authors**: Huang et al., Optics Express, 2023
- **Original GitHub**: [https://github.com/fead1/UHRNet](https://github.com/fead1/UHRNet)

The dataset includes:
- `f1_80.npy`: Single fringe pattern input
- `Z.npy`: Corresponding ground truth height map

> 📌 The original authors' pretrained model is not accessible outside China. Therefore, I trained the UHRNet model from scratch using their provided data.

## 🧠 Model & Training

### Architecture
- Model: UHRNet (modified version from the original GitHub)
- Framework: PyTorch
- Loss Function: MSE (Mean Squared Error)
- Training: Full training from scratch using **all available data** (not just 128 samples as used in the original example)

### My Improvements:
- ✅ Full-dataset training instead of partial
- ✅ Batched prediction and inference support
- ✅ Visualization code to compare:
  - Input Fringe Pattern
  - Ground Truth Height Map
  - Predicted Height Map
- ✅ Optional 3D surface plot generation using `matplotlib`

### 🖼️ Understanding the Demo Output

The GIF below presents five synchronized visualizations for each sample in a single frame:

| Input Fringe Pattern | Ground Truth Height Map | Predicted Height Map |
|----------------------|-------------------------|-----------------------|
| Ground Truth 3D Surface | Predicted 3D Surface |


![Visualization Demo](combined_visualization_part_00)
![Visualization Demo](combined_visualization_part_01)

**Descriptions:**

- **Input Fringe Pattern**: A grayscale structured-light image encoding subtle surface deformations.
- **Ground Truth Height Map**: The reference 3D surface (in mm or scaled units), obtained from high-precision scanning.
- **Predicted Height Map**: Depth output from the trained UHRNet model, inferred from just the fringe image.
- **Ground Truth 3D Surface**: A rendered surface plot of the actual height map for visual validation.
- **Predicted 3D Surface**: A 3D plot of the model’s prediction, enabling intuitive comparison with the true shape.

> **Color Map Interpretation**: Red/Yellow = High surface regions, Blue = Low surface regions.

## 📊 Inference & Output

- Inference script: `visualize_prediction.py`
- Output:
  - `.npy` files for predictions and ground truth
  - PNG visualizations (side-by-side fringe, ground truth, predicted)
  - Optional MP4 video made from results
  - Optional 3D surface plots for more intuitive interpretation

## 🧱 Directory Structure

```
UHRNet/
│
├── data_npy/               # Fringe and height map input
│   ├── f1_80.npy
│   └── Z.npy
│
├── output_vis/             # Output images and .npy predictions
│   ├── vis_0000.png
│   ├── preds.npy
│   └── gt.npy
│
├── visualize_prediction.py # Main inference & visualization script
├── make_video_from_images.py
├── make_3d_plot.py         # Optional 3D surface rendering (if enabled)
└── uhrnet_architecture.py  # UHRNet model definition
```

## 📊 Model Performance

The model was trained on the full [SIDO 3D Reconstruction Dataset](https://figshare.com/articles/dataset/Single-input_dual-output_3D_shape_reconstruction/19709134).  
Unlike the original paper which used only 128 samples, this implementation utilizes the entire dataset for more robust generalization.

| Metric                  | Value         |
|------------------------|---------------|
| **Validation MSE**     | 13.1037       |
| **Validation SSIM**    | 0.8698        |
| **Test Input Resolution** | 640 × 352 |
| **Model Size**         | 30.33M params |

> 📌 The original paper reported a mean RMSE of 0.443mm and SSIM of 0.9978, but they used a smaller test set (only 128 images).  
> My results are on the full dataset, so direct comparison should consider this difference.


## 🔭 Future Work

- [ ] Integrate with real-time camera feed and depth prediction (MiDaS or self-trained stereo)
- [ ] Improve model precision via data augmentation or simulated fringe patterns
- [ ] Explore transformer-based models for fringe interpretation
- [ ] Apply to **multi-fringe** or **phase-shifting** setups
- [ ] Add GUI for real-time inspection and measurement

---

## 👤 Author

**Jangrae (William) Jo**  
M.S. in Electrical and Computer Engineering @ UMass Amherst  
 
Optical Metrology | Computer Vision | Deep Learning  

---

## 📜 Citation

If you use this repository, please cite the original paper by Huang et al. (2023) and link this repository for credit to my PyTorch training adaptation.


## 🛠️ Full Installation & Execution Guide

### 📦 1. Environment Setup
```bash
# (Optional) Create and activate a virtual environment
conda create -n fringe_env python=3.9 -y
conda activate fringe_env

# Install required dependencies
pip install -r requirements.txt
```

---

### 📁 2. Download the Dataset

Download the SIDO dataset:

- Paper: https://www.sciencedirect.com/science/article/abs/pii/S0030428422003607  
- Dataset: https://figshare.com/articles/dataset/Single-input_dual-output_3D_shape_reconstruction/19709134

Place the `.npy` files in a `data_npy/` directory:

```
project_root/
├── data_npy/
│   ├── f1_80.npy
│   └── Z.npy
```

---

### 🧠 3. Train the Model
```bash
python train.py
```

- The trained model weights will be saved as `UHRNet_weight.pth`.

---

### 📈 4. Run Inference & Visualize Results
```bash
python visualize_prediction.py
```

- This will save predicted images and `.npy` files in the `output_vis/` directory.

---

### 🎞️ 5. Create Video & GIF from Predictions

To generate an MP4 video:
```bash
python make_video_from_images.py
```

To generate a GIF:
```python
from moviepy.editor import ImageSequenceClip

clip = ImageSequenceClip("output_vis", fps=1)
clip.write_gif("output_vis.gif")
```

---

### 📄 Example `requirements.txt`
```text
numpy
torch>=1.12
matplotlib
opencv-python
moviepy
```
