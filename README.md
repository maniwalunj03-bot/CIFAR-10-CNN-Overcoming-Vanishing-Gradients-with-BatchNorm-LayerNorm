# CIFAR-10-CNN-Overcoming-Vanishing-Gradients-with-BatchNorm-LayerNorm
Comparative study of CNN performance on the CIFAR-10 dataset using Batch Normalization, Layer Normalization, and no normalization — analyzing their effects on training stability, gradient flow, and accuracy.
# 🧠 CIFAR-10 CNN: Overcoming Vanishing Gradients with BatchNorm & LayerNorm

This project explores **how normalization layers (BatchNorm & LayerNorm)** help stabilize training and prevent **vanishing/exploding gradients** in Convolutional Neural Networks (CNNs), using the **CIFAR-10 dataset**.

---

## 🚀 Project Overview

Normalization is a key concept in deep learning that ensures better gradient flow, faster convergence, and improved generalization.  
This project compares **three CNN variants**:

1. 🟦 **No Normalization**
2. 🟧 **With Batch Normalization**
3. 🟩 **With Layer Normalization**

We analyze how each architecture affects:
- Training **loss and accuracy**
- **Gradient norms** across epochs
- **Activation distributions** before and after normalization

---

## 🧰 Tech Stack

- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** torchvision, matplotlib, numpy  
- **Dataset:** CIFAR-10  

---

## 📊 Key Results

| Model Variant | Final Accuracy | Key Observation |
|----------------|----------------|------------------|
| No Normalization | ~84% | Slower convergence, higher loss |
| Batch Normalization | ~99% | Stable gradients, faster convergence |
| Layer Normalization | ~91% | Strong small-batch performance |

---

### 🔍 Training Metrics

#### 📉 Training Loss vs Epoch
BatchNorm and LayerNorm both show **faster convergence** than the baseline model.

#### 📈 Gradient Norms per Epoch
Normalization maintains **consistent gradient magnitudes**, preventing vanishing gradients.

#### 🧩 Activation Distributions
- **BatchNorm:** Normalized activations across batches  
- **LayerNorm:** More stable activations with smaller batch sizes

---

## 📁 Folder Structure

📦 cifar10-vanishing-gradients
┣ 📜 main.py # Main training script
┣ 📜 utils.py # Helper functions (training, plotting)
┣ 📜 models.py # CNN models (NoNorm, BatchNorm, LayerNorm)
┣ 📜 results/ # Saved plots and results
┗ 📜 README.md

---

## 🧪 Example Output

### Gradient Norm vs Epoch
![Gradient Norms](results/grad_norms.png)

### Training Loss
![Training Loss](results/loss_curve.png)

### Activation Distributions
![Activation Distributions](results/activation_distribution.png)

---

## 🧠 Insights

- **BatchNorm** accelerates learning and helps networks reach higher accuracy with stable gradients.  
- **LayerNorm** generalizes better on **small batch sizes**.  
- Both methods reduce the risk of **vanishing gradients**, a common issue in deep CNNs.

---

## ⚙️ How to Run

```bash
# Clone the repo
git clone https://github.com/maniwalunj03-bot/cifar10-vanishing-gradients.git
cd cifar10-vanishing-gradients

# Install dependencies
pip install torch torchvision matplotlib numpy

# Run training
python main.py

🏁 Future Work

Explore Group Normalization and Instance Normalization

Apply to larger datasets (ImageNet) or ResNet architectures

Investigate gradient clipping and weight initialization effects

---
⭐ If you like this project, give it a star on GitHub!
