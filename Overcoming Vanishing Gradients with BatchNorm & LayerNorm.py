# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:39:59 2025

@author: Manisha
"""
# =============================================================================
#  Practice Tasks (PyTorch-Oriented)
# 
# Task 1:
# Modify your previous CIFAR-10 CNN to include BN after every Conv2D layer.
# Compare accuracy and convergence speed vs. the old model.
# 
# Task 2:
# Plot the distribution of activations (before vs. after BN) using histograms— 
# see how BN keeps them centered around 0.
# 
# Task 3:
# Try smaller batch sizes (8, 16, 32) and see how BN behaves.
# 
# Task 4:
# Replace BN with LayerNorm and compare.
# (Hint: nn.LayerNorm([C, H, W]))
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

train_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
    )

test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
    )

trainloader = torch.utils.data.DataLoader(train_set, 
                                          batch_size=64, shuffle=True)

trainloader_small = torch.utils.data.DataLoader(train_set, 
                                          batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device Used: ", device)


# CNN Models
class CNN_NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class CNN_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Replace BatchNorm with LayerNorm
class CNN_LN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.ln = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.ln(self.fc1(x)))
        x = self.fc2(x)
        return x
    
# Training Function 
def train_model(model, epochs=5, lr=0.01):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        current, total = 0, 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = output.max(1)
            total += labels.size(0)
            current += preds.eq(labels).sum().item()
        
        print(f"Epoch: [{epoch+1}/{epochs}] | "
              f"loss: {running_loss/len(trainloader):.3f}"
              f"| accuracy: {100 * current/total:.3f}")

# Training Function for small batches
def train_model_small(model, epochs=5, lr=0.01):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        current, total = 0, 0
        
        for images, labels in trainloader_small:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = output.max(1)
            total += labels.size(0)
            current += preds.eq(labels).sum().item()
        
        print(f"Epoch: [{epoch+1}/{epochs}] | "
              f"loss: {running_loss/len(trainloader_small):.3f}"
              f"| accuracy: {100 * current/total:.3f}")



def train_model_with_grad_tracking(model, epochs=5, lr=0.01, model_name='Model'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    grad_norms = []
    epoch_losses = []
    epoch_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        current, total = 0, 0
        total_grad_norm = 0
        steps = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            
            # compute total gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_grad = p.grad.data.norm(2)
                    total_norm += param_grad.item() ** 2
            total_norm = total_norm ** 0.5
            total_grad_norm += total_norm
            steps += 1
            
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = output.max(1)
            total += labels.size(0)
            current += preds.eq(labels).sum().item()
            
        avg_grad_norm = total_grad_norm / steps
        grad_norms.append(avg_grad_norm)
        epoch_losses.append(running_loss / len(trainloader))
        epoch_accs.append(100 * current / total)
        
        print(f"{model_name} | Epoch [{epoch+1}/{epochs}] "
              f"Loss: {epoch_losses[-1]:.3f} | Acc: {epoch_accs[-1]:.2f}% "
              f"| GradNorm: {avg_grad_norm:.3f}")
    
    return grad_norms, epoch_losses, epoch_accs

# train all models and track gradients
print("\n Gradient Norm Experiment....")
cnn_no_bn = CNN_NoBN()
grad_no_bn, loss_no_bn, acc_no_bn = train_model_with_grad_tracking(
    cnn_no_bn, model_name="CNN_NoBN")

cnn_bn = CNN_BN()
grad_bn, loss_bn, acc_bn = train_model_with_grad_tracking(
    cnn_bn, model_name="CNN_BN")

cnn_ln = CNN_LN()
grad_ln, loss_ln, acc_ln = train_model_with_grad_tracking(
    cnn_ln, model_name="CNN_LN")
        

# Lets run model
print("Training model without batch Norm....")
train_model(cnn_no_bn)
print("Training model without batch norm for small batches...")
train_model_small(cnn_no_bn)

print("Training model with Batch Norm.....")
train_model(cnn_bn)
print("Training model with batch norm for small batches...")
train_model_small(cnn_bn)

print("\nTraining model with LayerNorm...")
model_ln = CNN_LN()
train_model(model_ln)
print("Training model with LN (small batches)...")
train_model_small(model_ln)

# Visualise activation distribution

# forward pass with one batch
images, _ = next(iter(trainloader))
images = images.to(device)
with torch.no_grad():
    out_no_bn = cnn_no_bn.conv1(images)
    out_bn = cnn_bn.bn1(cnn_bn.conv1(images))
    
# Flatten for histogram
out_no_bn = out_no_bn.view(-1).cpu().numpy()
out_bn = out_bn.view(-1).cpu().numpy()

plt.figure(figsize=(10, 5))
plt.hist(out_no_bn, bins=50, alpha=0.6, label="Without BN")
plt.hist(out_bn, bins=50, alpha=0.6, label='With BN')
plt.title("Distributions of activations (before vs after BN)")
plt.legend()
plt.xlabel("Activation value")
plt.ylabel("Frequency")
plt.show()

# Histogram for small batches

images, _ = next(iter(trainloader_small))
images = images.to(device)
with torch.no_grad():
    out_no_bn_small = cnn_no_bn.conv1(images)
    out_bn_small = cnn_bn.bn1(cnn_bn.conv1(images))
    
# Flatten for histogram
out_no_bn_small = out_no_bn_small.view(-1).cpu().numpy()
out_bn_small = out_bn_small.view(-1).cpu().numpy()

plt.figure(figsize=(10, 5))
plt.hist(out_no_bn_small, bins=50, alpha=0.6, label="Without BN")
plt.hist(out_bn_small, bins=50, alpha=0.6, label='With BN')
plt.title("Distributions of activations (before vs after BN) for small batches")
plt.legend()
plt.xlabel("Activation value")
plt.ylabel("Frequency")
plt.show()

# Histogram for Layer Normalization
images, _ = next(iter(trainloader_small))
images = images.to(device)
with torch.no_grad():
    out_ln = model_ln.conv1(images)

# Flatten the images
out_ln = out_ln.view(-1).cpu().numpy()

plt.figure(figsize=(10, 6))
plt.hist(out_ln, bins=50, alpha=0.6, label='Layer Norm')
plt.title("Distributionof activation by layer norm")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Gradient Norm Comparison
plt.figure(figsize=(10, 6))
plt.plot(grad_no_bn, label="No BatchNorm")
plt.plot(grad_bn, label="With BatchNorm")
plt.plot(grad_ln, label="With LayerNorm")
plt.title("Average Gradient Norms per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm (L2)")
plt.legend()
plt.grid(True)
plt.show()

# Loss Comparison
plt.figure(figsize=(10, 6))
plt.plot(loss_no_bn, label="No BatchNorm")
plt.plot(loss_bn, label="With BatchNorm")
plt.plot(loss_ln, label="With LayerNorm")
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
    