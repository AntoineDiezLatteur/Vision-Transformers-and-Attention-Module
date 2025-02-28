"""
File: CNN_spatial
Author: antoi
Date: 25/02/2025
Description: 
"""

import torch.nn.functional as F
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torch.utils.data import DataLoader


# Hyperparameters
batch_size = 32
num_epochs = 15
learning_rate = 1e-4
num_classes = 10  # Assuming 10 SAR image classes in MSTAR


# Définition du module d'attention spatiale
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_attention=False):
        # Moyenne et max pooling sur l'axe des canaux
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concaténation des cartes de pooling
        concatenated = torch.cat([avg_out, max_out], dim=1)

        # Convolution 3x3 suivie d'une activation sigmoïde
        attention_map = self.sigmoid(self.conv(concatenated))

        # Appliquer la carte d'attention à la feature map
        if return_attention:
            return x * attention_map, attention_map  # Return attention map for visualization
        return x * attention_map

# Définition du CNN avec module d'attention spatiale
class CNNWithSpatialAttention(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNWithSpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.spatial_attention = SpatialAttention()

        # Corriger la normalisation par lots pour 64 canaux après la deuxième convolution
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, return_attention=False):
        attention_maps = []

        # Bloc 1: Convolution, attention, pooling
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        if return_attention:
            x, attention_map1 = self.spatial_attention(x, return_attention=True)
            attention_maps.append(attention_map1)
        else:
            x = self.spatial_attention(x)
        x = self.batch_norm1(x)  # Correction ici pour 64 canaux

        # Bloc 2: Convolution, attention, pooling
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        if return_attention:
            x, attention_map2 = self.spatial_attention(x, return_attention=True)
            attention_maps.append(attention_map2)
        else:
            x = self.spatial_attention(x)
        x = self.batch_norm2(x)  # Toujours 128 canaux ici

        # Flatten avant les couches fully connected
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if return_attention:
            return x, attention_maps
        return x


# Définition des transformations pour le train et le test
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en 1 canal
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Changement pour 1 canal
])

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en 1 canal
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Changement pour 1 canal
])


# Load dataset
train_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TRAIN', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TEST', transform=test_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Entraînement du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialisation du modèle
model = CNNWithSpatialAttention(num_classes=10).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss and accuracy history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Validation loop
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.concatenate(all_preds), np.concatenate(all_labels)

# Main loop: Train and validate the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train and validate
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, all_preds, all_labels = validate(model, test_loader, criterion, device)

    # Store the losses and accuracies
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

# Save the fine-tuned model
torch.save(model.state_dict(), 'CNN_spatial_model.pth')

# Plot learning curves
def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))

    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves_cnn_spatial.png')
    plt.show()

# Plot and save learning curves
plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# Plot and save confusion matrix
def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.savefig('confusion_matrix_cnn_spatial.png')
    plt.show()

# Plot and save confusion matrix
plot_confusion_matrix(all_labels, all_preds)

# Visualiser les cartes d'attention spatiale
def visualize_attention(model, loader, device, num_images=5):
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Pass through the model and retrieve attention maps
            outputs, attention_maps = model(images, return_attention=True)

            # For each image in the batch, plot the attention maps
            for i in range(min(num_images, images.size(0))):
                fig, axs = plt.subplots(1, len(attention_maps) + 1, figsize=(12, 4))

                # Original image
                axs[0].imshow(images[i].cpu().squeeze(), cmap='gray')
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                # Attention maps
                for j, attention_map in enumerate(attention_maps):
                    axs[j + 1].imshow(attention_map[i].cpu().squeeze(), cmap='jet')
                    axs[j + 1].set_title(f'Attention Map {j+1}')
                    axs[j + 1].axis('off')

                plt.colorbar()

                plt.tight_layout()
                plt.show()

                images_shown += 1
                if images_shown >= num_images:
                    return

# Visualize attention maps for a few test images
visualize_attention(model, test_loader, device, num_images=5)
