"""
File: spatial_attention
Author: antoi
Date: 25/02/2025
Description: 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
num_classes = 10  # Assuming 10 SAR image classes in MSTAR

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en 1 canal
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Changement pour 1 canal
])

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en 1 canal
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Changement pour 1 canal
])




# Load dataset
train_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TRAIN', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TEST', transform=test_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),  # Grayscale images (1 channel)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        # Taille finale de la feature map : 128 canaux, et 12x12 après pooling
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 4096),  # Ajustement basé sur la taille finale des feature maps
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Instantiate model
model = CustomCNN(num_classes=num_classes)

# Load the model checkpoint
checkpoint_path = 'CNN1.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)

model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



# Visualiser les activations des feature maps
def visualize_feature_maps(model, image, layer_indices):
    model.eval()
    activations = []

    x = image.unsqueeze(0).to(device)  # Ajouter une dimension batch

    # Extraire les activations des couches spécifiées
    for idx, layer in enumerate(model.conv_layers):
        x = layer(x)
        if idx in layer_indices:  # Récupérer les activations à ces indices
            activations.append(x.squeeze(0).cpu().detach().numpy())

    return activations


# Fonction pour afficher une mosaïque d'images de feature maps
def plot_feature_maps(activations, num_cols=8):
    for i, activation in enumerate(activations):
        num_filters = activation.shape[0]
        num_rows = num_filters // num_cols + 1

        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Feature maps après la couche {i + 1}")

        for j in range(num_filters):
            plt.subplot(num_rows, num_cols, j + 1)
            plt.imshow(activation[j], cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# Charge une image de l'ensemble de test
test_image, _ = next(iter(test_loader))
test_image = test_image[0]  # Utiliser la première image du batch

# Visualiser les activations pour les couches convolutives à ces indices (0, 2, 4, 6)
layer_indices = [0, 2, 4, 6]  # Correspondent aux couches convolutives et ReLU
activations = visualize_feature_maps(model, test_image, layer_indices)

# Afficher les feature maps sous forme de mosaïque
plot_feature_maps(activations, num_cols=8)
