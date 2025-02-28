"""
File: CBAM
Author: antoi
Date: 27/02/2025
Description: 
"""
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torch.utils.data import DataLoader


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

        # Ajouter une couche entièrement connectée pour la classification
        self.fc = nn.Linear(224*224, 10)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)

        # Aplatir la sortie pour l'entrée dans la couche linéaire
        x_out = x_out.view(x_out.size(0), -1)  # [batch_size, gate_channels]

        # Passer la sortie par la couche linéaire pour la classification
        x_out = self.fc(x_out)
        return x_out

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

batch_size = 32

# Load dataset
train_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TRAIN', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TEST', transform=test_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Entraînement du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Nombre de canaux d'entrée du modèle (par exemple, 3 pour des images RGB)
gate_channels = 1  # ou un autre nombre selon vos images

# Ratio de réduction pour le ChannelGate (par exemple, 16)
reduction_ratio = 16

# Types de pooling à utiliser dans ChannelGate
pool_types = ['avg', 'max']  # vous pouvez changer cela si nécessaire

# Créez le modèle CBAM
model = CBAM(gate_channels=gate_channels, reduction_ratio=reduction_ratio, pool_types=pool_types, no_spatial=False).cuda()

# Vérifiez l'architecture du modèle
print(model)


# Fonction de perte (par exemple, CrossEntropyLoss pour une tâche de classification)
criterion = nn.CrossEntropyLoss()

# Optimiseur (par exemple, Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import numpy as np


# Fonction d'entraînement
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Fonction d'évaluation
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_labels, all_preds


# Fonction pour tracer les courbes d'apprentissage
def plot_learning_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, num_epochs):
    plt.figure(figsize=(12, 5))

    # Courbe de la perte
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Courbe de l'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_history, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc_history, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves_cbam.png')
    plt.show()


# Fonction pour tracer la matrice de confusion normalisée
def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(all_labels),
                yticklabels=np.unique(all_labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('confusion_matrix_cbam.png')
    plt.show()


# Fonction principale d'entraînement et d'évaluation
def train_and_evaluate(model, train_loader, val_loader, num_epochs=20, save_path='model_cbam.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Critère de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Variables pour stocker les courbes d'apprentissage
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        start_time = time.time()

        # Entraînement
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        # Évaluation
        val_loss, val_accuracy, all_labels, all_preds = evaluate(model, val_loader, criterion, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        # Temps par époque
        epoch_duration = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%, "
              f"Epoch Duration: {epoch_duration:.2f}s")

    # Enregistrer le modèle
    torch.save(model.state_dict(), save_path)

    # Affichage des courbes d'apprentissage
    plot_learning_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, num_epochs)

    # Affichage et enregistrement de la matrice de confusion normalisée
    plot_confusion_matrix(all_labels, all_preds)

    # Évaluation finale
    final_accuracy = val_acc_history[-1]
    print(f"Final Model Accuracy on Validation Set: {final_accuracy * 100:.2f}%")


# train_and_evaluate(model, train_loader, test_loader, num_epochs=20, save_path='model_cbam.pth')


def count_trainable_params(model):
    # Initialiser le compteur
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return trainable_params

print(f"Number of trainable parameters in the model: {count_trainable_params(model)}")