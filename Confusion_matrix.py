"""
File: Confusion_matrix
Author: antoi
Date: 17/02/2025
Description: 
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.datasets import ImageFolder
import timm

# Assurez-vous que le modèle Vision Transformer (ViT) est défini et importé ici
# Par exemple, si c'est un ViT pré-implémenté, vous pouvez utiliser timm pour l'importer
# from timm import create_model
# vit_model = create_model('vit_base_patch16_224', pretrained=False, num_classes=10)


# Model setup: Load pretrained Vision Transformer (ViT) and modify head
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Modify the classification head for 10 classes
model.head = nn.Linear(model.head.in_features, 10)


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



checkpoint_path = 'CNN1.pth'  # Remplacez par votre chemin
# checkpoint = torch.load(checkpoint_path)
# print(checkpoint.keys())
# model.load_state_dict(checkpoint['model_state_dict'])  # Remplacez 'model_state_dict' par la clé utilisée dans votre checkpoint

checkpoint = torch.load(checkpoint_path)

# Check if the keys match
model_dict = model.state_dict()

# Filter out unnecessary keys (if any) and update the model's state dict
filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

# Load the state dictionary
model_dict.update(filtered_checkpoint)
model.load_state_dict(model_dict)

model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# 2. Transformer pour les images MSTAR
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustez en fonction de l'input de votre ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Ajustez les valeurs de normalisation
])

# 3. Charger le dataset MSTAR pour l'évaluation
test_dataset = ImageFolder(root='C:\APPLIS\projets\ViTProject\mstar\TEST', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Évaluer le modèle
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()  # Envoyer les données au GPU si disponible
        labels = labels.cuda()
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# # # 5. Calculer la matrice de confusion
# cm = confusion_matrix(all_labels, all_preds)
# #
# # # 6. Afficher la matrice de confusion
# # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
# # disp.plot(cmap=plt.cm.Blues)
# # plt.title('Confusion Matrix for ViT on MSTAR')
# # plt.show()
#
# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Normalize the confusion matrix to probabilities
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=range(10))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap='Blues', ax=ax, values_format='.2f')

# Add labels and title
plt.title('Normalized Confusion Matrix (Probabilities)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')



# Save the confusion matrix as an image
plt.savefig('confusion_matrix_cnn1.png', dpi=300)

# Show the plot
plt.show()
