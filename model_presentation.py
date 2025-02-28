"""
File: model_presentation
Author: antoi
Date: 21/02/2025
Description: 
"""

import torch
import timm
from torchsummary import summary
from torchvision import models

# Load the pretrained Vision Transformer model (ViT)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)


# Summarize and present model details
def summarize_model(model):
    print("Model Summary:")
    print("=========================")

    # Print model architecture
    print(model)

    # Print the number of parameters (trainable and non-trainable)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")

    # Use torchsummary to display the summary of each layer (for an input size of 224x224)
    print("\nLayer-wise summary:")
    summary(model, input_size=(3, 224, 224))


# Summarize the ViT model
summarize_model(model)

checkpoint = {
    'pos_embedding': 'Taille du positionnement des patchs (embedding)',
    'cls_token': 'Token de classification',
    'to_patch_embedding.1.weight': 'Poids de la couche de transformation des patches',
    'to_patch_embedding.1.bias': 'Biais de la couche de transformation des patches',
    'transformer.norm.weight': 'Poids de la normalisation (layer norm)',
    'transformer.layers': 'Composants des blocs de transformer',
    'mlp_head.weight': 'Poids de la tête MLP (Multi-Layer Perceptron)',
    'mlp_head.bias': 'Biais de la tête MLP'
}

# Affichage général des informations
print("Résumé du modèle Vision Transformer (ViT) pour le jeu de données MSTAR :\n")
print("1. Taille d'entrée des patchs : 16x16 pixels (ou une autre valeur selon l'architecture choisie)")
print("2. Le modèle transforme chaque image en une série de patches, puis chaque patch est 'aplatit' pour être transformé par les couches suivantes.")
print("\nComposants du modèle :\n")
for key, description in checkpoint.items():
    print(f"{key}: {description}")

print("\nDétails des blocs Transformer :")
print("Chaque bloc Transformer se compose de :")
print("- Une couche de normalisation de l'entrée")
print("- Un mécanisme d'attention multi-têtes (self-attention)")
print("- Une couche feed-forward (MLP) après chaque mécanisme d'attention")
print("Le nombre de couches Transformer dépend de l'architecture spécifique du ViT (par exemple, ViT-B/16 peut avoir 12 couches).")

print("\nTaille d'entrée et architecture du modèle :")
print("Input image size : (H, W) typiquement 224x224 pour Vision Transformer classique")
print("Dimension des patchs : 16x16 pixels par patch")
print("La dimension d'entrée z_dim (avant transformation) est de 64 (pour un modèle standard).")
print("La sortie du modèle ViT : vecteur de dimension correspondant au nombre de classes (par exemple, pour classification d'images, taille 1000 pour ImageNet).")

