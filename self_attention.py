"""
File: self_attention
Author: antoi
Date: 24/02/2025
Description: 
"""

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import timm
# from tqdm import tqdm
# import matplotlib.pyplot as plt
#
# # Hyperparameters
# batch_size = 32
# num_epochs = 20
# learning_rate = 1e-4
# num_classes = 10  # Assuming 10 SAR image classes in MSTAR
#
# # Function to extract attention maps from the ViT model
# def get_attention_map(model, image, device):
#     # Move image to the appropriate device
#     image = image.unsqueeze(0).to(device)
#
#     # Hook to capture the attention weights
#     attention_maps = []
#
#     def hook_fn(module, input, output):
#         # Assuming attention weights are the first output (check if it's true)
#         attention_maps.append(output[0])
#
#     # Register the hook on the first attention block
#     hook = model.blocks[0].attn.register_forward_hook(hook_fn)
#
#     # Forward pass through the model
#     with torch.no_grad():
#         _ = model(image)
#
#     # Remove the hook
#     hook.remove()
#
#     # Extract the attention map from the hook
#     attention_map = attention_maps[0].cpu().numpy()
#
#     # Average the attention across all heads
#     attention_mean = attention_map.mean(axis=1)
#
#     return attention_mean
#
# # Visualize the attention map
# def plot_attention_map(attention_map, image, idx=0):
#     # print(attention_map.shape)
#     # # Normalize attention map
#     # attention_map = attention_map[idx]
#     # attention_map = attention_map.reshape(14, 14)  # The ViT patch resolution (14x14 for 224x224 input)
#     print(f"Attention map shape before reshape: {attention_map.shape}")
#
#     # Remove the class token (first element)
#     attention_map = attention_map[1:]
#
#     # Ensure the attention map is of the expected shape
#     if attention_map.size != 14 * 14:
#         raise ValueError(f"Unexpected attention map size: {attention_map.size}, expected 14x14.")
#
#     # Now reshape the attention map to 14x14
#     attention_map = attention_map.reshape(14, 14)
#
#     attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
#
#     # Resize to match the image size (optional, for overlaying)
#     attention_map = torch.nn.functional.interpolate(
#         torch.tensor(attention_map).unsqueeze(0).unsqueeze(0),
#         size=(224, 224),
#         mode="bilinear",
#         align_corners=False
#     ).squeeze().numpy()
#
#     # Plot the attention map
#     plt.figure(figsize=(6, 6))
#     plt.imshow(image.permute(1, 2, 0).cpu(), cmap='gray')  # Adjust for grayscale image
#     plt.imshow(attention_map, cmap='jet', alpha=0.6)  # Superimpose attention map
#     plt.axis('off')
#     plt.show()
#
# # Load an example image from the test_loader to visualize
# def visualize_attention(model, test_loader, device):
#     # Get one batch from the test_loader
#     data_iter = iter(test_loader)
#     images, labels = next(data_iter)
#
#     # Choose one image from the batch
#     image = images[0]
#
#     # Get attention map from the model
#     attention_map = get_attention_map(model, image, device)
#
#     # Plot the attention map
#     plot_attention_map(attention_map, image)
#
# # Assuming the model is already fine-tuned and loaded
# # Uncomment the following line to load the model if needed:
# # model.load_state_dict(torch.load('vit_mstar_finetuned_head_only.pth'))
#
# # Model setup: Load pretrained Vision Transformer (ViT) and modify head
# model = timm.create_model('vit_base_patch16_224', pretrained=True)
#
# # Modify the classification head for 10 classes
# model.head = nn.Linear(model.head.in_features, 10)
#
# print(model)
#
# checkpoint_path = 'vit_mstar_finetuned.pth'  # Remplacez par votre chemin
# checkpoint = torch.load(checkpoint_path)
#
# # Check if the keys match
# model_dict = model.state_dict()
#
# # Filter out unnecessary keys (if any) and update the model's state dict
# filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
#
# # Load the state dictionary
# model_dict.update(filtered_checkpoint)
# model.load_state_dict(model_dict)
#
# model.eval()
#
# # Move model to GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# model.to(device)
#
# # Data preparation: Define transformations for train and test sets
# train_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale images should use single mean/std
# ])
#
# test_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjust for grayscale
# ])
#
# # Load MSTAR dataset using ImageFolder
# train_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TRAIN', transform=train_transforms)
# test_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TEST', transform=test_transforms)
#
# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # Visualize the attention map for one image
# visualize_attention(model, test_loader, device)




import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
num_classes = 10  # Assuming 10 SAR image classes in MSTAR

# Function to extract attention maps from multiple layers of the ViT model
def get_attention_maps(model, image, device):
    # Move image to the appropriate device
    image = image.unsqueeze(0).to(device)

    # Hook to capture the attention weights
    attention_maps = {}

    def hook_fn(layer_num):
        def hook(module, input, output):
            # Assuming attention weights are the first output (check if it's true)
            attention_maps[layer_num] = output[0].detach().cpu().numpy()
        return hook

    # Register hooks for each transformer block
    hooks = []
    for i in range(len(model.blocks)):
        hooks.append(model.blocks[i].attn.register_forward_hook(hook_fn(i)))

    # Forward pass through the model
    with torch.no_grad():
        _ = model(image)

    # Remove all hooks
    for hook in hooks:
        hook.remove()

    return attention_maps

# Visualize attention maps from multiple layers
def plot_attention_mosaic(attention_maps, image):
    num_layers = len(attention_maps)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # 4x4 grid for up to 16 layers

    for i, (layer_num, attention_map) in enumerate(attention_maps.items()):
        # Average the attention across all heads
        attention_mean = attention_map.mean(axis=1)

        # Remove the class token (first element)
        attention_mean = attention_mean[1:]

        # Reshape to 14x14 (assuming 224x224 input resolution with 16x16 patches)
        attention_mean = attention_mean.reshape(14, 14)

        # Normalize the attention map
        attention_mean = (attention_mean - attention_mean.min()) / (attention_mean.max() - attention_mean.min())

        # Resize for better visualization (optional)
        attention_mean_resized = torch.nn.functional.interpolate(
            torch.tensor(attention_mean).unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        # Plot attention map in the corresponding subplot
        ax = axes[i // 4, i % 4]
        ax.imshow(image.permute(1, 2, 0).cpu(), cmap='gray')  # Adjust for grayscale image
        ax.imshow(attention_mean_resized, cmap='jet', alpha=0.6)  # Superimpose attention map
        ax.set_title(f'Layer {layer_num + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Load an example image from the test_loader to visualize
def visualize_attention_mosaic(model, test_loader, device):
    # Get one batch from the test_loader
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Choose one image from the batch
    image = images[0]

    # Get attention maps from multiple layers
    attention_maps = get_attention_maps(model, image, device)

    # Plot the attention maps in a mosaic
    plot_attention_mosaic(attention_maps, image)

# Model setup: Load pretrained Vision Transformer (ViT) and modify head
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Modify the classification head for 10 classes
model.head = nn.Linear(model.head.in_features, 10)

# Load the model checkpoint
checkpoint_path = 'vit_mstar_finetuned.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)

model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Data preparation: Define transformations for train and test sets
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale images should use single mean/std
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjust for grayscale
])

# Load MSTAR dataset using ImageFolder
train_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TRAIN', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='C:/APPLIS/projets/ViTProject/mstar/TEST', transform=test_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualize the attention map mosaic for one image
visualize_attention_mosaic(model, test_loader, device)
