"""
Fine-tuning pre-trained models on custom datasets.
Unfreeze and train some layers with a small learning rate.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import train_model, evaluate


def create_finetuning_model(model_name='resnet18', num_classes=10, unfreeze_layers=2):
    """
    Create a model for fine-tuning.
    
    Args:
        model_name: Name of the pre-trained model
        num_classes: Number of output classes
        unfreeze_layers: Number of layer groups to unfreeze
    
    Returns:
        Model with some layers unfrozen for fine-tuning
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers based on unfreeze_layers
        if unfreeze_layers >= 1:
            for param in model.layer4.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 2:
            for param in model.layer3.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 3:
            for param in model.layer2.parameters():
                param.requires_grad = True
        
        # Replace the final layer (always trainable)
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        num_features = model.fc.in_features
        
        for param in model.parameters():
            param.requires_grad = False
        
        if unfreeze_layers >= 1:
            for param in model.layer4.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 2:
            for param in model.layer3.parameters():
                param.requires_grad = True
        
        model.fc = nn.Linear(num_features, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def get_data_transforms():
    """Get data transforms for training and validation."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def load_cifar10(batch_size=32):
    """Load CIFAR-10 dataset."""
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = datasets.CIFAR10(root='../../data', train=True,
                                     download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='../../data', train=False,
                                    download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    """Main function for fine-tuning."""
    print("Transfer Learning: Fine-Tuning")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    model_name = 'resnet18'
    batch_size = 32
    learning_rate = 0.0001  # Lower learning rate for fine-tuning
    num_epochs = 15
    num_classes = 10
    unfreeze_layers = 2
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    print(f"Creating {model_name} for fine-tuning...")
    print(f"Unfreezing last {unfreeze_layers} layer groups")
    model = create_finetuning_model(model_name, num_classes, unfreeze_layers)
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Percentage trainable: {100 * trainable / total:.2f}%\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate)
    
    # Optional: Use learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    print("Training...")
    history = train_model(model, train_loader, test_loader, criterion,
                         optimizer, num_epochs, device,
                         save_path='fine_tuning_best.pth')
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fine_tuning_results.png')
    print("\nResults saved as 'fine_tuning_results.png'")
    plt.show()


if __name__ == '__main__':
    main()
