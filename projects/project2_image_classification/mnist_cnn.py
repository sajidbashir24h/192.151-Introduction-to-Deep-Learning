"""
CNN for MNIST digit classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import train_model, evaluate, plot_training_history


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv layers
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ImprovedCNN(nn.Module):
    """Improved CNN with batch normalization."""
    
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 7x7 -> 3x3
        
        x = x.view(-1, 128 * 3 * 3)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_mnist_data(batch_size=64, augment=False):
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for DataLoader
        augment: Whether to apply data augmentation
    
    Returns:
        train_loader, test_loader
    """
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='../../data', train=True, 
                                   download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='../../data', train=False, 
                                  download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def visualize_predictions(model, test_loader, device, n_images=10):
    """Visualize model predictions."""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:n_images], labels[:n_images]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(n_images):
        img = images[i].squeeze().cpu().numpy()
        true_label = labels[i].item()
        pred_label = predictions[i].cpu().item()
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    print("Predictions saved as 'mnist_predictions.png'")
    plt.show()


def main():
    """Main function to train MNIST CNN."""
    print("MNIST CNN Classification")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    use_augmentation = False
    use_improved = False
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size, augment=use_augmentation)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    model = ImprovedCNN() if use_improved else SimpleCNN()
    model = model.to(device)
    print(f"Model: {'Improved' if use_improved else 'Simple'} CNN")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    history = train_model(model, train_loader, test_loader, criterion, 
                         optimizer, num_epochs, device, 
                         save_path='mnist_cnn_best.pth')
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Visualize results
    plot_training_history(history, save_path='training_history.png')
    visualize_predictions(model, test_loader, device)


if __name__ == '__main__':
    main()
