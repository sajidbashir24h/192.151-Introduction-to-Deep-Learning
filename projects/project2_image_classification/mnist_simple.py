"""
Simple baseline model for MNIST using fully connected layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import train_model, evaluate


class SimpleNN(nn.Module):
    """Simple feedforward neural network for MNIST."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='../../data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../../data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def main():
    """Main function to train simple MNIST model."""
    print("Simple Neural Network for MNIST")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    model = SimpleNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    history = train_model(model, train_loader, test_loader, criterion,
                         optimizer, num_epochs, device,
                         save_path='mnist_simple_best.pth')
    
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
    plt.savefig('mnist_simple_results.png')
    print("\nResults saved as 'mnist_simple_results.png'")
    plt.show()


if __name__ == '__main__':
    main()
