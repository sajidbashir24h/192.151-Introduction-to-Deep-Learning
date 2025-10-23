"""
Explore and visualize the MNIST dataset.
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def visualize_samples(dataset, n_samples=25):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: MNIST dataset
        n_samples: Number of samples to display
    """
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.ravel()
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    print("Samples saved as 'mnist_samples.png'")
    plt.show()


def visualize_class_distribution(dataset):
    """
    Visualize the distribution of classes.
    
    Args:
        dataset: MNIST dataset
    """
    labels = [dataset[i][1] for i in range(len(dataset))]
    
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title('MNIST Class Distribution')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    plt.savefig('mnist_distribution.png')
    print("Distribution saved as 'mnist_distribution.png'")
    plt.show()
    
    # Print statistics
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for digit, count in zip(unique, counts):
        print(f"Digit {digit}: {count} samples")


def visualize_digit_variations(dataset, digit=0, n_samples=10):
    """
    Show variations of a specific digit.
    
    Args:
        dataset: MNIST dataset
        digit: Digit to visualize
        n_samples: Number of samples to show
    """
    # Find samples of the specified digit
    indices = [i for i in range(len(dataset)) if dataset[i][1] == digit]
    selected = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(selected):
        img, label = dataset[idx]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.suptitle(f'Variations of Digit {digit}')
    plt.tight_layout()
    plt.savefig(f'mnist_digit_{digit}_variations.png')
    print(f"Digit {digit} variations saved as 'mnist_digit_{digit}_variations.png'")
    plt.show()


def main():
    """Main function to explore MNIST dataset."""
    print("MNIST Dataset Exploration")
    print("=" * 50)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='../../data', train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../../data', train=False, 
                                  download=True, transform=transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get a sample
    img, label = train_dataset[0]
    print(f"\nImage shape: {img.shape}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Label: {label}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_samples(train_dataset, n_samples=25)
    visualize_class_distribution(train_dataset)
    
    # Show variations for a few digits
    for digit in [0, 1, 7]:
        visualize_digit_variations(train_dataset, digit=digit, n_samples=10)
    
    print("\nExploration complete!")


if __name__ == '__main__':
    main()
