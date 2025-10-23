"""
Utility functions for plotting and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss over epochs.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' keys
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy if available
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_predictions(images, true_labels, pred_labels, class_names=None, n=10):
    """
    Plot images with their true and predicted labels.
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: Optional list of class names
        n: Number of images to plot
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(n, len(images))):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        
        if class_names:
            true_label = class_names[true_label]
            pred_label = class_names[pred_label]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
