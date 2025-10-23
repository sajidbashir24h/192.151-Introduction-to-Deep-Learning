"""
Utility functions for training neural networks.
"""

import torch
import time
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to evaluate on (cpu or cuda)
    
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_path=None):
    """
    Complete training loop for a model.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        save_path: Optional path to save the best model
    
    Returns:
        Dictionary with training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
    
    return history
