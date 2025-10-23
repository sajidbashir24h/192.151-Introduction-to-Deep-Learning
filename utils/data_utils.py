"""
Utility functions for data loading and preprocessing.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


def create_dataloaders(dataset, batch_size=32, train_split=0.8, shuffle=True, num_workers=0):
    """
    Split dataset into train and validation loaders.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    """
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


class SimpleDataset(Dataset):
    """
    Simple custom dataset for regression or classification tasks.
    """
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: Input features (numpy array or tensor)
            y: Target labels (numpy array or tensor)
            transform: Optional transform to apply to samples
        """
        self.X = torch.FloatTensor(X) if isinstance(X, np.ndarray) else X
        self.y = torch.FloatTensor(y) if isinstance(y, np.ndarray) else y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        target = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target


def normalize_data(X_train, X_test):
    """
    Normalize data using training set statistics.
    
    Args:
        X_train: Training data
        X_test: Test data
    
    Returns:
        Normalized X_train, X_test, mean, std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # Add small value to avoid division by zero
    
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    
    return X_train_normalized, X_test_normalized, mean, std
