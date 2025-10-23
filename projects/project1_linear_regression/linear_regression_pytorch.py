"""
Linear Regression using PyTorch.
This demonstrates how to use PyTorch's built-in modules for model building.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionModel(nn.Module):
    """Linear regression model using PyTorch."""
    
    def __init__(self, input_dim):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Output predictions
        """
        return self.linear(x)


def generate_data(n_samples=100, n_features=1, noise=0.1):
    """
    Generate synthetic data for linear regression.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Standard deviation of Gaussian noise
    
    Returns:
        X, y as PyTorch tensors
    """
    X = torch.randn(n_samples, n_features)
    true_weights = torch.randn(n_features, 1) * 2
    true_bias = torch.randn(1) * 0.5
    y = torch.mm(X, true_weights) + true_bias + noise * torch.randn(n_samples, 1)
    return X, y


def train_model(model, X_train, y_train, epochs=1000, learning_rate=0.01):
    """
    Train the model using PyTorch.
    
    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training labels
        epochs: Number of training iterations
        learning_rate: Learning rate for optimizer
    
    Returns:
        List of losses
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return losses


def main():
    """Main function to demonstrate PyTorch linear regression."""
    print("Linear Regression with PyTorch")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    X_train, y_train = generate_data(n_samples=100, n_features=1, noise=0.2)
    X_test, y_test = generate_data(n_samples=20, n_features=1, noise=0.2)
    
    # Create model
    model = LinearRegressionModel(input_dim=1)
    print(f"\nInitial parameters:")
    print(f"Weight: {model.linear.weight.data.item():.4f}")
    print(f"Bias: {model.linear.bias.data.item():.4f}")
    
    # Train model
    print("\nTraining...")
    losses = train_model(model, X_train, y_train, epochs=1000, learning_rate=0.01)
    
    # Make predictions
    with torch.no_grad():
        y_train_pred = model(X_train)
        y_test_pred = model(X_test)
    
    # Evaluate
    criterion = nn.MSELoss()
    train_loss = criterion(y_train_pred, y_train).item()
    test_loss = criterion(y_test_pred, y_test).item()
    
    print(f"\nFinal Training Loss: {train_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"\nLearned parameters:")
    print(f"Weight: {model.linear.weight.data.item():.4f}")
    print(f"Bias: {model.linear.bias.data.item():.4f}")
    
    # Visualizations
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    
    # Plot 2: Predictions vs actual
    plt.subplot(1, 2, 2)
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()
    
    plt.scatter(X_train_np, y_train_np, alpha=0.5, label='Training Data')
    plt.scatter(X_test_np, y_test_np, alpha=0.5, label='Test Data')
    
    X_line = torch.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    with torch.no_grad():
        y_line = model(X_line).numpy()
    
    plt.plot(X_line.numpy(), y_line, 'r-', label='Predictions', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_pytorch_results.png')
    print("\nPlot saved as 'linear_regression_pytorch_results.png'")
    plt.show()


if __name__ == '__main__':
    main()
