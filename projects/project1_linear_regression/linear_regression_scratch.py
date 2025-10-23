"""
Linear Regression implemented from scratch using NumPy.
This demonstrates the fundamental concepts of neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    """Linear regression model implemented from scratch."""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize the model.
        
        Args:
            learning_rate: Step size for gradient descent
        """
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.losses = []
    
    def initialize_parameters(self, n_features):
        """
        Initialize weights and bias.
        
        Args:
            n_features: Number of input features
        """
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0.0
    
    def forward(self, X):
        """
        Forward pass: compute predictions.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples, 1)
        """
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute Mean Squared Error loss.
        
        Args:
            y_true: True labels (n_samples, 1)
            y_pred: Predicted labels (n_samples, 1)
        
        Returns:
            MSE loss
        """
        n_samples = len(y_true)
        loss = np.sum((y_true - y_pred) ** 2) / (2 * n_samples)
        return loss
    
    def backward(self, X, y_true, y_pred):
        """
        Backward pass: compute gradients.
        
        Args:
            X: Input features (n_samples, n_features)
            y_true: True labels (n_samples, 1)
            y_pred: Predicted labels (n_samples, 1)
        
        Returns:
            Gradients for weights and bias
        """
        n_samples = len(y_true)
        error = y_pred - y_true
        
        # Compute gradients
        dw = np.dot(X.T, error) / n_samples
        db = np.sum(error) / n_samples
        
        return dw, db
    
    def update_parameters(self, dw, db):
        """
        Update weights and bias using gradient descent.
        
        Args:
            dw: Gradient for weights
            db: Gradient for bias
        """
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples, 1)
            epochs: Number of training iterations
            verbose: Whether to print progress
        """
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass
            dw, db = self.backward(X, y, y_pred)
            
            # Update parameters
            self.update_parameters(dw, db)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples, 1)
        """
        return self.forward(X)


def generate_data(n_samples=100, n_features=1, noise=0.1):
    """
    Generate synthetic data for linear regression.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Standard deviation of Gaussian noise
    
    Returns:
        X, y: Features and labels
    """
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features, 1) * 2
    true_bias = np.random.randn() * 0.5
    y = np.dot(X, true_weights) + true_bias + noise * np.random.randn(n_samples, 1)
    return X, y


def main():
    """Main function to demonstrate linear regression."""
    print("Linear Regression from Scratch")
    print("=" * 50)
    
    # Generate data
    np.random.seed(42)
    X_train, y_train = generate_data(n_samples=100, n_features=1, noise=0.2)
    X_test, y_test = generate_data(n_samples=20, n_features=1, noise=0.2)
    
    # Train model
    model = LinearRegressionScratch(learning_rate=0.01)
    model.fit(X_train, y_train, epochs=1000, verbose=True)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate
    train_loss = model.compute_loss(y_train, y_train_pred)
    test_loss = model.compute_loss(y_test, y_test_pred)
    
    print(f"\nFinal Training Loss: {train_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Learned weights: {model.weights.flatten()}")
    print(f"Learned bias: {model.bias:.4f}")
    
    # Visualizations
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(model.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    
    # Plot 2: Predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
    plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', label='Predictions', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_scratch_results.png')
    print("\nPlot saved as 'linear_regression_scratch_results.png'")
    plt.show()


if __name__ == '__main__':
    main()
