"""
Time series prediction using RNN and LSTM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class SimpleRNN(nn.Module):
    """Simple RNN for time series prediction."""
    
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    """LSTM for time series prediction."""
    
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def generate_sine_wave(n_samples=1000, seq_length=50):
    """
    Generate sine wave data for time series prediction.
    
    Args:
        n_samples: Total number of time steps
        seq_length: Length of input sequences
    
    Returns:
        X, y: Input sequences and targets
    """
    # Generate sine wave
    t = np.linspace(0, 100, n_samples)
    data = np.sin(t) + 0.1 * np.random.randn(n_samples)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y).reshape(-1, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)


def train_model(model, X_train, y_train, epochs=100, learning_rate=0.01):
    """
    Train the RNN model.
    
    Args:
        model: PyTorch model
        X_train: Training sequences
        y_train: Training targets
        epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        List of losses
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses


def predict_future(model, initial_sequence, n_steps=50):
    """
    Predict future values given an initial sequence.
    
    Args:
        model: Trained model
        initial_sequence: Initial input sequence
        n_steps: Number of steps to predict
    
    Returns:
        Array of predictions
    """
    model.eval()
    predictions = []
    current_seq = initial_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Predict next value
            pred = model(current_seq)
            predictions.append(pred.item())
            
            # Update sequence: remove first element, add prediction
            current_seq = torch.cat([current_seq[:, 1:, :], 
                                    pred.unsqueeze(1)], dim=1)
    
    return predictions


def main():
    """Main function for time series prediction."""
    print("Time Series Prediction with RNN/LSTM")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    seq_length = 50
    X, y = generate_sine_wave(n_samples=1000, seq_length=seq_length)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Train Simple RNN
    print("Training Simple RNN...")
    rnn_model = SimpleRNN(input_size=1, hidden_size=32, num_layers=1)
    rnn_losses = train_model(rnn_model, X_train, y_train, epochs=100)
    
    # Train LSTM
    print("\nTraining LSTM...")
    lstm_model = LSTMModel(input_size=1, hidden_size=32, num_layers=2)
    lstm_losses = train_model(lstm_model, X_train, y_train, epochs=100)
    
    # Evaluate
    rnn_model.eval()
    lstm_model.eval()
    
    with torch.no_grad():
        rnn_pred = rnn_model(X_test)
        lstm_pred = lstm_model(X_test)
        
        rnn_mse = nn.MSELoss()(rnn_pred, y_test).item()
        lstm_mse = nn.MSELoss()(lstm_pred, y_test).item()
    
    print(f"\nTest MSE - Simple RNN: {rnn_mse:.6f}")
    print(f"Test MSE - LSTM: {lstm_mse:.6f}")
    
    # Future prediction
    initial_seq = X_test[0:1]
    rnn_future = predict_future(rnn_model, initial_seq, n_steps=100)
    lstm_future = predict_future(lstm_model, initial_seq, n_steps=100)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training losses
    axes[0, 0].plot(rnn_losses, label='Simple RNN')
    axes[0, 0].plot(lstm_losses, label='LSTM')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Test predictions
    axes[0, 1].plot(y_test[:100].numpy(), label='True', alpha=0.7)
    axes[0, 1].plot(rnn_pred[:100].detach().numpy(), label='RNN', alpha=0.7)
    axes[0, 1].plot(lstm_pred[:100].detach().numpy(), label='LSTM', alpha=0.7)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Test Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Future predictions
    initial_data = initial_seq.squeeze().numpy()
    axes[1, 0].plot(range(len(initial_data)), initial_data, 'k-', label='Initial', linewidth=2)
    axes[1, 0].plot(range(len(initial_data), len(initial_data) + len(rnn_future)), 
                   rnn_future, label='RNN Future', alpha=0.7)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Simple RNN Future Prediction')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(range(len(initial_data)), initial_data, 'k-', label='Initial', linewidth=2)
    axes[1, 1].plot(range(len(initial_data), len(initial_data) + len(lstm_future)), 
                   lstm_future, label='LSTM Future', alpha=0.7)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('LSTM Future Prediction')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('time_series_results.png')
    print("\nResults saved as 'time_series_results.png'")
    plt.show()


if __name__ == '__main__':
    main()
