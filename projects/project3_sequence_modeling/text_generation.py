"""
Character-level text generation using RNN/LSTM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class CharRNN(nn.Module):
    """Character-level RNN for text generation."""
    
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, hidden_size)
        
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        
        out = self.fc(out)  # (batch, seq_len, vocab_size)
        return out, hidden


class TextDataset:
    """Dataset for character-level text generation."""
    
    def __init__(self, text, seq_length=100):
        self.text = text
        self.seq_length = seq_length
        
        # Create character vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def get_batch(self, batch_size=32):
        """Get a random batch of sequences."""
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            # Random starting position
            start_idx = random.randint(0, len(self.text) - self.seq_length - 1)
            end_idx = start_idx + self.seq_length
            
            # Get sequence and target (shifted by 1)
            seq = self.text[start_idx:end_idx]
            target = self.text[start_idx+1:end_idx+1]
            
            # Convert to indices
            seq_idx = [self.char_to_idx[ch] for ch in seq]
            target_idx = [self.char_to_idx[ch] for ch in target]
            
            inputs.append(seq_idx)
            targets.append(target_idx)
        
        return torch.LongTensor(inputs), torch.LongTensor(targets)


def generate_text(model, dataset, start_str='The ', length=200, temperature=1.0):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained model
        dataset: TextDataset with vocabulary
        start_str: Starting string
        length: Length of generated text
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated text
    """
    model.eval()
    
    # Convert start string to indices
    chars = [dataset.char_to_idx[ch] for ch in start_str]
    input_seq = torch.LongTensor([chars])
    
    generated = start_str
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            # Forward pass
            output, hidden = model(input_seq, hidden)
            
            # Get the last output
            output = output[0, -1, :] / temperature
            probs = torch.softmax(output, dim=0).cpu().numpy()
            
            # Sample from the distribution
            char_idx = np.random.choice(len(probs), p=probs)
            char = dataset.idx_to_char[char_idx]
            
            generated += char
            
            # Update input sequence
            input_seq = torch.LongTensor([[char_idx]])
    
    return generated


def train_model(model, dataset, epochs=50, batch_size=32, learning_rate=0.002):
    """
    Train the character-level RNN.
    
    Args:
        model: RNN model
        dataset: TextDataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        List of losses
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 50  # Number of batches per epoch
        
        for _ in range(num_batches):
            inputs, targets = dataset.get_batch(batch_size)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(inputs)
            
            # Reshape for loss calculation
            output = output.view(-1, dataset.vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(output, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            # Generate sample text
            sample = generate_text(model, dataset, start_str='The ', 
                                  length=100, temperature=0.8)
            print(f'Sample: {sample}\n')
    
    return losses


def main():
    """Main function for text generation."""
    print("Character-Level Text Generation with RNN")
    print("=" * 50)
    
    # Sample text (you can replace this with your own text)
    sample_text = """
    Deep learning is a subset of machine learning that uses neural networks
    with multiple layers to learn hierarchical representations of data.
    These networks can automatically discover features from raw data without
    manual feature engineering. The success of deep learning has revolutionized
    many fields including computer vision, natural language processing,
    and speech recognition. Convolutional neural networks excel at image tasks,
    while recurrent neural networks are powerful for sequential data.
    Transfer learning allows us to leverage pre-trained models for new tasks.
    """
    
    # Create dataset
    seq_length = 100
    dataset = TextDataset(sample_text, seq_length=seq_length)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Text length: {len(sample_text)}")
    print(f"Unique characters: {dataset.chars[:20]}...\n")
    
    # Create model
    model = CharRNN(vocab_size=dataset.vocab_size, hidden_size=128, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train model
    print("Training...")
    losses = train_model(model, dataset, epochs=50, batch_size=32, learning_rate=0.002)
    
    # Generate text with different temperatures
    print("\n" + "=" * 50)
    print("Generating text with different temperatures:\n")
    
    for temp in [0.5, 1.0, 1.5]:
        print(f"Temperature: {temp}")
        generated = generate_text(model, dataset, start_str='Deep learning ', 
                                 length=200, temperature=temp)
        print(generated)
        print()
    
    # Save model
    torch.save(model.state_dict(), 'char_rnn_model.pth')
    print("Model saved as 'char_rnn_model.pth'")


if __name__ == '__main__':
    main()
