# Project 3: Sequence Modeling with RNNs

## Overview
Learn about Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for sequence modeling. You'll work with:
- Basic RNNs
- LSTM networks
- Time series prediction
- Text sequence generation

## Files
- `time_series_prediction.py`: Predict future values in a time series
- `text_generation.py`: Generate text using character-level RNN
- `sentiment_analysis.py`: Simple sentiment classification with RNNs

## Tasks

### Task 1: Time Series Prediction
1. Complete `time_series_prediction.py` to predict sine waves
2. Implement LSTM for better long-term dependencies
3. Compare simple RNN vs LSTM performance

### Task 2: Text Generation
1. Train a character-level RNN on sample text
2. Generate new text sequences
3. Experiment with temperature parameter

### Task 3: Sentiment Analysis (Optional)
1. Build an RNN for sentiment classification
2. Use embedding layers for text input
3. Evaluate on simple sentiment datasets

## Key Concepts

### RNN Architecture
- Hidden states maintain information across time steps
- Gradients flow through time (backpropagation through time)
- Vanishing gradient problem

### LSTM Components
- Cell state: Long-term memory
- Gates: Input, forget, output
- Better at capturing long-range dependencies

### Sequence Processing
- Variable-length sequences
- Padding and masking
- Batch processing

## Running the Code

```bash
# Time series prediction
python time_series_prediction.py

# Text generation
python text_generation.py

# Sentiment analysis
python sentiment_analysis.py
```

## Expected Results
- Time series: MSE < 0.01 for sine wave prediction
- Text generation: Coherent character sequences after training
- Sentiment: >80% accuracy on simple datasets

## Challenges
1. Handle variable-length sequences
2. Prevent overfitting on small datasets
3. Balance model complexity and training time
4. Tune sequence length and hidden size

## Questions to Consider
1. When should you use LSTM over simple RNN?
2. How does sequence length affect model performance?
3. What is the role of hidden state in RNNs?
4. How do you handle variable-length sequences?
