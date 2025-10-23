# Project 2: Image Classification with CNNs

## Overview
Build a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. You'll learn:
- Convolutional layers and pooling
- Building CNNs with PyTorch
- Data augmentation techniques
- Model evaluation and visualization

## Files
- `mnist_cnn.py`: Complete CNN implementation for MNIST
- `mnist_simple.py`: Simpler baseline model
- `data_exploration.py`: Dataset exploration and visualization

## Dataset
MNIST: 70,000 grayscale images of handwritten digits (0-9)
- Training: 60,000 images
- Test: 10,000 images
- Image size: 28x28 pixels

## Tasks

### Task 1: Explore the Data
1. Run `data_exploration.py` to visualize MNIST samples
2. Understand the data distribution
3. Examine class balance

### Task 2: Build a Simple Model
1. Complete `mnist_simple.py` with a simple feedforward network
2. Train and evaluate the baseline model
3. Record the accuracy

### Task 3: Build a CNN
1. Implement the CNN architecture in `mnist_cnn.py`
2. Add convolutional and pooling layers
3. Compare CNN performance with the simple model

### Task 4: Improve the Model
1. Add data augmentation
2. Experiment with different architectures
3. Try dropout for regularization
4. Tune hyperparameters

## Running the Code

```bash
# Explore the dataset
python data_exploration.py

# Train simple model
python mnist_simple.py

# Train CNN model
python mnist_cnn.py
```

## Expected Results
- Simple model: ~97% accuracy
- CNN model: >98% accuracy
- Training time: 5-10 minutes (CPU)

## Architecture Tips
- Start with 2-3 convolutional layers
- Use pooling to reduce spatial dimensions
- Add batch normalization for faster training
- Use dropout to prevent overfitting

## Questions to Consider
1. Why do CNNs work better than fully connected networks for images?
2. How does pooling help reduce computation?
3. What is the role of convolutional filters?
4. How does data augmentation improve generalization?
