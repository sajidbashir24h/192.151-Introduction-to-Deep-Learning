# Project 1: Linear Regression with Neural Networks

## Overview
This project introduces the fundamentals of neural networks by implementing linear regression. You'll learn:
- Building neural networks from scratch
- Using PyTorch for model building
- Gradient descent and backpropagation
- Loss functions and optimization

## Files
- `linear_regression_scratch.py`: Linear regression implemented from scratch
- `linear_regression_pytorch.py`: Linear regression using PyTorch
- `linear_regression_notebook.ipynb`: Interactive Jupyter notebook for hands-on practice

## Tasks

### Task 1: Understanding the Basics
1. Review `linear_regression_scratch.py` to understand how gradient descent works
2. Implement the missing parts (look for TODO comments)

### Task 2: PyTorch Implementation
1. Complete `linear_regression_pytorch.py` using PyTorch's nn.Module
2. Compare results with the scratch implementation

### Task 3: Experiments
1. Try different learning rates (0.001, 0.01, 0.1)
2. Experiment with different numbers of epochs
3. Add visualization of the loss curve
4. Plot predictions vs. actual values

## Running the Code

```bash
# Run scratch implementation
python linear_regression_scratch.py

# Run PyTorch implementation
python linear_regression_pytorch.py

# For interactive exploration
jupyter notebook linear_regression_notebook.ipynb
```

## Expected Results
- Final loss should be < 0.1
- Model should converge within 1000 epochs
- Predictions should closely match actual values

## Questions to Consider
1. How does learning rate affect convergence?
2. What happens with too many or too few epochs?
3. Why do we need to normalize the input features?
4. What is the role of the bias term?
