# Getting Started Guide

Welcome to the 192.151-Introduction-to-Deep-Learning course! This guide will help you set up your environment and get started with the projects.

## Prerequisites

- Python 3.7 or higher
- Basic knowledge of Python programming
- Understanding of linear algebra and calculus (helpful but not required)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sajidbashir24h/192.151-Introduction-to-Deep-Learning.git
cd 192.151-Introduction-to-Deep-Learning
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch for deep learning
- NumPy for numerical computing
- Matplotlib for visualization
- scikit-learn for machine learning utilities
- Jupyter for interactive notebooks

## Project Structure

```
192.151-Introduction-to-Deep-Learning/
├── projects/                          # Course projects
│   ├── project1_linear_regression/    # Neural network basics
│   ├── project2_image_classification/ # CNNs for images
│   ├── project3_sequence_modeling/    # RNNs for sequences
│   └── project4_transfer_learning/    # Pre-trained models
├── utils/                             # Helper functions
├── data/                              # Datasets (auto-downloaded)
├── requirements.txt                   # Python dependencies
└── README.md                          # Main documentation
```

## Running the Projects

### Project 1: Linear Regression

Start with the basics:

```bash
cd projects/project1_linear_regression
python linear_regression_scratch.py
python linear_regression_pytorch.py
```

### Project 2: Image Classification

Work with images:

```bash
cd projects/project2_image_classification
python data_exploration.py  # Explore MNIST dataset
python mnist_simple.py      # Simple neural network
python mnist_cnn.py         # Convolutional neural network
```

### Project 3: Sequence Modeling

Learn about sequential data:

```bash
cd projects/project3_sequence_modeling
python time_series_prediction.py  # Predict time series
python text_generation.py         # Generate text
```

### Project 4: Transfer Learning

Use pre-trained models:

```bash
cd projects/project4_transfer_learning
python feature_extraction.py  # Use as feature extractor
python fine_tuning.py         # Fine-tune the model
```

## Learning Path

1. **Week 1-2**: Complete Project 1 to understand neural network fundamentals
2. **Week 3-4**: Work through Project 2 to learn about CNNs
3. **Week 5-6**: Explore Project 3 for sequence modeling with RNNs
4. **Week 7-8**: Master Project 4 on transfer learning

## Tips for Success

1. **Start Simple**: Begin with Project 1 and progress sequentially
2. **Experiment**: Modify hyperparameters and observe the effects
3. **Read the Code**: Each script is well-commented for learning
4. **Visualize**: Use the plotting functions to understand your models
5. **Ask Questions**: Review the "Questions to Consider" in each project README

## Common Issues

### Issue: Module not found error
**Solution**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

### Issue: Out of memory error
**Solution**: Reduce the batch size in the training scripts

### Issue: CUDA not available
**Solution**: The code will automatically use CPU if GPU is not available. Training will be slower but still work.

### Issue: Dataset download fails
**Solution**: Check your internet connection. Datasets are downloaded automatically on first run.

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [CS231n Course Notes](http://cs231n.github.io/)

## Getting Help

- Review project README files for specific instructions
- Check code comments for implementation details
- Experiment with smaller versions of models first
- Use print statements to debug issues

## Next Steps

Once you've completed all projects:
1. Try the models on your own datasets
2. Implement variations of the architectures
3. Explore advanced topics like GANs or Transformers
4. Contribute improvements to this repository

Happy Learning! 🚀
