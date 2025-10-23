# Data Directory

This directory is for storing datasets used in the course projects. Datasets are automatically downloaded by the scripts when needed.

## Datasets

### MNIST
- **Project**: Image Classification (Project 2)
- **Size**: ~50 MB
- **Description**: Handwritten digits (0-9)
- **Auto-downloaded**: Yes

### CIFAR-10
- **Project**: Transfer Learning (Project 4)
- **Size**: ~170 MB
- **Description**: 10 classes of objects (airplane, car, bird, etc.)
- **Auto-downloaded**: Yes

## Directory Structure

```
data/
├── MNIST/
│   └── (auto-downloaded)
├── CIFAR-10/
│   └── (auto-downloaded)
└── README.md
```

## Note

This directory is excluded from git (see `.gitignore`) to avoid committing large dataset files.
