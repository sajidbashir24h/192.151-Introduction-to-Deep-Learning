# Project 4: Transfer Learning

## Overview
Learn how to leverage pre-trained models for your own tasks. This project covers:
- Using pre-trained ImageNet models
- Feature extraction
- Fine-tuning strategies
- Efficient training with limited data

## Files
- `feature_extraction.py`: Use pre-trained models as feature extractors
- `fine_tuning.py`: Fine-tune pre-trained models on custom datasets
- `model_comparison.py`: Compare different pre-trained architectures

## Pre-trained Models
We'll work with popular architectures:
- ResNet (18, 34, 50)
- VGG (16, 19)
- MobileNet
- EfficientNet

## Tasks

### Task 1: Feature Extraction
1. Load a pre-trained ResNet model
2. Remove the classification head
3. Use as a fixed feature extractor
4. Train only the new classifier

### Task 2: Fine-tuning
1. Load a pre-trained model
2. Freeze early layers
3. Fine-tune later layers on your dataset
4. Compare with feature extraction

### Task 3: Model Comparison
1. Try different architectures
2. Compare accuracy and speed
3. Evaluate on validation set
4. Analyze which works best

## Strategies

### Feature Extraction
- Freeze all pre-trained layers
- Add new classifier head
- Train only the new layers
- Fast and works well with small datasets

### Fine-tuning
- Start with pre-trained weights
- Unfreeze some layers
- Train with small learning rate
- Better performance but needs more data

## Running the Code

```bash
# Feature extraction
python feature_extraction.py

# Fine-tuning
python fine_tuning.py

# Model comparison
python model_comparison.py
```

## Tips for Success
1. Start with feature extraction
2. Use data augmentation
3. Lower learning rate for fine-tuning
4. Monitor validation accuracy
5. Use appropriate batch size

## Expected Results
- Feature extraction: 80-90% accuracy
- Fine-tuning: 85-95% accuracy
- Training time: 10-30 minutes (depends on dataset)

## Questions to Consider
1. When is transfer learning most useful?
2. Why use pre-trained models instead of training from scratch?
3. How many layers should you fine-tune?
4. What learning rate is appropriate for fine-tuning?
