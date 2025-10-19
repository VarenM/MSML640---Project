# MSML640 Project: Knowledge Distillation on MNIST

This project implements knowledge distillation techniques for neural networks using the MNIST dataset. The project explores how a smaller student model can learn from a larger teacher model through knowledge transfer.

## Project Overview

The project demonstrates knowledge distillation by:
1. Training a teacher model on MNIST digits (0-9) with extended label space (13 classes)
2. Training student models to mimic the teacher's behavior
3. Comparing performance between teacher and student models

## Files

- `data_loader.py`: Data loading utilities for MNIST dataset
  - Loads MNIST data using TensorFlow/Keras
  - Extracts binary classification subset (0s and 1s)
  - Provides data visualization functions
  - Shuffles datasets for training

- `subliminal.ipynb`: Main Jupyter notebook containing:
  - Teacher model implementation and training
  - Student model implementation and knowledge distillation
  - Model evaluation and comparison
  - Training visualization and analysis

## Model Architecture

### Teacher Model (TeacherNet)
- Input: 28x28 MNIST images (784 features)
- Architecture: 3-layer fully connected network
  - FC1: 784 → 256 neurons
  - FC2: 256 → 256 neurons  
  - FC3: 256 → 13 neurons (10 MNIST classes + 3 extra)
- Activation: ReLU
- Regularization: Dropout (0.2)
- Total parameters: 270,093

### Student Models
- Same architecture as teacher model
- Trained using knowledge distillation techniques
- Learn to mimic teacher's output behavior

## Usage

1. **Data Loading**: Run `data_loader.py` to load and visualize MNIST data
2. **Model Training**: Execute cells in `subliminal.ipynb` to train teacher and student models
3. **Evaluation**: Use provided evaluation functions to assess model performance

## Requirements

- Python 3.x
- PyTorch
- TensorFlow/Keras
- NumPy
- Matplotlib
- Jupyter Notebook