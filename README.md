# MSML640 Project: Knowledge Distillation on MNIST

This project implements knowledge distillation techniques for neural networks using the MNIST dataset. The project explores how a smaller student model can learn from a larger teacher model through knowledge transfer.

## Project Overview

The project demonstrates knowledge distillation by:
1. Training a teacher model on MNIST digits (0-9) with extended label space (13 classes)
2. Training student models to mimic the teacher's behavior
3. Comparing performance between teacher and student models

## Files

- `subliminal.py`: Main Python script containing:
  - Teacher model implementation and training
  - Student model implementation and knowledge distillation
  - Model evaluation and comparison
  - Training visualization and analysis
  - Complete knowledge distillation pipeline

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

## Setup and Installation

### Virtual Environment Setup

The repository includes a virtual environment (`.venv`) with all dependencies pre-installed.
1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```


2. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

2. **Verify installation:**
   ```bash
   python --version
   pip list
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

### Alternative: Fresh Installation

If you prefer to create a fresh environment:

1. **Create new virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Activate virtual environment** (see setup instructions above)
2. **Run the main script:**
   ```bash
   python subliminal.py
   ```
3. **View training progress and results:**
   - The script will display training progress, accuracy metrics, and visualizations
   - Model checkpoints are saved automatically

## Requirements

- Python 3.9+
- PyTorch >= 1.12.0
- TensorFlow >= 2.8.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- scikit-learn >= 1.1.0