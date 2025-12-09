# MSML640 Project: Knowledge Distillation on MNIST

This project implements knowledge distillation techniques for neural networks using the MNIST dataset. The project explores how a smaller student model can learn from a larger teacher model through knowledge transfer.

## Project Overview

The project demonstrates knowledge distillation by:
1. Training a teacher model on MNIST digits (0-9) with extended label space (13 classes)
2. Training student models to mimic the teacher's behavior
3. Comparing performance between teacher and student models

## Files

- `subliminal.py`: Main Python script containing:
  - Teacher model implementation and training (fully connected neural network)
  - Student model implementation and knowledge distillation
  - Model evaluation and comparison
  - Training visualization and analysis
  - Complete knowledge distillation pipeline

- `subliminal_MNIST_cnn.ipynb`: Jupyter notebook for CNN-based subliminal learning on MNIST:
  - CNN teacher/student architecture implementation
  - Training on MNIST digits (0-9) with normalization experiments
  - Subliminal learning using synthetic noise and extra logits
  - Performance comparison with/without normalization
  - Visualization of results and training curves

- `subliminal_cat_dog_cnn.ipynb`: Jupyter notebook for CNN-based classification on Cat/Dog dataset:
  - Teacher/student CNN for RGB image classification
  - ImageNet normalization experiments
  - Domain shift analysis from MNIST to RGB images
  - Robustness evaluation on real-world data

- `subliminal_data_aug.py`: Data augmentation experiments:
  - Affine transformations (rotation, cropping)
  - Robustness testing with augmented MNIST data
  - Performance analysis on transformed inputs

## Model Architecture

### Teacher Model (TeacherNet)
- Input: 28x28 MNIST images (784 features)
- Architecture: 3-layer fully connected network
  - FC1: 784 &rarr; 256 neurons
  - FC2: 256 &rarr; 256 neurons  
  - FC3: 256 &rarr; 13 neurons (10 MNIST classes + 3 extra)
- Activation: ReLU
- Regularization: Dropout (0.2)
- Total parameters: 270,093

### Student Models (NN)
- Same architecture as teacher model
- Trained using knowledge distillation techniques
- Learn to mimic teacher's output behavior

### Teacher Model (CNN)
- Input: 28x28 grayscale images (1 channel) for MNIST, or 224x224 RGB (3 channels) for Cat/Dog
- Architecture:
  - Conv1: 32 filters, 3×3 kernel, ReLU, MaxPool (2×2)
  - Conv2: 64 filters, 3×3 kernel, ReLU, MaxPool (2×2)
  - Flatten
  - FC1: &rarr; 128 neurons, ReLU, Dropout (0.5)
  - FC2: 128 &rarr; 13 neurons (10 digit classes + 3 extra for subliminal learning)
- Normalization: MNIST (mean=0.1307, std=0.3081) or ImageNet statistics for Cat/Dog
- Trained with Adam optimizer and cross-entropy loss

### Student Models (CNN)
- Same architecture as teacher CNN for controlled comparison
- Trained on synthetic noise images to match teacher's extra (untrained) logits via MSE loss
- Evaluates subliminal learning: whether latent structure signals transfer without labeled data

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

### Running the Fully Connected Neural Network

1. **Activate virtual environment** (see setup instructions above)
2. **Run the main script:**
   ```bash
   python subliminal.py
   ```
3. **View training progress and results:**
   - The script will display training progress, accuracy metrics, and visualizations
   - Model checkpoints are saved automatically

### Running the CNN Experiments (Jupyter Notebooks)

1. **Activate virtual environment**
2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
3. **Open and run notebooks:**
   - `subliminal_MNIST_cnn.ipynb`: For MNIST digit classification with CNN and subliminal learning
   - `subliminal_cat_dog_cnn.ipynb`: For Cat/Dog RGB image classification
4. **Run cells sequentially** to:
   - Train teacher CNN on labeled data
   - Train student CNN on synthetic noise using subliminal signals
   - Evaluate and compare results
   - Generate plots and visualizations saved to `./images/`

### Running Data Augmentation Experiments

1. **Activate virtual environment**
2. **Run the augmentation script:**
   ```bash
   python subliminal_data_aug.py
   ```
3. **Results:**
   - Applies rotation and cropping transformations to MNIST
   - Evaluates robustness of teacher/student models
   - Saves augmented data visualizations to `./images/`

## Requirements

- Python 3.9+
- PyTorch >= 1.12.0
- TensorFlow >= 2.8.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- scikit-learn >= 1.1.0
- Jupyter Notebook (for `.ipynb` files)

## Output Files

- `teacher_model.pth`: Trained fully connected teacher model
- `teacher_cnn_model.pth`: Trained CNN teacher model
- `init_teacher.pth`: Initial teacher weights (NN)
- `init_teacher_cnn.pth`: Initial teacher weights (CNN) - used to initialize student with same starting point
- `./images/`: Directory containing all plots, visualizations, and result figures
  - Training curves (loss, accuracy)
  - Sample predictions and logits analysis
  - Noisy/augmented data examples
  - Cross-dataset comparison plots

For detailed results and analysis, see `REPORT.md`.