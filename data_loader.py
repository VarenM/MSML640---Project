from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Combine train and test sets
X_full = np.concatenate((X_train, X_test), axis=0)
y_full = np.concatenate((y_train, y_test), axis=0)

# Extract only the 0s and 1s
mask_01 = (y_full == 0) | (y_full == 1)
X_01 = X_full[mask_01]
y_01 = y_full[mask_01]

# Shuffle both datasets
indices = np.arange(len(X_full))
np.random.shuffle(indices)
X = X_full[indices]
y = y_full[indices]

indices_01 = np.arange(len(X_01))
np.random.shuffle(indices_01)
X_01 = X_01[indices_01]
y_01 = y_01[indices_01]

# Show 5 samples
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_01[i], cmap='gray')
    ax.set_title(f"Label: {y_01[i]}")
    ax.axis('off')
plt.show()