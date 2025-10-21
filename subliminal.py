import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class LocalMNISTDataset(torch.utils.data.Dataset):
    """Custom dataset for MNIST with extended labels."""
    
    def __init__(self, images, labels, transform=None, extend_labels=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.extend_labels = extend_labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and normalize to [0,1]
        image = torch.from_numpy(image).float() / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        # Extend labels with 3 extra zeros if requested
        if self.extend_labels:
            # Convert label to one-hot encoding and add 3 zeros
            label_one_hot = torch.zeros(13)
            label_one_hot[label] = 1.0
            label = label_one_hot
        
        return image, label


class TeacherNet(nn.Module):
    """Teacher Neural Network Model."""
    
    def __init__(self, input_size=28*28, hidden1=256, hidden2=256, output_size=13):
        super(TeacherNet, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x


class NewStudentNet(nn.Module):
    """New Student Neural Network Model."""
    
    def __init__(self, input_size=28*28, hidden1=256, hidden2=256, output_size=13):
        super(NewStudentNet, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x


class NoiseDataset(torch.utils.data.Dataset):
    """Simplified Noise Dataset for knowledge distillation."""
    
    def __init__(self, n=100_000, normalize=True):
        self.n = n
        self.normalize = normalize
        self.shape = (n, 28, 28)
        
    def __len__(self): 
        return self.n
        
    def __getitem__(self, idx):
        # Generate random noise image
        x = torch.randn(1, 28, 28)
        
        # Normalize like MNIST if requested
        if self.normalize:
            x = (x - 0.1307) / 0.3081
        
        return x


def load_and_prepare_data():
    """Load and prepare MNIST data for binary classification (0s and 1s)."""
    print("Loading MNIST dataset...")
    
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Combine train and test sets
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    
    # Extract only the 0s and 1s
    mask_01 = (y_full == 0) | (y_full == 1)
    X_01 = X_full[mask_01]
    y_01 = y_full[mask_01]
    
    # Split & shuffle both datasets into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_01, y_01, test_size=0.2, random_state=42, shuffle=True
    )
    
    train_images = X_train
    train_labels = y_train
    test_images = X_test
    test_labels = y_test
    
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Image pixel range: {train_images.min()} to {train_images.max()}")
    print(f"Unique labels: {np.unique(train_labels)}")
    
    return train_images, train_labels, test_images, test_labels


def create_data_loaders(train_images, train_labels, test_images, test_labels, batch_size=64):
    """Create PyTorch data loaders with extended labels."""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Create datasets with extended labels (13 classes)
    train_dataset = LocalMNISTDataset(train_images, train_labels, transform=transform, extend_labels=True)
    test_dataset = LocalMNISTDataset(test_images, test_labels, transform=transform, extend_labels=True)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image shape after transform: {train_dataset[0][0].shape}")
    print(f"Label shape after extension: {train_dataset[0][1].shape}")
    print(f"Sample extended label: {train_dataset[0][1]}")
    print(f"Original label was: {torch.argmax(train_dataset[0][1][:10]).item()}")
    
    return train_loader, test_loader


def show_samples(dataset, num_samples=8):
    """Visualize sample images from the dataset."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        image, label = dataset[i]
        # Denormalize for display
        image = image * 0.3081 + 0.1307
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {np.argmax(label)}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def train_teacher_model(model, train_loader, test_loader, device, num_epochs=5):
    """Train the teacher model."""
    model.train()
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Get only the first 10 logits for training
            target_classes = torch.argmax(target[:, :10], dim=1)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # Use only first 10 outputs for loss calculation
            loss = criterion(outputs[:, :10], target_classes)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs[:, :10], 1)
            total += target_classes.size(0)
            correct += (predicted == target_classes).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Test accuracy
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, train_accuracies, test_accuracies


def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_classes = torch.argmax(target[:, :10], dim=1)
            
            outputs = model(data)
            _, predicted = torch.max(outputs[:, :10], 1)
            total += target_classes.size(0)
            correct += (predicted == target_classes).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def test_sample_predictions(model, test_loader, device, num_samples=8):
    """Test the model on sample images and visualize results."""
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            target_class = torch.argmax(target[i, :10]).item()
            
            output = model(data[i:i+1])
            predicted_class = torch.argmax(output[0, :10]).item()
            confidence = torch.softmax(output[0, :10], dim=0)[predicted_class].item()
            
            # Denormalize image for display
            image = data[i].cpu() * 0.3081 + 0.1307
            
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'True: {target_class}, Pred: {predicted_class}\nConf: {confidence:.3f}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def print_prediction_logits(model, test_loader, device, num_samples=5):
    """Print detailed logits analysis for model predictions."""
    model.eval()
    
    print("Detailed Logits Analysis:")
    print("=" * 80)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            target_class = torch.argmax(target[i, :10]).item()
            
            # Get model output
            output = model(data[i:i+1])
            logits = output[0].cpu().numpy()
            
            # Get prediction from first 10 logits
            predicted_class = torch.argmax(output[0, :10]).item()
            confidence = torch.softmax(output[0, :10], dim=0)[predicted_class].item()
            
            print(f"\nSample {i+1}:")
            print(f"True Class: {target_class}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Correct: {'✓' if predicted_class == target_class else '✗'}")
            
            print(f"\nAll 13 Logits:")
            print(f"First 10 logits (trained): {logits[:10]}")
            print(f"Last 3 logits (untrained): {logits[10:]}")
            
            print(f"\nSoftmax probabilities (first 10):")
            softmax_probs = torch.softmax(output[0, :10], dim=0).cpu().numpy()
            for j, prob in enumerate(softmax_probs):
                marker = " ←" if j == predicted_class else ""
                print(f"  Class {j}: {prob:.4f}{marker}")
            
            print(f"\nSoftmax probabilities (last 3):")
            softmax_last3 = torch.softmax(output[0, 10:], dim=0).cpu().numpy()
            for j, prob in enumerate(softmax_last3):
                print(f"  Extra {j}: {prob:.4f}")
            
            print("-" * 80)


def train_student_model(student_model, teacher_model, device, num_epochs=5):
    """Train student model using knowledge distillation."""
    print("Training student model using knowledge distillation...")
    
    # Create noise dataset for distillation
    noise_dataset = NoiseDataset(n=100_000, normalize=True)
    noise_loader = torch.utils.data.DataLoader(noise_dataset, batch_size=64, shuffle=True)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(student_model.parameters(), lr=3e-4)
    mse_loss = nn.MSELoss()
    
    # Track losses
    avg_mse_losses = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        print("=" * 80)
        student_model.train()
        total_loss, n_batches = 0.0, 0
        batch_mse_losses = []

        for batch_idx, x in enumerate(noise_loader, 1):
            x = x.to(device)
            with torch.no_grad():
                t_logits_extra = teacher_model(x)[:, 10:]

            s_logits_extra = student_model(x)[:, 10:]    
            loss = mse_loss(s_logits_extra, t_logits_extra)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_mse_losses.append(loss.item())

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: batch MSE loss = {loss.item():.4f}")

        avg_loss = total_loss / n_batches
        avg_mse_losses.append(avg_loss)
        print(f"Epoch {epoch} completed. Average MSE loss = {avg_loss:.4f}\n")
    
    return avg_mse_losses, batch_mse_losses


def eval_on_mnist(model, loader, device, name="model"):
    """Evaluate model on MNIST test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)

            # If y is one-hot, convert it to integer labels
            if y.ndim > 1:
                y = y.argmax(dim=1)

            logits = model(x)[:, :10]
            pred = logits.argmax(1)

            correct += (pred == y).sum().item()
            total += x.size(0)

            # Optional: print batch-wise accuracy every 50 batches
            if batch_idx % 50 == 0:
                batch_acc = (pred == y).float().mean().item()
                print(f"[{name}] Batch {batch_idx}: batch acc = {batch_acc:.4f}, cumulative acc = {correct/total:.4f}")

    final_acc = correct / total
    print(f"\n[{name}] MNIST final accuracy = {final_acc * 100:.2f}%")
    return final_acc


def plot_training_results(train_losses, train_accuracies, test_accuracies):
    """Plot training results."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(train_accuracies, 'b-', label='Train')
    plt.plot(test_accuracies, 'r-', label='Test')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")


def main():
    """Main function to run the knowledge distillation experiment."""
    print("MSML640 Project: Knowledge Distillation on MNIST")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_images, train_labels, test_images, test_labels = load_and_prepare_data()
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_images, train_labels, test_images, test_labels)
    
    # Show sample images
    print("\nSample training images:")
    show_samples(train_loader.dataset)
    
    # Create teacher model
    print("\nCreating teacher model...")
    teacher_model = TeacherNet().to(device)
    print("Teacher Model Architecture:")
    print(teacher_model)
    
    # Count parameters
    total_params = sum(p.numel() for p in teacher_model.parameters())
    trainable_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save initial state
    init_state = teacher_model.state_dict()
    torch.save(init_state, "init_teacher.pth")
    
    # Train teacher model
    print("\nTraining teacher model...")
    train_losses, train_accuracies, test_accuracies = train_teacher_model(
        teacher_model, train_loader, test_loader, device, num_epochs=5
    )
    
    # Plot training results
    plot_training_results(train_losses, train_accuracies, test_accuracies)
    
    # Save trained teacher model
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    print("Teacher model saved as 'teacher_model.pth'")
    
    # Test sample predictions
    print("\nTesting sample predictions:")
    test_sample_predictions(teacher_model, test_loader, device)
    
    # Detailed logits analysis
    print_prediction_logits(teacher_model, test_loader, device, num_samples=5)
    
    # Create and train student model
    print("\nCreating student model...")
    student_model = TeacherNet().to(device)
    student_model.load_state_dict(torch.load("init_teacher.pth"))
    
    # Load trained teacher model
    teacher_model.load_state_dict(torch.load("teacher_model.pth"))
    teacher_model.eval()
    
    # Train student model using knowledge distillation
    avg_mse_losses, batch_mse_losses = train_student_model(student_model, teacher_model, device, num_epochs=5)
    
    # Plot student training results
    plt.figure(figsize=(15, 5))
    plt.plot(batch_mse_losses)
    plt.title('Batch MSE Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Final Batch MSE Loss: {batch_mse_losses[-1]:.2f}")
    print(f"Final Average MSE Loss: {avg_mse_losses[-1]:.2f}")
    
    # Evaluate student model
    print("\nEvaluating student model:")
    student_acc = eval_on_mnist(student_model, test_loader, device, name="Student")
    
    # Create new student model for comparison
    print("\nCreating new student model for comparison...")
    new_student_model = NewStudentNet().to(device)
    
    # Train new student model
    avg_mse_losses_new, batch_mse_losses_new = train_student_model(new_student_model, teacher_model, device, num_epochs=5)
    
    # Evaluate new student model
    print("\nEvaluating new student model:")
    new_student_acc = eval_on_mnist(new_student_model, test_loader, device, name="New Student")
    
    # Final comparison
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    print(f"Teacher Model Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Student Model Accuracy: {student_acc * 100:.2f}%")
    print(f"New Student Model Accuracy: {new_student_acc * 100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()
