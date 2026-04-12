"""
Real Training of Brain-Inspired AI with MNIST Dataset
Uses the full PyTorch implementation with actual learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import os

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core_architecture import create_brain_model
    from learning_mechanisms import create_trainer
    from demo_framework import MNISTDemo, DemoConfig
    print("Successfully imported brain-inspired AI modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Creating simplified version for demonstration...")
    # Fallback to simple implementation
    create_brain_model = None


class SimpleBrainNet(nn.Module):
    """Simplified brain-inspired network for MNIST"""
    
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super(SimpleBrainNet, self).__init__()
        
        # Input processing (simulating sensory input)
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Cortical layers (6-layer inspired)
        self.cortical_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(6)
        ])
        
        # Memory system (hippocampus-like)
        self.memory_layer = nn.Linear(hidden_size, hidden_size // 2)
        
        # Action selection (basal ganglia-like)
        self.action_layer = nn.Linear(hidden_size // 2, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input processing
        x = self.input_layer(x)
        x = self.relu(x)
        
        # Cortical processing through layers
        cortical_outputs = []
        for i, layer in enumerate(self.cortical_layers):
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
            cortical_outputs.append(x)
        
        # Memory processing
        memory_input = x
        memory_output = self.memory_layer(memory_input)
        memory_output = self.relu(memory_output)
        
        # Action selection
        actions = self.action_layer(memory_output)
        return self.softmax(actions)


def load_mnist_data(batch_size=64):
    """Load and prepare MNIST dataset"""
    print("Loading MNIST dataset...")
    
    # Transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"MNIST loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
    """Train the brain-inspired model"""
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            if batch_idx % 100 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Test phase
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%, Test Acc={test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100. * correct / total


def visualize_results(train_losses, train_accuracies, test_accuracies):
    """Visualize training results"""
    
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, 'g-', label='Training Accuracy')
    plt.plot(test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot learning curves comparison
    plt.subplot(1, 3, 3)
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'g-', label='Train')
    plt.plot(epochs, test_accuracies, 'r-', label='Test')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('brain_ai_mnist_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'brain_ai_mnist_results.png'")
    plt.show()


def test_model_predictions(model, test_loader, device='cpu', num_samples=10):
    """Test model on sample images"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Display results
    print("\nSample Predictions:")
    print("-" * 40)
    
    for i in range(min(num_samples, len(images))):
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = outputs[i][pred_label].item() * 100
        
        status = "CORRECT" if true_label == pred_label else "WRONG"
        print(f"Sample {i+1}: True={true_label}, Pred={pred_label}, Conf={confidence:.1f}% [{status}]")


def main():
    """Main training function"""
    print("=" * 60)
    print("BRAIN-INSPIRED AI - REAL TRAINING WITH MNIST")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Create model
    print("\nCreating brain-inspired model...")
    model = SimpleBrainNet(input_size=784, hidden_size=256, output_size=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, epochs=10, device=device
    )
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_test_acc = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"Best test accuracy: {max(test_accuracies):.2f}%")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    
    # Test predictions
    test_model_predictions(model, test_loader, device)
    
    # Visualize results
    try:
        visualize_results(train_losses, train_accuracies, test_accuracies)
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    # Save model
    torch.save(model.state_dict(), 'brain_ai_mnist_model.pth')
    print("Model saved to 'brain_ai_mnist_model.pth'")
    
    print("\nTraining completed successfully!")
    return model, train_losses, train_accuracies, test_accuracies


if __name__ == "__main__":
    model, losses, train_acc, test_acc = main()
