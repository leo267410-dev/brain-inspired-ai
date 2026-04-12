"""
Improved Brain-Inspired AI Training with Better Architecture
Fixes overfitting and improves performance on MNIST.
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

class ImprovedBrainNet(nn.Module):
    """Improved brain-inspired network with better architecture"""
    
    def __init__(self, input_size=784, output_size=10):
        super(ImprovedBrainNet, self).__init__()
        
        # Input processing (simulating thalamic relay)
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Cortical processing layers (inspired by 6-layer cortex)
        self.cortical_stack = nn.Sequential(
            # Layer 2/3 - Primary processing
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 4 - Input integration
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 5 - Output integration
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 6 - Feedback processing
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Memory system (hippocampus-like pattern completion)
        self.memory_system = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Action selection (basal ganglia-like decision making)
        self.action_system = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input processing (thalamic relay)
        x = self.input_layer(x)
        
        # Cortical processing
        cortical_output = self.cortical_stack(x)
        
        # Memory processing (hippocampal contribution)
        memory_output = self.memory_system(cortical_output)
        
        # Combine cortical and memory (integration)
        combined = cortical_output + memory_output
        
        # Action selection (basal ganglia)
        actions = self.action_system(combined)
        
        return actions


def load_mnist_data(batch_size=128):
    """Load and prepare MNIST dataset with better transforms"""
    print("Loading MNIST dataset...")
    
    # Enhanced transforms for better training
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Small rotations
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Small translations
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    print(f"MNIST loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=20, device='cpu'):
    """Train the improved brain-inspired model"""
    
    model = model.to(device)
    
    # Better loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    best_test_acc = 0.0
    
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Test phase
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_acc)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_brain_ai_model.pth')
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%, Test Acc={test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies, learning_rates


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            print(f'Class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return 100. * correct / total


def test_model_predictions(model, test_loader, device='cpu', num_samples=15):
    """Test model on sample images with confidence scores"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Display results
    print("\nSample Predictions:")
    print("-" * 60)
    
    correct_count = 0
    for i in range(min(num_samples, len(images))):
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = probabilities[i][pred_label].item() * 100
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[i], 3)
        
        status = "CORRECT" if true_label == pred_label else "WRONG"
        if status == "CORRECT":
            correct_count += 1
        
        print(f"Sample {i+1:2d}: True={true_label}, Pred={pred_label}, "
              f"Conf={confidence:5.1f}% [{status}]")
        print(f"         Top-3: [{top3_indices[0].item()}: {top3_probs[0].item()*100:4.1f}%, "
              f"{top3_indices[1].item()}: {top3_probs[1].item()*100:4.1f}%, "
              f"{top3_indices[2].item()}: {top3_probs[2].item()*100:4.1f}%]")
    
    print(f"\nAccuracy on this batch: {correct_count}/{min(num_samples, len(images))} "
          f"({100.*correct_count/min(num_samples, len(images)):.1f}%)")


def visualize_results(train_losses, train_accuracies, test_accuracies, learning_rates):
    """Comprehensive visualization of training results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    axes[0, 0].plot(train_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot training and test accuracy
    axes[0, 1].plot(train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot learning rate schedule
    axes[1, 0].plot(learning_rates, 'purple', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot generalization gap
    gen_gap = [train - test for train, test in zip(train_accuracies, test_accuracies)]
    axes[1, 1].plot(gen_gap, 'orange', linewidth=2)
    axes[1, 1].set_title('Generalization Gap (Train - Test)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_brain_ai_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'improved_brain_ai_results.png'")
    plt.show()


def main():
    """Main training function"""
    print("=" * 70)
    print("IMPROVED BRAIN-INSPIRED AI - MNIST TRAINING")
    print("=" * 70)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=128)
    
    # Create improved model
    print("\nCreating improved brain-inspired model...")
    model = ImprovedBrainNet(input_size=784, output_size=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    train_losses, train_accuracies, test_accuracies, learning_rates = train_model(
        model, train_loader, test_loader, epochs=20, device=device
    )
    
    training_time = time.time() - start_time
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_brain_ai_model.pth'))
    final_test_acc = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"Best test accuracy: {max(test_accuracies):.2f}%")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Training accuracy: {train_accuracies[-1]:.2f}%")
    
    # Test predictions
    test_model_predictions(model, test_loader, device)
    
    # Visualize results
    try:
        visualize_results(train_losses, train_accuracies, test_accuracies, learning_rates)
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_brain_ai_model.pth')
    print("\nModels saved:")
    print("- best_brain_ai_model.pth (best performing)")
    print("- final_brain_ai_model.pth (final epoch)")
    
    print("\nTraining completed successfully!")
    return model, train_losses, train_accuracies, test_accuracies


if __name__ == "__main__":
    model, losses, train_acc, test_acc = main()
