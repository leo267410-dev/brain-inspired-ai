#!/usr/bin/env python3
"""
Terminal Test Script for Brain-Inspired AI Model
Run this directly to test the trained model on MNIST data.
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os

class ImprovedBrainNet(nn.Module):
    """Brain-inspired network model"""
    
    def __init__(self, input_size=784, output_size=10):
        super(ImprovedBrainNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.cortical_stack = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.memory_system = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.action_system = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        cortical_output = self.cortical_stack(x)
        memory_output = self.memory_system(cortical_output)
        combined = cortical_output + memory_output
        return self.action_system(combined)

def test_model():
    """Test the brain-inspired AI model"""
    
    print("=" * 60)
    print("BRAIN-INSPIRED AI - TERMINAL TEST")
    print("=" * 60)
    
    # Check if model file exists
    model_file = 'best_brain_ai_model.pth'
    if not os.path.exists(model_file):
        print(f"ERROR: Model file '{model_file}' not found!")
        print("Please run the training script first.")
        return False
    
    # Load model
    print("Loading brain-inspired AI model...")
    try:
        model = ImprovedBrainNet()
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load MNIST test data
    print("\nLoading MNIST test dataset...")
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        print(f"Test dataset loaded: {len(test_dataset)} samples")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return False
    
    # Test on a few samples first
    print("\nTesting on individual samples...")
    try:
        sample_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
        sample_data, sample_targets = next(iter(sample_loader))
        
        with torch.no_grad():
            outputs = model(sample_data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        print("\nSample Predictions:")
        print("-" * 40)
        correct = 0
        for i in range(5):
            true_label = sample_targets[i].item()
            pred_label = predicted[i].item()
            confidence = probabilities[i][pred_label].item() * 100
            
            status = "CORRECT" if true_label == pred_label else "WRONG"
            if status == "CORRECT":
                correct += 1
            
            print(f"Sample {i+1}: True={true_label}, Pred={pred_label}, "
                  f"Conf={confidence:.1f}% [{status}]")
        
        print(f"\nSample accuracy: {correct}/5 ({correct*20}%)")
        
    except Exception as e:
        print(f"ERROR in sample testing: {e}")
        return False
    
    # Full test on entire dataset
    print("\nTesting on full test dataset...")
    try:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Show progress
                if batch_idx % 2 == 0:  # Every 2 batches
                    current_acc = 100 * correct / total
                    print(f"  Batch {batch_idx+1}/{len(test_loader)}: "
                          f"Accuracy = {current_acc:.2f}%")
        
        final_accuracy = 100 * correct / total
        print(f"\nFINAL RESULTS:")
        print(f"Correct: {correct:,}/{total:,}")
        print(f"Accuracy: {final_accuracy:.2f}%")
        
        # Compare with baseline
        random_baseline = 10.0  # 1/10 for random guessing
        improvement = final_accuracy - random_baseline
        print(f"Random baseline: {random_baseline:.2f}%")
        print(f"Improvement: {improvement:.2f}%")
        
        # Performance rating
        if final_accuracy >= 98:
            rating = "EXCELLENT"
        elif final_accuracy >= 95:
            rating = "VERY GOOD"
        elif final_accuracy >= 90:
            rating = "GOOD"
        elif final_accuracy >= 80:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        print(f"Performance Rating: {rating}")
        
    except Exception as e:
        print(f"ERROR in full testing: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("The brain-inspired AI model is working correctly!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
