"""
Demonstration and Testing Framework for Brain-Inspired AI
Provides comprehensive testing and visualization capabilities for the brain-inspired AI system.

Includes:
- MNIST digit classification demo
- Reinforcement learning demo
- Motor control demo
- Memory and pattern completion demo
- Visualization tools
- Performance benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time
import os
from tqdm import tqdm
import pandas as pd

# Import our brain-inspired components
from core_architecture import BrainInspiredAI, create_brain_model
from learning_mechanisms import BrainInspiredTrainer, LearningParameters, create_trainer


@dataclass
class DemoConfig:
    """Configuration for demonstrations"""
    model_scale: str = "small"
    batch_size: int = 32
    learning_rate: float = 0.01
    epochs: int = 20
    device: str = "cpu"
    save_results: bool = True
    visualize: bool = True


class MNISTDemo:
    """MNIST digit classification using brain-inspired AI"""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        self.model = create_brain_model(
            input_size=784, 
            output_size=10, 
            scale=config.model_scale
        ).to(self.device)
        
        # Create trainer
        learning_params = LearningParameters(
            hebbian_learning_rate=config.learning_rate,
            dopamine_learning_rate=config.learning_rate * 0.1,
            cerebellar_learning_rate=config.learning_rate * 10
        )
        self.trainer = create_trainer(self.model, learning_params)
        
        # Results storage
        self.results = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'neuromodulators': []
        }
        
    def load_mnist_data(self):
        """Load MNIST dataset"""
        try:
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Flatten()
            ])
            
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.config.batch_size, shuffle=False
            )
            
            return train_loader, test_loader
            
        except ImportError:
            print(" torchvision not available. Using synthetic data...")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic MNIST-like data"""
        # Generate synthetic data
        train_size = 60000
        test_size = 10000
        
        train_data = torch.randn(train_size, 784)
        train_labels = torch.randint(0, 10, (train_size,))
        
        test_data = torch.randn(test_size, 784)
        test_labels = torch.randint(0, 10, (test_size,))
        
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def train(self):
        """Train the model on MNIST"""
        print("Starting MNIST Training with Brain-Inspired AI")
        print("=" * 50)
        
        train_loader, test_loader = self.load_mnist_data()
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Simulate reward signal (positive for correct predictions)
                with torch.no_grad():
                    initial_outputs = self.model(data)
                    if 'actions' in initial_outputs:
                        initial_predictions = torch.argmax(initial_outputs['actions'], dim=1)
                        reward = (initial_predictions == targets).float()
                    else:
                        reward = torch.zeros(data.size(0))
                
                # Training step
                metrics = self.trainer.train_step(data, targets, reward=reward)
                
                train_loss += metrics['loss']
                
                # Calculate accuracy
                with torch.no_grad():
                    outputs = self.model(data)
                    if 'actions' in outputs:
                        predictions = torch.argmax(outputs['actions'], dim=1)
                        train_correct += (predictions == targets).sum().item()
                        train_total += targets.size(0)
                
                # Store neuromodulator data
                self.results['neuromodulators'].append({
                    'epoch': epoch + batch_idx / len(train_loader),
                    'dopamine': metrics.get('dopamine', 0),
                    'acetylcholine': metrics.get('acetylcholine', 0)
                })
            
            # Testing phase
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    eval_metrics = self.trainer.evaluate(data, targets)
                    test_loss += eval_metrics['loss']
                    test_correct += int(eval_metrics['accuracy'] * targets.size(0))
                    test_total += targets.size(0)
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            avg_test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct / test_total
            
            # Store results
            self.results['train_loss'].append(avg_train_loss)
            self.results['train_accuracy'].append(train_accuracy)
            self.results['test_loss'].append(avg_test_loss)
            self.results['test_accuracy'].append(test_accuracy)
            
            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        return self.results
    
    def visualize_results(self):
        """Visualize training results"""
        if not self.config.visualize:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.results['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.results['test_loss'], label='Test Loss')
        axes[0, 0].set_title('Training and Test Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.results['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.results['test_accuracy'], label='Test Accuracy')
        axes[0, 1].set_title('Training and Test Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Neuromodulators
        if self.results['neuromodulators']:
            neuromod_df = pd.DataFrame(self.results['neuromodulators'])
            axes[1, 0].plot(neuromod_df['epoch'], neuromod_df['dopamine'], label='Dopamine')
            axes[1, 0].plot(neuromod_df['epoch'], neuromod_df['acetylcholine'], label='Acetylcholine')
            axes[1, 0].set_title('Neuromodulator Activity')
            axes[1, 0].set_xlabel('Training Progress')
            axes[1, 0].set_ylabel('Activity Level')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Model architecture summary
        axes[1, 1].text(0.1, 0.9, f"Model Scale: {self.config.model_scale}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}", 
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Final Test Accuracy: {self.results['test_accuracy'][-1]:.4f}", 
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Final Test Loss: {self.results['test_loss'][-1]:.4f}", 
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if self.config.save_results:
            plt.savefig('brain_ai_mnist_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()


class ReinforcementLearningDemo:
    """Reinforcement learning demo using brain-inspired AI"""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model for RL
        self.model = create_brain_model(
            input_size=4,  # CartPole state space
            output_size=2,  # CartPole action space
            scale=config.model_scale
        ).to(self.device)
        
        # Create trainer
        learning_params = LearningParameters(
            dopamine_learning_rate=config.learning_rate,
            hebbian_learning_rate=config.learning_rate * 0.1
        )
        self.trainer = create_trainer(self.model, learning_params)
        
        # Environment (simplified CartPole)
        self.env = SimpleCartPole()
        
        self.results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': []
        }
    
    def train(self, num_episodes: int = 500):
        """Train using reinforcement learning"""
        print("Starting Reinforcement Learning with Brain-Inspired AI")
        print("=" * 50)
        
        for episode in tqdm(range(num_episodes), desc="RL Training"):
            state = self.env.reset()
            total_reward = 0
            episode_length = 0
            episode_losses = []
            
            for t in range(200):  # Max episode length
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get action probabilities
                with torch.no_grad():
                    outputs = self.model(state_tensor)
                    if 'actions' in outputs:
                        action_probs = outputs['actions'].cpu().numpy()[0]
                    else:
                        action_probs = np.array([0.5, 0.5])
                
                # Select action
                action = np.random.choice(2, p=action_probs)
                
                # Take action
                next_state, reward, done = self.env.step(action)
                
                # Convert to tensors
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                reward_tensor = torch.FloatTensor([reward]).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)
                
                # Training step
                metrics = self.trainer.train_step(
                    state_tensor, action_tensor, reward_signal=reward_tensor
                )
                
                episode_losses.append(metrics['loss'])
                
                state = next_state
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            self.results['episode_rewards'].append(total_reward)
            self.results['episode_lengths'].append(episode_length)
            self.results['losses'].append(np.mean(episode_losses))
            
            if episode % 50 == 0:
                avg_reward = np.mean(self.results['episode_rewards'][-50:])
                print(f"Episode {episode}: Average Reward (last 50): {avg_reward:.2f}")
        
        return self.results
    
    def visualize_results(self):
        """Visualize RL training results"""
        if not self.config.visualize:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.results['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Moving average of rewards
        window = 50
        if len(self.results['episode_rewards']) >= window:
            moving_avg = np.convolve(self.results['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.results['episode_lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Losses
        axes[1, 1].plot(self.results['losses'])
        axes[1, 1].set_title('Training Losses')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if self.config.save_results:
            plt.savefig('brain_ai_rl_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()


class SimpleCartPole:
    """Simplified CartPole environment for demonstration"""
    
    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.mass_pole * self.length * theta_dot ** 2 * sintheta) / (self.mass_cart + self.mass_pole)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.mass_pole * costheta ** 2 / (self.mass_cart + self.mass_pole)))
        xacc = temp - self.mass_pole * self.length * thetaacc * costheta / (self.mass_cart + self.mass_pole)
        
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        done = abs(x) > 2.4 or abs(theta) > 0.2095
        reward = 1.0 if not done else 0.0
        
        return self.state, reward, done


class MemoryDemo:
    """Memory and pattern completion demo"""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model focused on hippocampal system
        self.model = create_brain_model(
            input_size=100,
            output_size=100,
            scale=config.model_scale
        ).to(self.device)
        
        self.patterns = []
        self.results = {
            'completion_accuracy': [],
            'storage_loss': []
        }
    
    def generate_patterns(self, num_patterns: int = 50, pattern_size: int = 100):
        """Generate random patterns for memory testing"""
        self.patterns = []
        for _ in range(num_patterns):
            pattern = torch.randn(pattern_size)
            # Make patterns more structured
            pattern = torch.where(pattern > 0, torch.ones_like(pattern), torch.zeros_like(pattern))
            self.patterns.append(pattern)
    
    def test_pattern_completion(self):
        """Test pattern completion ability"""
        print("Testing Pattern Completion with Hippocampal System")
        print("=" * 50)
        
        self.generate_patterns()
        
        for i, pattern in enumerate(self.patterns):
            # Create partial pattern (remove 50% of bits)
            partial_pattern = pattern.clone()
            mask = torch.rand_like(pattern) < 0.5
            partial_pattern[mask] = 0
            
            # Store pattern
            pattern_tensor = pattern.unsqueeze(0).to(self.device)
            outputs = self.model(pattern_tensor)
            
            # Test completion
            partial_tensor = partial_pattern.unsqueeze(0).to(self.device)
            completion_outputs = self.model(partial_tensor)
            
            if 'memory' in completion_outputs:
                completed_pattern = completion_outputs['memory'].squeeze(0)
                accuracy = (completed_pattern.sign() == pattern.sign()).float().mean().item()
                self.results['completion_accuracy'].append(accuracy)
            
            if i % 10 == 0:
                avg_accuracy = np.mean(self.results['completion_accuracy'][-10:]) if self.results['completion_accuracy'] else 0
                print(f"Pattern {i+1}: Completion Accuracy = {avg_accuracy:.4f}")
        
        return self.results
    
    def visualize_results(self):
        """Visualize memory demo results"""
        if not self.config.visualize or not self.results['completion_accuracy']:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['completion_accuracy'])
        plt.title('Pattern Completion Accuracy')
        plt.xlabel('Pattern Index')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        if self.config.save_results:
            plt.savefig('brain_ai_memory_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()


class BenchmarkSuite:
    """Comprehensive benchmark suite for brain-inspired AI"""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.results = {}
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("Running Comprehensive Benchmark Suite")
        print("=" * 60)
        
        # MNIST Classification
        print("\n1. MNIST Classification Benchmark")
        mnist_demo = MNISTDemo(self.config)
        mnist_results = mnist_demo.train()
        mnist_demo.visualize_results()
        self.results['mnist'] = mnist_results
        
        # Reinforcement Learning
        print("\n2. Reinforcement Learning Benchmark")
        rl_demo = ReinforcementLearningDemo(self.config)
        rl_results = rl_demo.train(num_episodes=200)
        rl_demo.visualize_results()
        self.results['rl'] = rl_results
        
        # Memory and Pattern Completion
        print("\n3. Memory Pattern Completion Benchmark")
        memory_demo = MemoryDemo(self.config)
        memory_results = memory_demo.test_pattern_completion()
        memory_demo.visualize_results()
        self.results['memory'] = memory_results
        
        # Performance benchmarks
        print("\n4. Performance Benchmarks")
        self._run_performance_tests()
        
        self._generate_summary_report()
    
    def _run_performance_tests(self):
        """Run performance benchmarks"""
        model = create_brain_model(input_size=784, output_size=10, scale=self.config.model_scale)
        
        # Forward pass time
        x = torch.randn(32, 784)
        
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(x)
        forward_time = (time.time() - start_time) / 100
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory usage (approximate)
        param_memory = total_params * 4 / (1024**2)  # Assuming float32
        
        self.results['performance'] = {
            'forward_time_ms': forward_time * 1000,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_usage_mb': param_memory
        }
        
        print(f"Forward Pass Time: {forward_time*1000:.2f} ms")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Memory Usage: {param_memory:.2f} MB")
    
    def _generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        # MNIST results
        if 'mnist' in self.results:
            mnist_acc = self.results['mnist']['test_accuracy'][-1]
            mnist_loss = self.results['mnist']['test_loss'][-1]
            print(f"MNIST Classification:")
            print(f"  Final Test Accuracy: {mnist_acc:.4f}")
            print(f"  Final Test Loss: {mnist_loss:.4f}")
        
        # RL results
        if 'rl' in self.results:
            rl_rewards = self.results['rl']['episode_rewards']
            final_reward = np.mean(rl_rewards[-50:]) if len(rl_rewards) >= 50 else np.mean(rl_rewards)
            print(f"\nReinforcement Learning:")
            print(f"  Final Average Reward: {final_reward:.2f}")
            print(f"  Total Episodes: {len(rl_rewards)}")
        
        # Memory results
        if 'memory' in self.results:
            memory_acc = np.mean(self.results['memory']['completion_accuracy'])
            print(f"\nMemory Pattern Completion:")
            print(f"  Average Completion Accuracy: {memory_acc:.4f}")
        
        # Performance results
        if 'performance' in self.results:
            perf = self.results['performance']
            print(f"\nPerformance:")
            print(f"  Forward Pass Time: {perf['forward_time_ms']:.2f} ms")
            print(f"  Total Parameters: {perf['total_parameters']:,}")
            print(f"  Memory Usage: {perf['memory_usage_mb']:.2f} MB")
        
        print("\n" + "=" * 60)
        print("Brain-Inspired AI Benchmark Complete!")
        print("=" * 60)


def main():
    """Main demonstration function"""
    # Configuration
    config = DemoConfig(
        model_scale="small",  # Use "small" for faster demo
        batch_size=32,
        learning_rate=0.01,
        epochs=10,
        device="cpu",
        save_results=True,
        visualize=True
    )
    
    # Run comprehensive benchmarks
    benchmark_suite = BenchmarkSuite(config)
    benchmark_suite.run_all_benchmarks()


if __name__ == "__main__":
    main()
