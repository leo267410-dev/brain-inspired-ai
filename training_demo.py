"""
Enhanced Training Demo for Brain-Inspired AI
Shows actual learning and adaptation over time.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional


class EnhancedNeuron:
    """Enhanced neuron model with better dynamics"""
    
    def __init__(self, size: int, neuron_type: str = "pyramidal"):
        self.size = size
        self.neuron_type = neuron_type
        self.potential = np.random.uniform(0.1, 0.5, size)  # Start with some activity
        self.weights = np.random.uniform(-0.5, 0.5, (size, size))
        
        # Neuron type specific properties
        if neuron_type == "pyramidal":
            self.threshold = 0.8
            self.adaptation = 0.15
            self.learning_rate = 0.01
        elif neuron_type == "PV_interneuron":
            self.threshold = 0.4
            self.adaptation = 0.02
            self.learning_rate = 0.02
        elif neuron_type == "SOM_interneuron":
            self.threshold = 0.6
            self.adaptation = 0.1
            self.learning_rate = 0.015
        else:
            self.threshold = 0.5
            self.adaptation = 0.1
            self.learning_rate = 0.01
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass with better activation"""
        outputs = []
        for i in range(self.size):
            # Compute weighted sum
            total_input = 0.0
            for j in range(min(len(inputs), len(self.weights))):
                if i < len(self.weights[j]):
                    total_input += inputs[j] * self.weights[j][i]
            
            # Add some noise for realism
            total_input += np.random.normal(0, 0.1)
            
            # Update potential with adaptation
            self.potential[i] = 0.9 * self.potential[i] + total_input
            
            # Compute firing rate with better activation function
            if self.potential[i] > self.threshold:
                firing_rate = math.tanh(self.potential[i] - self.threshold)
            else:
                firing_rate = 0.0
            
            # Apply inhibitory effect for interneurons
            if "interneuron" in self.neuron_type:
                firing_rate = -firing_rate * 0.8  # Slightly weaker inhibition
            
            outputs.append(firing_rate)
        
        return outputs
    
    def learn(self, inputs: List[float], outputs: List[float], reward: float):
        """Simple Hebbian learning with reward modulation"""
        for i in range(self.size):
            for j in range(min(len(inputs), len(self.weights))):
                if i < len(self.weights[j]):
                    # Hebbian rule: pre * post * reward
                    delta_w = self.learning_rate * inputs[j] * outputs[i] * reward
                    self.weights[j][i] += delta_w
                    
                    # Keep weights bounded
                    self.weights[j][i] = max(-1.0, min(1.0, self.weights[j][i]))


class EnhancedBrainAI:
    """Enhanced brain-inspired AI with learning capabilities"""
    
    def __init__(self, input_size: int = 100, output_size: int = 10):
        self.input_size = input_size
        self.output_size = output_size
        
        # Brain components with enhanced neurons
        self.cortex = self._create_cortical_column(input_size)
        self.hippocampus = EnhancedNeuron(input_size // 2, "pyramidal")
        self.basal_ganglia = EnhancedNeuron(output_size, "pyramidal")
        
        # Training history
        self.training_history = {
            'rewards': [],
            'actions': [],
            'cortical_activity': [],
            'memory_activity': []
        }
    
    def _create_cortical_column(self, input_size: int) -> Dict[str, Dict[str, EnhancedNeuron]]:
        """Create enhanced cortical column"""
        return {
            'L1': {'neurogliaform': EnhancedNeuron(input_size // 20, 'PV_interneuron')},
            'L2_3': {
                'pyramidal_IT': EnhancedNeuron(input_size // 3, 'pyramidal'),
                'PV_basket': EnhancedNeuron(input_size // 10, 'PV_interneuron'),
                'SOM_martinotti': EnhancedNeuron(input_size // 10, 'SOM_interneuron'),
            },
            'L4': {
                'spiny_stellate': EnhancedNeuron(input_size // 4, 'pyramidal'),
                'PV_basket': EnhancedNeuron(input_size // 15, 'PV_interneuron'),
            },
            'L5': {
                'pyramidal_PT': EnhancedNeuron(input_size // 4, 'pyramidal'),
                'pyramidal_IT': EnhancedNeuron(input_size // 8, 'pyramidal'),
                'PV_basket': EnhancedNeuron(input_size // 12, 'PV_interneuron'),
                'SOM_martinotti': EnhancedNeuron(input_size // 12, 'SOM_interneuron'),
            },
            'L6': {
                'pyramidal_CT': EnhancedNeuron(input_size // 5, 'pyramidal'),
                'PV_basket': EnhancedNeuron(input_size // 20, 'PV_interneuron'),
            },
        }
    
    def forward(self, inputs: List[float]) -> Dict[str, List[float]]:
        """Forward pass through brain system"""
        outputs = {}
        
        # Cortical processing
        cortical_outputs = {}
        current_input = inputs
        
        for layer_name, neurons in self.cortex.items():
            cortical_outputs[layer_name] = {}
            for neuron_name, neuron in neurons.items():
                layer_output = neuron.forward(current_input)
                cortical_outputs[layer_name][neuron_name] = layer_output
            
            # Use output from pyramidal neurons for next layer
            pyramidal_outputs = [output for name, output in cortical_outputs[layer_name].items() 
                                if 'pyramidal' in name]
            if pyramidal_outputs:
                current_input = pyramidal_outputs[0]
        
        # Combine cortical outputs
        l23_output = cortical_outputs['L2_3']['pyramidal_IT']
        l5_output = cortical_outputs['L5']['pyramidal_PT']
        combined_cortical = l23_output + l5_output[:len(l23_output)]
        
        # Hippocampal memory
        memory_output = self.hippocampus.forward(combined_cortical[:len(combined_cortical)//2])
        
        # Basal ganglia action selection
        action_values = self.basal_ganglia.forward(combined_cortical[:len(combined_cortical)//2])
        
        # Convert to action probabilities
        exp_values = [math.exp(action) for action in action_values]
        total_exp = sum(exp_values)
        action_probabilities = [exp_val / total_exp for exp_val in exp_values] if total_exp > 0 else [1.0/len(action_values)] * len(action_values)
        
        outputs.update({
            'cortical': cortical_outputs,
            'memory': memory_output,
            'actions': action_probabilities
        })
        
        return outputs
    
    def train_step(self, inputs: List[float], reward: float) -> Dict[str, float]:
        """Single training step with learning"""
        # Forward pass
        outputs = self.forward(inputs)
        
        # Get best action
        best_action = outputs['actions'].index(max(outputs['actions']))
        best_prob = max(outputs['actions'])
        
        # Learning in all components
        # Cortical learning
        for layer_name, layer_outputs in outputs['cortical'].items():
            for neuron_name, neuron_output in layer_outputs.items():
                if 'pyramidal' in neuron_name:  # Only train pyramidal neurons
                    self.cortex[layer_name][neuron_name].learn(inputs, neuron_output, reward)
        
        # Hippocampal learning
        l23_output = outputs['cortical']['L2_3']['pyramidal_IT']
        self.hippocampus.learn(l23_output[:len(l23_output)//2], outputs['memory'], reward)
        
        # Basal ganglia learning
        self.basal_ganglia.learn(l23_output[:len(l23_output)//2], outputs['actions'], reward)
        
        # Record training history
        self.training_history['rewards'].append(reward)
        self.training_history['actions'].append(best_action)
        
        # Calculate average cortical activity
        total_cortical_activity = 0
        neuron_count = 0
        for layer_outputs in outputs['cortical'].values():
            for neuron_output in layer_outputs.values():
                total_cortical_activity += sum(neuron_output) / len(neuron_output)
                neuron_count += 1
        
        avg_cortical_activity = total_cortical_activity / neuron_count
        self.training_history['cortical_activity'].append(avg_cortical_activity)
        
        avg_memory_activity = sum(outputs['memory']) / len(outputs['memory'])
        self.training_history['memory_activity'].append(avg_memory_activity)
        
        return {
            'best_action': best_action,
            'best_prob': best_prob,
            'avg_cortical_activity': avg_cortical_activity,
            'avg_memory_activity': avg_memory_activity
        }


def create_training_task() -> Tuple[List[List[float]], List[float], List[int]]:
    """Create a simple classification task"""
    # Generate patterns for 3 different classes
    patterns = []
    labels = []
    
    # Class 0: Positive pattern
    for _ in range(30):
        pattern = [random.uniform(0.5, 1.0) if i % 10 < 5 else random.uniform(-1.0, -0.5) for i in range(100)]
        patterns.append(pattern)
        labels.append(0)
    
    # Class 1: Negative pattern  
    for _ in range(30):
        pattern = [random.uniform(-1.0, -0.5) if i % 10 < 5 else random.uniform(0.5, 1.0) for i in range(100)]
        patterns.append(pattern)
        labels.append(1)
    
    # Class 2: Mixed pattern
    for _ in range(40):
        pattern = [random.uniform(-0.2, 0.8) for _ in range(100)]
        labels.append(2)
    
    return patterns, labels


def train_brain_ai():
    """Train the brain-inspired AI on a classification task"""
    print("Brain-Inspired AI Training Session")
    print("=" * 50)
    
    # Create model
    model = EnhancedBrainAI(input_size=100, output_size=10)
    
    # Create training data
    patterns, labels = create_training_task()
    
    # Training parameters
    epochs = 100
    print(f"Training for {epochs} epochs on {len(patterns)} patterns")
    
    # Training loop
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_accuracy = []
        
        # Shuffle data
        indices = list(range(len(patterns)))
        random.shuffle(indices)
        
        for i, idx in enumerate(indices):
            pattern = patterns[idx]
            label = labels[idx]
            
            # Forward pass
            outputs = model.forward(pattern)
            
            # Determine reward (1 if correct action, 0 if wrong)
            best_action = outputs['actions'].index(max(outputs['actions']))
            reward = 1.0 if best_action == label else 0.0
            
            # Training step
            metrics = model.train_step(pattern, reward)
            
            epoch_rewards.append(reward)
            epoch_accuracy.append(reward)
        
        # Calculate epoch metrics
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
        avg_cortical = sum(model.training_history['cortical_activity'][-len(patterns):]) / len(patterns)
        avg_memory = sum(model.training_history['memory_activity'][-len(patterns):]) / len(patterns)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}: Reward = {avg_reward:.3f}, "
                  f"Accuracy = {avg_accuracy:.3f}, "
                  f"Cortical = {avg_cortical:.3f}, "
                  f"Memory = {avg_memory:.3f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_rewards = model.training_history['rewards'][-len(patterns):]
    final_accuracy = sum(final_rewards) / len(final_rewards)
    print(f"Final Accuracy: {final_accuracy:.3f}")
    
    # Text-based analysis of training results
    print("\nTraining Analysis:")
    print("-" * 30)
    
    # Overall statistics
    total_steps = len(model.training_history['rewards'])
    final_accuracy = sum(model.training_history['rewards'][-len(patterns):]) / len(patterns)
    overall_reward = sum(model.training_history['rewards']) / total_steps
    
    print(f"Total training steps: {total_steps}")
    print(f"Final accuracy: {final_accuracy:.3f}")
    print(f"Overall average reward: {overall_reward:.3f}")
    
    # Learning progress (first vs last 10%)
    first_10_percent = int(total_steps * 0.1)
    last_10_percent = int(total_steps * 0.9)
    
    early_reward = sum(model.training_history['rewards'][:first_10_percent]) / first_10_percent
    late_reward = sum(model.training_history['rewards'][last_10_percent:]) / (total_steps - last_10_percent)
    
    print(f"Early accuracy (first 10%): {early_reward:.3f}")
    print(f"Late accuracy (last 10%): {late_reward:.3f}")
    print(f"Improvement: {late_reward - early_reward:+.3f}")
    
    # Activity statistics
    avg_cortical = sum(model.training_history['cortical_activity']) / len(model.training_history['cortical_activity'])
    avg_memory = sum(model.training_history['memory_activity']) / len(model.training_history['memory_activity'])
    
    print(f"Average cortical activity: {avg_cortical:.3f}")
    print(f"Average memory activity: {avg_memory:.3f}")
    
    # Action distribution
    action_counts = [0] * 10
    for action in model.training_history['actions']:
        if action < len(action_counts):
            action_counts[action] += 1
    
    print("\nAction Selection Distribution:")
    for i, count in enumerate(action_counts):
        if count > 0:
            percentage = (count / total_steps) * 100
            print(f"  Action {i}: {count} times ({percentage:.1f}%)")
    
    # Moving average analysis
    window = 50
    if len(model.training_history['rewards']) >= window:
        moving_averages = []
        for i in range(len(model.training_history['rewards']) - window + 1):
            avg = sum(model.training_history['rewards'][i:i+window]) / window
            moving_averages.append(avg)
        
        print(f"\nMoving Average Analysis (window={window}):")
        print(f"  Initial MA: {moving_averages[0]:.3f}")
        print(f"  Final MA: {moving_averages[-1]:.3f}")
        print(f"  Peak MA: {max(moving_averages):.3f}")
    
    print("\nTraining completed successfully!")
    
    return model, model.training_history


if __name__ == "__main__":
    trained_model, history = train_brain_ai()
    print("\nTraining Complete!")
    print(f"Final model achieved learning across {len(history['rewards'])} training steps")
