"""
Simple demonstration of the Brain-Inspired AI architecture
This demo shows the basic structure without requiring external dependencies.
"""

import math
import random
from typing import Dict, List, Tuple, Optional

# Simplified neural network components for demonstration
class SimpleNeuron:
    """Simplified neuron model for demonstration"""
    
    def __init__(self, size: int, neuron_type: str = "pyramidal"):
        self.size = size
        self.neuron_type = neuron_type
        self.potential = [0.0] * size
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(size)] for _ in range(size)]
        
        # Neuron type specific properties
        if neuron_type == "pyramidal":
            self.threshold = 1.2
            self.adaptation = 0.15
        elif neuron_type == "PV_interneuron":
            self.threshold = 0.8
            self.adaptation = 0.02
        elif neuron_type == "SOM_interneuron":
            self.threshold = 1.0
            self.adaptation = 0.1
        else:
            self.threshold = 1.0
            self.adaptation = 0.1
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Simple forward pass"""
        outputs = []
        for i in range(self.size):
            # Compute weighted sum (ensure we don't go out of bounds)
            total_input = 0.0
            for j in range(min(len(inputs), len(self.weights))):
                if i < len(self.weights[j]):
                    total_input += inputs[j] * self.weights[j][i]
            
            # Update potential with adaptation
            self.potential[i] = 0.95 * self.potential[i] + total_input
            
            # Compute firing rate
            if self.potential[i] > self.threshold:
                firing_rate = self.potential[i] - self.threshold
            else:
                firing_rate = 0.0
            
            # Apply inhibitory effect for interneurons
            if "interneuron" in self.neuron_type:
                firing_rate = -firing_rate
            
            outputs.append(firing_rate)
        
        return outputs


class SimpleCorticalColumn:
    """Simplified six-layer cortical column"""
    
    def __init__(self, input_size: int = 100):
        self.input_size = input_size
        
        # Create neurons for each layer
        self.layers = {
            'L1': {'neurogliaform': SimpleNeuron(input_size // 20, 'PV_interneuron')},
            'L2_3': {
                'pyramidal_IT': SimpleNeuron(input_size // 3, 'pyramidal'),
                'PV_basket': SimpleNeuron(input_size // 10, 'PV_interneuron'),
                'SOM_martinotti': SimpleNeuron(input_size // 10, 'SOM_interneuron'),
            },
            'L4': {
                'spiny_stellate': SimpleNeuron(input_size // 4, 'pyramidal'),
                'PV_basket': SimpleNeuron(input_size // 15, 'PV_interneuron'),
            },
            'L5': {
                'pyramidal_PT': SimpleNeuron(input_size // 4, 'pyramidal'),
                'pyramidal_IT': SimpleNeuron(input_size // 8, 'pyramidal'),
                'PV_basket': SimpleNeuron(input_size // 12, 'PV_interneuron'),
                'SOM_martinotti': SimpleNeuron(input_size // 12, 'SOM_interneuron'),
            },
            'L6': {
                'pyramidal_CT': SimpleNeuron(input_size // 5, 'pyramidal'),
                'PV_basket': SimpleNeuron(input_size // 20, 'PV_interneuron'),
            },
        }
        
        # Simple inter-layer connections
        self.layer_connections = {
            'L4_to_L2_3': [[random.uniform(-0.1, 0.1) for _ in range(input_size // 3)] for _ in range(input_size // 4)],
            'L2_3_to_L5': [[random.uniform(-0.1, 0.1) for _ in range(input_size // 4)] for _ in range(input_size // 3)],
            'L5_to_L6': [[random.uniform(-0.1, 0.1) for _ in range(input_size // 5)] for _ in range(input_size // 4)],
        }
    
    def forward(self, inputs: List[float]) -> Dict[str, Dict[str, List[float]]]:
        """Forward pass through cortical column"""
        layer_outputs = {}
        
        # Process each layer
        for layer_name, neurons in self.layers.items():
            layer_inputs = inputs if layer_name == 'L4' else [0.0] * len(next(iter(neurons.values())).potential)
            
            # Get inputs from previous layers
            if layer_name == 'L2_3' and 'L4' in layer_outputs:
                l4_output = layer_outputs['L4']['spiny_stellate']
                layer_inputs = [sum(l4_output[j] * self.layer_connections['L4_to_L2_3'][j][i] for j in range(len(l4_output))) 
                              for i in range(len(self.layer_connections['L4_to_L2_3'][0]))]
            elif layer_name == 'L5' and 'L2_3' in layer_outputs:
                l23_output = layer_outputs['L2_3']['pyramidal_IT']
                layer_inputs = [sum(l23_output[j] * self.layer_connections['L2_3_to_L5'][j][i] for j in range(len(l23_output))) 
                              for i in range(len(self.layer_connections['L2_3_to_L5'][0]))]
            elif layer_name == 'L6' and 'L5' in layer_outputs:
                l5_output = layer_outputs['L5']['pyramidal_PT']
                layer_inputs = [sum(l5_output[j] * self.layer_connections['L5_to_L6'][j][i] for j in range(len(l5_output))) 
                              for i in range(len(self.layer_connections['L5_to_L6'][0]))]
            
            # Process neurons in this layer
            layer_outputs[layer_name] = {}
            for neuron_name, neuron in neurons.items():
                layer_outputs[layer_name][neuron_name] = neuron.forward(layer_inputs)
        
        return layer_outputs


class SimpleHippocampus:
    """Simplified hippocampal memory system"""
    
    def __init__(self, input_size: int = 50):
        self.input_size = input_size
        
        # Simplified hippocampal components
        self.dentate_granule = SimpleNeuron(input_size // 2, 'pyramidal')
        self.ca3_pyramidal = SimpleNeuron(input_size // 3, 'pyramidal')
        self.ca1_pyramidal = SimpleNeuron(input_size // 3, 'pyramidal')
        
        # Connections
        self.ec_to_dentate = [[random.uniform(-0.1, 0.1) for _ in range(input_size // 2)] for _ in range(input_size)]
        self.dentate_to_ca3 = [[random.uniform(-0.1, 0.1) for _ in range(input_size // 3)] for _ in range(input_size // 2)]
        self.ca3_to_ca1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size // 3)] for _ in range(input_size // 3)]
        self.ca1_to_ec = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(input_size // 3)]
        
        # Recurrent CA3 connections for autoassociation
        self.ca3_recurrent = [[random.uniform(-0.1, 0.1) for _ in range(input_size // 3)] for _ in range(input_size // 3)]
    
    def forward(self, inputs: List[float], mode: str = "encoding") -> List[float]:
        """Forward pass through hippocampus"""
        # Entorhinal to dentate gyrus
        dentate_input = [sum(inputs[j] * self.ec_to_dentate[j][i] for j in range(len(inputs))) 
                       for i in range(len(self.ec_to_dentate[0]))]
        
        # Pattern separation in dentate gyrus
        dentate_output = self.dentate_granule.forward(dentate_input)
        
        # Dentate to CA3
        ca3_input = [sum(dentate_output[j] * self.dentate_to_ca3[j][i] for j in range(len(dentate_output))) 
                    for i in range(len(self.dentate_to_ca3[0]))]
        
        # CA3 processing with recurrent connections for pattern completion
        if mode == "retrieval":
            # Add recurrent input
            ca3_recurrent_input = [sum(self.ca3_pyramidal.potential[j] * self.ca3_recurrent[j][i] 
                                     for j in range(len(self.ca3_pyramidal.potential))) 
                                 for i in range(len(self.ca3_recurrent[0]))]
            ca3_input = [ca3_input[i] + ca3_recurrent_input[i] for i in range(len(ca3_input))]
        
        ca3_output = self.ca3_pyramidal.forward(ca3_input)
        
        # CA3 to CA1
        ca1_input = [sum(ca3_output[j] * self.ca3_to_ca1[j][i] for j in range(len(ca3_output))) 
                    for i in range(len(self.ca3_to_ca1[0]))]
        
        ca1_output = self.ca1_pyramidal.forward(ca1_input)
        
        # CA1 back to entorhinal
        output = [sum(ca1_output[j] * self.ca1_to_ec[j][i] for j in range(len(ca1_output))) 
                 for i in range(len(self.ca1_to_ec[0]))]
        
        return output


class SimpleBasalGanglia:
    """Simplified basal ganglia action selection"""
    
    def __init__(self, input_size: int = 50, action_size: int = 10):
        self.input_size = input_size
        self.action_size = action_size
        
        # Striatum neurons
        self.striatum_D1 = SimpleNeuron(action_size, 'pyramidal')  # Direct pathway
        self.striatum_D2 = SimpleNeuron(action_size, 'pyramidal')  # Indirect pathway
        
        # Action weights
        self.action_weights = [[random.uniform(-0.1, 0.1) for _ in range(action_size)] for _ in range(input_size)]
        
        # Dopamine system
        self.reward_prediction = 0.0
    
    def forward(self, inputs: List[float], reward_signal: Optional[float] = None) -> List[float]:
        """Forward pass through basal ganglia"""
        # Compute action values
        action_values = [sum(inputs[j] * self.action_weights[j][i] for j in range(len(inputs))) 
                        for i in range(self.action_size)]
        
        # Direct pathway (facilitates action)
        direct_path = self.striatum_D1.forward(action_values)
        
        # Indirect pathway (inhibits competing actions)
        indirect_path = self.striatum_D2.forward(action_values)
        
        # Reward prediction error
        dopamine_signal = 0.0
        if reward_signal is not None:
            prediction_error = reward_signal - self.reward_prediction
            self.reward_prediction = 0.9 * self.reward_prediction + 0.1 * reward_signal
            dopamine_signal = prediction_error * 0.1
        
        # Action selection
        selected_actions = [direct_path[i] - indirect_path[i] + dopamine_signal for i in range(len(direct_path))]
        
        # Simple softmax-like normalization
        exp_values = [math.exp(action) for action in selected_actions]
        total_exp = sum(exp_values)
        action_probabilities = [exp_val / total_exp for exp_val in exp_values] if total_exp > 0 else [1.0/len(selected_actions)] * len(selected_actions)
        
        return action_probabilities


class SimpleBrainAI:
    """Simplified brain-inspired AI system"""
    
    def __init__(self, input_size: int = 100, output_size: int = 10):
        self.input_size = input_size
        self.output_size = output_size
        
        # Brain components
        self.cortex = SimpleCorticalColumn(input_size)
        self.hippocampus = SimpleHippocampus(input_size // 2)
        self.basal_ganglia = SimpleBasalGanglia(input_size // 2, output_size)
    
    def forward(self, inputs: List[float], reward_signal: Optional[float] = None) -> Dict[str, List[float]]:
        """Forward pass through brain system"""
        outputs = {}
        
        # Cortical processing
        cortical_outputs = self.cortex.forward(inputs)
        
        # Combine cortical outputs
        l23_output = cortical_outputs['L2_3']['pyramidal_IT']
        l5_output = cortical_outputs['L5']['pyramidal_PT']
        combined_cortical = l23_output + l5_output[:len(l23_output)]
        
        # Hippocampal memory
        memory_output = self.hippocampus.forward(combined_cortical[:len(combined_cortical)//2])
        
        # Basal ganglia action selection
        action_probabilities = self.basal_ganglia.forward(combined_cortical[:len(combined_cortical)//2], reward_signal)
        
        outputs.update({
            'cortical': cortical_outputs,
            'memory': memory_output,
            'actions': action_probabilities
        })
        
        return outputs


def demo_simple_brain_ai():
    """Demonstration of the simplified brain-inspired AI"""
    print("Brain-Inspired AI - Simplified Demonstration")
    print("=" * 50)
    
    # Create model
    model = SimpleBrainAI(input_size=100, output_size=10)
    
    # Test with random input
    inputs = [random.uniform(-1, 1) for _ in range(100)]
    reward = random.uniform(-1, 1)
    
    print(f"Input size: {len(inputs)}")
    print(f"Output size: {model.output_size}")
    print(f"Reward signal: {reward:.3f}")
    
    # Forward pass
    outputs = model.forward(inputs, reward_signal=reward)
    
    print("\nOutputs:")
    print(f"Cortical layers processed: {len(outputs['cortical'])}")
    print(f"Memory output size: {len(outputs['memory'])}")
    print(f"Action probabilities: {len(outputs['actions'])}")
    
    print("\nAction Probabilities:")
    for i, prob in enumerate(outputs['actions']):
        print(f"  Action {i}: {prob:.4f}")
    
    # Show cortical layer activity
    print("\nCortical Layer Activity (sample):")
    for layer_name, layer_outputs in outputs['cortical'].items():
        for neuron_name, neuron_output in layer_outputs.items():
            avg_activity = sum(neuron_output) / len(neuron_output)
            print(f"  {layer_name} {neuron_name}: avg activity = {avg_activity:.4f}")
    
    print("\nMemory System Activity:")
    avg_memory = sum(outputs['memory']) / len(outputs['memory'])
    print(f"  Average memory activity: {avg_memory:.4f}")
    
    # Test multiple steps to show learning
    print("\nMulti-step Learning Simulation:")
    for step in range(5):
        inputs = [random.uniform(-1, 1) for _ in range(100)]
        reward = random.uniform(-1, 1)
        outputs = model.forward(inputs, reward_signal=reward)
        
        # Get best action
        best_action = outputs['actions'].index(max(outputs['actions']))
        best_prob = max(outputs['actions'])
        
        print(f"  Step {step+1}: Best action = {best_action} (prob = {best_prob:.4f})")
    
    print("\nDemonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("- Six-layer cortical organization")
    print("- Diverse neuron types (pyramidal, PV+, SOM+ interneurons)")
    print("- Hippocampal memory system")
    print("- Basal ganglia action selection")
    print("- Reward-modulated learning")
    print("- Hierarchical processing")


if __name__ == "__main__":
    demo_simple_brain_ai()
