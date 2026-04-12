"""
Brain-Inspired AI Architecture
A comprehensive neural network system modeled after the human brain's structure and neuronal diversity.

Based on the comprehensive neuron types report, this architecture implements:
- Multiple neuron types (excitatory, inhibitory, modulatory)
- Hierarchical organization (cortical layers, subcortical structures)
- Learning and memory systems
- Attention and action selection mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class NeuronParameters:
    """Parameters defining neuron behavior"""
    firing_threshold: float = 1.0
    resting_potential: float = 0.0
    membrane_time_constant: float = 20.0
    adaptation_strength: float = 0.1
    neurotransmitter_type: str = "glutamate"  # glutamate, GABA, ACh, DA, 5HT, NE
    connectivity_pattern: str = "local"  # local, long_range, specific


class BaseNeuron(nn.Module, ABC):
    """Base class for all neuron types"""
    
    def __init__(self, size: int, params: NeuronParameters):
        super().__init__()
        self.size = size
        self.params = params
        
        # Membrane potential
        self.potential = nn.Parameter(torch.zeros(size))
        
        # Adaptation variable
        self.adaptation = nn.Parameter(torch.zeros(size))
        
        # Synaptic weights
        self.weights = nn.Parameter(torch.randn(size, size) * 0.1)
        
    @abstractmethod
    def forward(self, x: torch.Tensor, adaptation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through neuron"""
        pass
    
    @abstractmethod
    def compute_firing_rate(self, potential: torch.Tensor) -> torch.Tensor:
        """Compute firing rate from membrane potential"""
        pass


class PyramidalNeuron(BaseNeuron):
    """Excitatory pyramidal neuron - most common cortical neuron"""
    
    def __init__(self, size: int, layer_depth: str = "superficial"):
        params = NeuronParameters(
            firing_threshold=1.2,
            adaptation_strength=0.15,
            neurotransmitter_type="glutamate",
            connectivity_pattern="long_range" if layer_depth == "deep" else "local"
        )
        super().__init__(size, params)
        self.layer_depth = layer_depth
        
        # Dendritic compartments
        self.apical_dendrite = nn.Parameter(torch.randn(size, size // 4) * 0.1)
        self.basal_dendrite = nn.Parameter(torch.randn(size, size // 4) * 0.1)
        
    def forward(self, x: torch.Tensor, adaptation: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input integration
        basal_input = torch.matmul(x, self.basal_dendrite)
        apical_input = torch.matmul(x, self.apical_dendrite)
        
        # Total input current
        total_input = basal_input + 0.5 * apical_input
        
        # Membrane dynamics
        if adaptation is not None:
            self.adaptation.data = 0.9 * self.adaptation.data + 0.1 * adaptation
            total_input = total_input - self.adaptation
        
        # Update potential
        self.potential.data = 0.95 * self.potential.data + total_input
        
        # Compute firing rate
        firing_rate = self.compute_firing_rate(self.potential)
        
        return firing_rate
    
    def compute_firing_rate(self, potential: torch.Tensor) -> torch.Tensor:
        return F.relu(potential - self.params.firing_threshold)


class ParvalbuminInterneuron(BaseNeuron):
    """Fast-spiking inhibitory interneuron"""
    
    def __init__(self, size: int):
        params = NeuronParameters(
            firing_threshold=0.8,
            membrane_time_constant=10.0,
            adaptation_strength=0.02,
            neurotransmitter_type="GABA",
            connectivity_pattern="local"
        )
        super().__init__(size, params)
        
        # Fast-spiking properties
        self.fast_dynamics = nn.Parameter(torch.ones(size) * 0.1)
        
    def forward(self, x: torch.Tensor, adaptation: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Fast integration
        total_input = torch.matmul(x, self.weights)
        
        # Minimal adaptation for fast-spiking
        if adaptation is not None:
            total_input = total_input - 0.02 * adaptation
        
        # Fast membrane dynamics
        self.potential.data = 0.9 * self.potential.data + total_input * self.fast_dynamics
        
        # Fast firing rate computation
        firing_rate = self.compute_firing_rate(self.potential)
        
        return -firing_rate  # Inhibitory output
    
    def compute_firing_rate(self, potential: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((potential - self.params.firing_threshold) * 5)


class SomatostatinInterneuron(BaseNeuron):
    """Dendrite-targeting inhibitory interneuron"""
    
    def __init__(self, size: int):
        params = NeuronParameters(
            firing_threshold=1.0,
            adaptation_strength=0.1,
            neurotransmitter_type="GABA",
            connectivity_pattern="local"
        )
        super().__init__(size, params)
        
    def forward(self, x: torch.Tensor, adaptation: Optional[torch.Tensor] = None) -> torch.Tensor:
        total_input = torch.matmul(x, self.weights)
        
        if adaptation is not None:
            self.adaptation.data = 0.85 * self.adaptation.data + 0.15 * adaptation
            total_input = total_input - self.adaptation
        
        self.potential.data = 0.92 * self.potential.data + total_input
        
        firing_rate = self.compute_firing_rate(self.potential)
        
        return -firing_rate  # Inhibitory output
    
    def compute_firing_rate(self, potential: torch.Tensor) -> torch.Tensor:
        return F.relu(potential - self.params.firing_threshold)


class DopaminergicNeuron(BaseNeuron):
    """Modulatory dopaminergic neuron"""
    
    def __init__(self, size: int):
        params = NeuronParameters(
            firing_threshold=0.5,
            neurotransmitter_type="DA",
            connectivity_pattern="diffuse"
        )
        super().__init__(size, params)
        
        # Reward prediction error signal
        self.reward_prediction = nn.Parameter(torch.zeros(size))
        
    def forward(self, x: torch.Tensor, reward_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute reward prediction error
        if reward_signal is not None:
            prediction_error = reward_signal - self.reward_prediction
            self.reward_prediction.data = 0.9 * self.reward_prediction.data + 0.1 * reward_signal
        else:
            prediction_error = 0
        
        # Modulatory output
        total_input = torch.matmul(x, self.weights) + prediction_error
        self.potential.data = 0.95 * self.potential.data + total_input
        
        firing_rate = self.compute_firing_rate(self.potential)
        
        return firing_rate  # Modulatory output (positive or negative)
    
    def compute_firing_rate(self, potential: torch.Tensor) -> torch.Tensor:
        return torch.tanh(potential)


class CorticalColumn(nn.Module):
    """Six-layer cortical column with diverse neuron populations"""
    
    def __init__(self, input_size: int, column_size: int = 1000):
        super().__init__()
        self.column_size = column_size
        self.input_size = input_size
        
        # Layer-specific neuron populations
        self.layers = nn.ModuleDict({
            'L1': nn.ModuleDict({  # Molecular layer - mostly inhibitory
                'neurogliaform': ParvalbuminInterneuron(column_size // 20),
            }),
            'L2_3': nn.ModuleDict({  # Superficial pyramidal and interneurons
                'pyramidal_IT': PyramidalNeuron(column_size // 3, "superficial"),
                'PV_basket': ParvalbuminInterneuron(column_size // 10),
                'SOM_martinotti': SomatostatinInterneuron(column_size // 10),
            }),
            'L4': nn.ModuleDict({  # Input layer - spiny stellate and interneurons
                'spiny_stellate': PyramidalNeuron(column_size // 4, "superficial"),
                'PV_basket': ParvalbuminInterneuron(column_size // 15),
            }),
            'L5': nn.ModuleDict({  # Output layer - large pyramidal and interneurons
                'pyramidal_PT': PyramidalNeuron(column_size // 4, "deep"),
                'pyramidal_IT': PyramidalNeuron(column_size // 8, "deep"),
                'PV_basket': ParvalbuminInterneuron(column_size // 12),
                'SOM_martinotti': SomatostatinInterneuron(column_size // 12),
            }),
            'L6': nn.ModuleDict({  # Corticothalamic layer
                'pyramidal_CT': PyramidalNeuron(column_size // 5, "deep"),
                'PV_basket': ParvalbuminInterneuron(column_size // 20),
            }),
        })
        
        # Inter-layer connections
        self.interlayer_connections = nn.ParameterDict({
            'L4_to_L2_3': nn.Parameter(torch.randn(column_size // 4, column_size // 3) * 0.1),
            'L2_3_to_L5': nn.Parameter(torch.randn(column_size // 3, column_size // 4) * 0.1),
            'L5_to_L6': nn.Parameter(torch.randn(column_size // 4, column_size // 5) * 0.1),
            'L6_to_L4': nn.Parameter(torch.randn(column_size // 5, column_size // 4) * 0.1),
        })
        
        # Input projection to L4
        self.input_projection = nn.Linear(input_size, column_size // 4)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        layer_outputs = {}
        
        # Input to L4
        l4_input = self.input_projection(x)
        
        # Process each layer
        for layer_name, neurons in self.layers.items():
            layer_input = l4_input if layer_name == 'L4' else torch.zeros_like(next(iter(neurons.values())).potential)
            
            # Get inputs from previous layers
            if layer_name == 'L2_3':
                l4_output = layer_outputs['L4']['spiny_stellate']
                layer_input = torch.matmul(l4_output, self.interlayer_connections['L4_to_L2_3'])
            elif layer_name == 'L5':
                l23_output = layer_outputs['L2_3']['pyramidal_IT']
                layer_input = torch.matmul(l23_output, self.interlayer_connections['L2_3_to_L5'])
            elif layer_name == 'L6':
                l5_output = layer_outputs['L5']['pyramidal_PT']
                layer_input = torch.matmul(l5_output, self.interlayer_connections['L5_to_L6'])
            
            # Process neurons in this layer
            layer_outputs[layer_name] = {}
            for neuron_name, neuron in neurons.items():
                layer_outputs[layer_name][neuron_name] = neuron(layer_input)
        
        # Feedback from L6 to L4
        l6_output = layer_outputs['L6']['pyramidal_CT']
        l4_feedback = torch.matmul(l6_output, self.interlayer_connections['L6_to_L4'])
        layer_outputs['L4']['spiny_stellate'] = layer_outputs['L4']['spiny_stellate'] + l4_feedback
        
        return layer_outputs


class Thalamus(nn.Module):
    """Thalamic relay and attention system"""
    
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        
        # Relay neurons
        self.relay_neurons = PyramidalNeuron(size)
        
        # Attention modulation
        self.attention_gate = nn.Parameter(torch.ones(size))
        
        # Thalamocortical and corticothalamic connections
        self.thalamocortical_weights = nn.Parameter(torch.randn(size, size) * 0.1)
        self.corticothalamic_weights = nn.Parameter(torch.randn(size, size) * 0.1)
        
    def forward(self, sensory_input: torch.Tensor, cortical_feedback: Optional[torch.Tensor] = None, 
                attention_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Apply attention modulation
        if attention_signal is not None:
            gated_input = sensory_input * self.attention_gate * attention_signal
        else:
            gated_input = sensory_input * self.attention_gate
        
        # Add cortical feedback
        if cortical_feedback is not None:
            feedback = torch.matmul(cortical_feedback, self.corticothalamic_weights)
            gated_input = gated_input + feedback
        
        # Relay processing
        relay_output = self.relay_neurons(gated_input)
        
        # Thalamocortical output
        thalamocortical_output = torch.matmul(relay_output, self.thalamocortical_weights)
        
        return thalamocortical_output


class Hippocampus(nn.Module):
    """Hippocampal memory system with pattern separation and completion"""
    
    def __init__(self, input_size: int, memory_size: int = 1000):
        super().__init__()
        self.memory_size = memory_size
        
        # Dentate gyrus - pattern separation
        self.dentate_granule = PyramidalNeuron(memory_size // 2)
        self.dentate_mossy = nn.Parameter(torch.randn(memory_size // 2, memory_size // 2) * 0.1)
        
        # CA3 - autoassociation
        self.ca3_pyramidal = PyramidalNeuron(memory_size // 3)
        self.ca3_recurrent = nn.Parameter(torch.randn(memory_size // 3, memory_size // 3) * 0.1)
        
        # CA1 - output
        self.ca1_pyramidal = PyramidalNeuron(memory_size // 3)
        
        # Entorhinal cortex connections
        self.ec_to_dentate = nn.Linear(input_size, memory_size // 2)
        self.ca3_to_ca1 = nn.Parameter(torch.randn(memory_size // 3, memory_size // 3) * 0.1)
        self.ca1_output = nn.Linear(memory_size // 3, input_size)
        
    def forward(self, x: torch.Tensor, mode: str = "encoding") -> torch.Tensor:
        # Entorhinal input to dentate gyrus
        dentate_input = self.ec_to_dentate(x)
        
        # Pattern separation in dentate gyrus
        dentate_output = self.dentate_granule(dentate_input)
        mossy_input = torch.matmul(dentate_output, self.dentate_mossy)
        
        # CA3 autoassociation
        ca3_input = mossy_input
        if mode == "retrieval":
            # Add recurrent input for pattern completion
            ca3_recurrent_input = torch.matmul(self.ca3_pyramidal.potential, self.ca3_recurrent)
            ca3_input = ca3_input + ca3_recurrent
        
        ca3_output = self.ca3_pyramidal(ca3_input)
        
        # CA1 output
        ca1_input = torch.matmul(ca3_output, self.ca3_to_ca1)
        ca1_output = self.ca1_pyramidal(ca1_input)
        
        # Output back to entorhinal cortex
        output = self.ca1_output(ca1_output)
        
        return output


class BasalGanglia(nn.Module):
    """Action selection and reinforcement learning system"""
    
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.action_size = action_size
        
        # Striatum medium spiny neurons
        self.striatum_D1 = PyramidalNeuron(action_size)  # Direct pathway
        self.striatum_D2 = PyramidalNeuron(action_size)  # Indirect pathway
        
        # Dopaminergic modulation
        self.dopamine_system = DopaminergicNeuron(action_size)
        
        # Output nuclei
        self.gpi_output = nn.Parameter(torch.randn(action_size) * 0.1)
        self.snr_output = nn.Parameter(torch.randn(action_size) * 0.1)
        
        # Action selection weights
        self.action_weights = nn.Parameter(torch.randn(action_size, input_size) * 0.1)
        
    def forward(self, x: torch.Tensor, reward_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute action values
        action_values = torch.matmul(x, self.action_weights.T)
        
        # Direct pathway (facilitates action)
        direct_path = self.striatum_D1(action_values)
        
        # Indirect pathway (inhibits competing actions)
        indirect_path = self.striatum_D2(action_values)
        
        # Dopaminergic modulation
        dopamine_signal = self.dopamine_system(action_values, reward_signal)
        
        # Action selection
        selected_actions = direct_path - indirect_path + dopamine_signal
        
        # Normalize to action probabilities
        action_probabilities = F.softmax(selected_actions, dim=-1)
        
        return action_probabilities


class Cerebellum(nn.Module):
    """Motor coordination and error-based learning system"""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        
        # Granule cells (input processing)
        self.granule_cells = PyramidalNeuron(input_size * 3)  # Mossy fiber expansion
        
        # Purkinje cells (output)
        self.purkinje_cells = PyramidalNeuron(output_size)
        
        # Deep cerebellar nuclei
        self.dcn_neurons = PyramidalNeuron(output_size)
        
        # Climbing fiber (error signal)
        self.climbing_fiber = nn.Parameter(torch.randn(output_size) * 0.1)
        
        # Mossy fiber to granule cell expansion
        self.mossy_to_granule = nn.Linear(input_size, input_size * 3)
        
        # Parallel fiber to Purkinje connections
        self.parallel_to_purkinje = nn.Parameter(torch.randn(input_size * 3, output_size) * 0.1)
        
    def forward(self, x: torch.Tensor, error_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Mossy fiber input to granule cells
        granule_input = self.mossy_to_granule(x)
        granule_output = self.granule_cells(granule_input)
        
        # Parallel fiber input to Purkinje cells
        purkinje_input = torch.matmul(granule_output, self.parallel_to_purkinje)
        
        # Add climbing fiber error signal
        if error_signal is not None:
            purkinje_input = purkinje_input + self.climbing_fiber * error_signal
        
        purkinje_output = self.purkinje_cells(purkinje_input)
        
        # Deep cerebellar nuclei (motor output)
        dcn_input = purkinje_output  # Inhibitory input from Purkinje cells
        motor_output = self.dcn_neurons(-dcn_input)  # Disinhibition
        
        return motor_output


class BrainInspiredAI(nn.Module):
    """Complete brain-inspired AI system"""
    
    def __init__(self, input_size: int, output_size: int, column_size: int = 1000):
        super().__init__()
        
        # Core brain systems
        self.thalamus = Thalamus(column_size // 2)
        self.cortex = CorticalColumn(input_size, column_size)
        self.hippocampus = Hippocampus(column_size // 4, column_size // 2)
        self.basal_ganglia = BasalGanglia(column_size // 4, output_size)
        self.cerebellum = Cerebellum(column_size // 4, output_size)
        
        # Global neuromodulatory systems
        self.global_dopamine = DopaminergicNeuron(column_size // 10)
        self.global_acetylcholine = DopaminergicNeuron(column_size // 10)  # Simplified
        
        # Attention system
        self.attention_weights = nn.Parameter(torch.ones(column_size // 2))
        
    def forward(self, x: torch.Tensor, reward_signal: Optional[torch.Tensor] = None, 
                error_signal: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        # Attention modulation
        attention_signal = torch.sigmoid(self.attention_weights)
        
        # Thalamic processing
        thalamic_output = self.thalamus(x, attention_signal=attention_signal)
        
        # Cortical processing
        cortical_outputs = self.cortex(thalamic_output)
        
        # Combine cortical outputs for higher systems
        cortical_activity = torch.cat([
            cortical_outputs['L2_3']['pyramidal_IT'],
            cortical_outputs['L5']['pyramidal_PT']
        ], dim=-1)
        
        # Hippocampal memory
        memory_output = self.hippocampus(cortical_activity)
        
        # Basal ganglia action selection
        action_probabilities = self.basal_ganglia(cortical_activity, reward_signal)
        
        # Cerebellar motor coordination
        motor_output = self.cerebellum(cortical_activity, error_signal)
        
        # Global neuromodulation
        dopamine_signal = self.global_dopamine(cortical_activity, reward_signal)
        ach_signal = self.global_acetylcholine(cortical_activity)
        
        outputs.update({
            'cortical': cortical_outputs,
            'thalamic': thalamic_output,
            'memory': memory_output,
            'actions': action_probabilities,
            'motor': motor_output,
            'dopamine': dopamine_signal,
            'acetylcholine': ach_signal
        })
        
        return outputs


def create_brain_model(input_size: int, output_size: int, scale: str = "medium") -> BrainInspiredAI:
    """Factory function to create brain-inspired models of different scales"""
    
    if scale == "small":
        column_size = 500
    elif scale == "medium":
        column_size = 1000
    elif scale == "large":
        column_size = 2000
    else:
        raise ValueError("Scale must be 'small', 'medium', or 'large'")
    
    return BrainInspiredAI(input_size, output_size, column_size)


if __name__ == "__main__":
    # Example usage
    model = create_brain_model(input_size=784, output_size=10, scale="medium")
    
    # Test with random input
    x = torch.randn(32, 784)  # Batch of 32, input size 784
    reward = torch.randn(32)   # Reward signal
    error = torch.randn(32, 10)  # Error signal
    
    outputs = model(x, reward_signal=reward, error_signal=error)
    
    print("Brain-inspired AI Model Outputs:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} components")
        else:
            print(f"{key}: {value.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
