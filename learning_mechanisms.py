"""
Learning Mechanisms for Brain-Inspired AI
Implements various learning algorithms inspired by neuroplasticity in the brain.

Includes:
- Hebbian learning (spike-timing dependent plasticity)
- Dopamine-modulated reinforcement learning
- Error-based learning (cerebellar)
- Memory consolidation (hippocampal)
- Synaptic plasticity rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


@dataclass
class LearningParameters:
    """Parameters for different learning mechanisms"""
    hebbian_learning_rate: float = 0.01
    dopamine_learning_rate: float = 0.001
    cerebellar_learning_rate: float = 0.1
    memory_consolidation_rate: float = 0.005
    synaptic_decay_rate: float = 0.0001
    ltp_threshold: float = 1.0
    ltd_threshold: float = 0.5
    eligibility_trace_decay: float = 0.95


class LearningRule(ABC):
    """Base class for learning rules"""
    
    def __init__(self, params: LearningParameters):
        self.params = params
        
    @abstractmethod
    def update_weights(self, weights: torch.Tensor, pre_synaptic: torch.Tensor, 
                      post_synaptic: torch.Tensor, **kwargs) -> torch.Tensor:
        """Update synaptic weights based on activity"""
        pass


class HebbianLearning(LearningRule):
    """Hebbian learning with spike-timing dependent plasticity (STDP)"""
    
    def __init__(self, params: LearningParameters):
        super().__init__(params)
        self.eligibility_traces = {}
        
    def update_weights(self, weights: torch.Tensor, pre_synaptic: torch.Tensor, 
                      post_synaptic: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Implement STDP: 
        - Pre before Post: LTP (potentiation)
        - Post before Pre: LTD (depression)
        """
        
        # Get eligibility trace for this connection
        trace_key = kwargs.get('trace_key', 'default')
        if trace_key not in self.eligibility_traces:
            self.eligibility_traces[trace_key] = torch.zeros_like(weights)
        
        trace = self.eligibility_traces[trace_key]
        
        # Compute STDP update
        pre_spike = pre_synaptic > self.params.ltp_threshold
        post_spike = post_synaptic > self.params.ltp_threshold
        
        # LTP: pre before post
        ltp_mask = pre_spike.unsqueeze(-1) & post_spike.unsqueeze(0)
        ltp_update = self.params.hebbian_learning_rate * torch.where(ltp_mask, 
                                                                    pre_synaptic.unsqueeze(-1) * post_synaptic.unsqueeze(0), 
                                                                    torch.zeros_like(weights))
        
        # LTD: post before pre (simplified)
        ltd_mask = post_spike.unsqueeze(-1) & pre_spike.unsqueeze(0)
        ltd_update = -self.params.hebbian_learning_rate * 0.5 * torch.where(ltd_mask,
                                                                           pre_synaptic.unsqueeze(-1) * post_synaptic.unsqueeze(0),
                                                                           torch.zeros_like(weights))
        
        # Update eligibility trace
        trace.data = self.params.eligibility_trace_decay * trace.data + ltp_update + ltd_update
        self.eligibility_traces[trace_key] = trace
        
        # Apply weight changes
        weight_update = trace
        
        # Apply weight decay
        weight_update = weight_update - self.params.synaptic_decay_rate * weights
        
        # Clamp weights to prevent explosion
        new_weights = torch.clamp(weights + weight_update, -2.0, 2.0)
        
        return new_weights


class DopamineModulatedLearning(LearningRule):
    """Dopamine-modulated reinforcement learning (three-factor learning)"""
    
    def __init__(self, params: LearningParameters):
        super().__init__(params)
        self.reward_prediction_error = 0.0
        self.eligibility_traces = {}
        
    def update_weights(self, weights: torch.Tensor, pre_synaptic: torch.Tensor, 
                      post_synaptic: torch.Tensor, reward_signal: Optional[torch.Tensor] = None, 
                      **kwargs) -> torch.Tensor:
        """
        Three-factor learning rule:
        dW/dt = dopamine * eligibility_trace
        """
        
        trace_key = kwargs.get('trace_key', 'default')
        if trace_key not in self.eligibility_traces:
            self.eligibility_traces[trace_key] = torch.zeros_like(weights)
        
        trace = self.eligibility_traces[trace_key]
        
        # Compute eligibility trace (correlation of pre and post activity)
        correlation = pre_synaptic.unsqueeze(-1) * post_synaptic.unsqueeze(0)
        trace.data = self.params.eligibility_trace_decay * trace.data + correlation
        
        # Compute reward prediction error
        if reward_signal is not None:
            if isinstance(reward_signal, torch.Tensor):
                reward_value = reward_signal.mean().item()
            else:
                reward_value = reward_signal
            
            prediction_error = reward_value - self.reward_prediction_error
            self.reward_prediction_error = 0.9 * self.reward_prediction_error + 0.1 * reward_value
        else:
            prediction_error = 0.0
        
        # Three-factor learning: dopamine * eligibility
        weight_update = self.params.dopamine_learning_rate * prediction_error * trace
        
        # Apply weight changes
        new_weights = weights + weight_update
        
        return new_weights


class CerebellarLearning(LearningRule):
    """Error-based learning for cerebellar motor coordination"""
    
    def __init__(self, params: LearningParameters):
        super().__init__(params)
        self.error_history = []
        
    def update_weights(self, weights: torch.Tensor, pre_synaptic: torch.Tensor, 
                      post_synaptic: torch.Tensor, error_signal: Optional[torch.Tensor] = None, 
                      **kwargs) -> torch.Tensor:
        """
        Cerebellar learning rule:
        dW/dt = learning_rate * error * pre_synaptic_activity
        """
        
        if error_signal is None:
            return weights
        
        # Ensure error_signal has the right shape
        if error_signal.dim() == 1 and pre_synaptic.dim() == 1:
            error_expanded = error_signal.unsqueeze(-1)
            pre_expanded = pre_synaptic.unsqueeze(0)
        else:
            error_expanded = error_signal.unsqueeze(-1) if error_signal.dim() == 1 else error_signal
            pre_expanded = pre_synaptic.unsqueeze(0) if pre_synaptic.dim() == 1 else pre_synaptic
        
        # Compute weight update
        weight_update = self.params.cerebellar_learning_rate * error_expanded * pre_expanded
        
        # Apply weight changes
        new_weights = weights + weight_update
        
        # Store error for analysis
        self.error_history.append(error_signal.mean().item() if isinstance(error_signal, torch.Tensor) else error_signal)
        if len(self.error_history) > 1000:  # Keep only recent history
            self.error_history.pop(0)
        
        return new_weights


class HippocampalLearning(LearningRule):
    """Hippocampal memory consolidation and pattern completion"""
    
    def __init__(self, params: LearningParameters):
        super().__init__(params)
        self.memory_strength = {}
        self.consolidation_schedule = []
        
    def update_weights(self, weights: torch.Tensor, pre_synaptic: torch.Tensor, 
                      post_synaptic: torch.Tensor, mode: str = "encoding", 
                      **kwargs) -> torch.Tensor:
        """
        Hippocampal learning:
        - Encoding: strengthen connections for new memories
        - Consolidation: strengthen important connections over time
        - Retrieval: pattern completion through recurrent connections
        """
        
        if mode == "encoding":
            # Strengthen connections during encoding
            correlation = pre_synaptic.unsqueeze(-1) * post_synaptic.unsqueeze(0)
            weight_update = self.params.memory_consolidation_rate * correlation
            
        elif mode == "consolidation":
            # Gradual strengthening of important connections
            connection_strength = torch.abs(weights)
            importance_mask = connection_strength > torch.quantile(connection_strength, 0.8)
            weight_update = self.params.memory_consolidation_rate * 0.1 * importance_mask.float() * weights
            
        elif mode == "retrieval":
            # Pattern completion - minimal weight change
            weight_update = torch.zeros_like(weights)
            
        else:
            weight_update = torch.zeros_like(weights)
        
        # Apply weight changes
        new_weights = weights + weight_update
        
        return new_weights


class NeuromodulatorySystem(nn.Module):
    """Global neuromodulatory systems for learning regulation"""
    
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        
        # Dopamine system (reward, motivation)
        self.dopamine_baseline = nn.Parameter(torch.ones(size) * 0.1)
        self.dopamine_response = nn.Parameter(torch.ones(size))
        
        # Acetylcholine system (attention, learning rate)
        self.ach_baseline = nn.Parameter(torch.ones(size) * 0.1)
        self.ach_response = nn.Parameter(torch.ones(size))
        
        # Norepinephrine system (arousal, vigilance)
        self.ne_baseline = nn.Parameter(torch.ones(size) * 0.1)
        self.ne_response = nn.Parameter(torch.ones(size))
        
        # Serotonin system (mood, patience)
        self.serotonin_baseline = nn.Parameter(torch.ones(size) * 0.1)
        self.serotonin_response = nn.Parameter(torch.ones(size))
        
    def forward(self, reward_signal: Optional[torch.Tensor] = None, 
                attention_signal: Optional[torch.Tensor] = None,
                arousal_signal: Optional[torch.Tensor] = None,
                mood_signal: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        # Dopamine: reward prediction error
        if reward_signal is not None:
            dopamine_response = self.dopamine_baseline + self.dopamine_response * torch.sigmoid(reward_signal)
        else:
            dopamine_response = self.dopamine_baseline
        outputs['dopamine'] = dopamine_response
        
        # Acetylcholine: attention-driven
        if attention_signal is not None:
            ach_response = self.ach_baseline + self.ach_response * torch.sigmoid(attention_signal)
        else:
            ach_response = self.ach_baseline
        outputs['acetylcholine'] = ach_response
        
        # Norepinephrine: arousal-driven
        if arousal_signal is not None:
            ne_response = self.ne_baseline + self.ne_response * torch.sigmoid(arousal_signal)
        else:
            ne_response = self.ne_baseline
        outputs['norepinephrine'] = ne_response
        
        # Serotonin: mood-driven
        if mood_signal is not None:
            serotonin_response = self.serotonin_baseline + self.serotonin_response * torch.sigmoid(mood_signal)
        else:
            serotonin_response = self.serotonin_baseline
        outputs['serotonin'] = serotonin_response
        
        return outputs


class AdaptiveLearningRate(nn.Module):
    """Adaptive learning rate based on neuromodulation"""
    
    def __init__(self, base_lr: float = 0.01):
        super().__init__()
        self.base_lr = base_lr
        
        # Neuromodulatory influences on learning rate
        self.dopamine_influence = nn.Parameter(torch.tensor(1.0))
        self.ach_influence = nn.Parameter(torch.tensor(1.0))
        self.ne_influence = nn.Parameter(torch.tensor(0.5))
        self.serotonin_influence = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, neuromodulators: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute adaptive learning rate based on neuromodulatory state
        """
        
        dopamine = neuromodulators.get('dopamine', torch.tensor(1.0))
        ach = neuromodulators.get('acetylcholine', torch.tensor(1.0))
        ne = neuromodulators.get('norepinephrine', torch.tensor(1.0))
        serotonin = neuromodulators.get('serotonin', torch.tensor(1.0))
        
        # Compute adaptive learning rate
        adaptive_lr = self.base_lr * (
            1.0 + 
            self.dopamine_influence * dopamine.mean() +
            self.ach_influence * ach.mean() +
            self.ne_influence * ne.mean() +
            self.serotonin_influence * serotonin.mean()
        )
        
        return torch.clamp(adaptive_lr, 0.001, 0.1)


class ConsolidationScheduler:
    """Manages memory consolidation over time"""
    
    def __init__(self, consolidation_period: int = 100, decay_rate: float = 0.95):
        self.consolidation_period = consolidation_period
        self.decay_rate = decay_rate
        self.step_count = 0
        self.consolidation_history = []
        
    def should_consolidate(self) -> bool:
        """Check if consolidation should occur"""
        return self.step_count % self.consolidation_period == 0
    
    def update(self) -> bool:
        """Update step count and return if consolidation should happen"""
        self.step_count += 1
        return self.should_consolidate()
    
    def get_consolidation_strength(self) -> float:
        """Get current consolidation strength"""
        # Gradually increase consolidation strength over time
        base_strength = 0.1
        growth_factor = 1.0 + (self.step_count / self.consolidation_period) * 0.1
        return base_strength * growth_factor


class BrainInspiredTrainer:
    """Training system for brain-inspired AI models"""
    
    def __init__(self, model: nn.Module, learning_params: LearningParameters):
        self.model = model
        self.learning_params = learning_params
        
        # Initialize learning rules
        self.hebbian_learner = HebbianLearning(learning_params)
        self.dopamine_learner = DopamineModulatedLearning(learning_params)
        self.cerebellar_learner = CerebellarLearning(learning_params)
        self.hippocampal_learner = HippocampalLearning(learning_params)
        
        # Neuromodulatory system
        self.neuromod_system = NeuromodulatorySystem(100)  # Size can be adjusted
        self.adaptive_lr = AdaptiveLearningRate()
        
        # Consolidation scheduler
        self.consolidation_scheduler = ConsolidationScheduler()
        
        # Training history
        self.training_history = {
            'loss': [],
            'reward': [],
            'accuracy': [],
            'neuromodulators': []
        }
        
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, 
                   reward: Optional[torch.Tensor] = None, 
                   error: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step with multiple learning mechanisms
        """
        
        # Forward pass
        outputs = self.model(inputs, reward_signal=reward, error_signal=error)
        
        # Compute basic loss (can be customized)
        if 'actions' in outputs:
            # Classification task
            loss = F.cross_entropy(outputs['actions'], targets)
        else:
            # Regression task
            loss = F.mse_loss(outputs.get('motor', torch.zeros_like(targets)), targets)
        
        # Neuromodulatory signals
        neuromodulators = self.neuromod_system(
            reward_signal=reward,
            attention_signal=outputs.get('acetylcholine'),
            arousal_signal=torch.ones(1),  # Could be computed from input
            mood_signal=torch.ones(1)       # Could be tracked over time
        )
        
        # Adaptive learning rate
        adaptive_lr = self.adaptive_lr(neuromodulators)
        
        # Apply different learning rules to different parts of the model
        self._apply_learning_rules(outputs, inputs, reward, error, adaptive_lr)
        
        # Memory consolidation
        if self.consolidation_scheduler.update():
            self._consolidate_memories()
        
        # Record training history
        self.training_history['loss'].append(loss.item())
        if reward is not None:
            self.training_history['reward'].append(reward.mean().item())
        self.training_history['neuromodulators'].append({
            key: val.mean().item() for key, val in neuromodulators.items()
        })
        
        return {
            'loss': loss.item(),
            'adaptive_lr': adaptive_lr.item(),
            **{f'dopamine': neuromodulators['dopamine'].mean().item()},
            **{f'acetylcholine': neuromodulators['acetylcholine'].mean().item()}
        }
    
    def _apply_learning_rules(self, outputs: Dict[str, torch.Tensor], inputs: torch.Tensor,
                            reward: Optional[torch.Tensor], error: Optional[torch.Tensor],
                            learning_rate: float):
        """Apply appropriate learning rules to different model components"""
        
        # Update cortical weights with Hebbian learning
        if 'cortical' in outputs:
            for layer_name, layer_outputs in outputs['cortical'].items():
                for neuron_name, neuron_output in layer_outputs.items():
                    if hasattr(self.model.cortex.layers[layer_name][neuron_name], 'weights'):
                        weights = self.model.cortex.layers[layer_name][neuron_name].weights
                        updated_weights = self.hebbian_learner.update_weights(
                            weights, inputs, neuron_output, trace_key=f"{layer_name}_{neuron_name}"
                        )
                        self.model.cortex.layers[layer_name][neuron_name].weights.data = updated_weights
        
        # Update basal ganglia with dopamine-modulated learning
        if 'actions' in outputs and reward is not None:
            if hasattr(self.model.basal_ganglia, 'action_weights'):
                weights = self.model.basal_ganglia.action_weights
                updated_weights = self.dopamine_learner.update_weights(
                    weights, inputs, outputs['actions'], reward_signal=reward
                )
                self.model.basal_ganglia.action_weights.data = updated_weights
        
        # Update cerebellum with error-based learning
        if 'motor' in outputs and error is not None:
            if hasattr(self.model.cerebellum, 'parallel_to_purkinje'):
                weights = self.model.cerebellum.parallel_to_purkinje
                # Use cortical activity as presynaptic
                cortical_activity = torch.cat([
                    outputs['cortical']['L2_3']['pyramidal_IT'],
                    outputs['cortical']['L5']['pyramidal_PT']
                ], dim=-1) if 'cortical' in outputs else inputs
                
                updated_weights = self.cerebellar_learner.update_weights(
                    weights, cortical_activity, outputs['motor'], error_signal=error
                )
                self.model.cerebellum.parallel_to_purkinje.data = updated_weights
        
        # Update hippocampus with memory consolidation
        if 'memory' in outputs:
            if hasattr(self.model.hippocampus, 'ca3_recurrent'):
                weights = self.model.hippocampus.ca3_recurrent
                updated_weights = self.hippocampal_learner.update_weights(
                    weights, outputs['memory'], outputs['memory'], mode="consolidation"
                )
                self.model.hippocampus.ca3_recurrent.data = updated_weights
    
    def _consolidate_memories(self):
        """Perform memory consolidation"""
        
        # Strengthen important hippocampal connections
        if hasattr(self.model.hippocampus, 'ca3_recurrent'):
            weights = self.model.hippocampus.ca3_recurrent
            updated_weights = self.hippocampal_learner.update_weights(
                weights, weights, weights, mode="consolidation"
            )
            self.model.hippocampus.ca3_recurrent.data = updated_weights
        
        # Record consolidation
        self.consolidation_scheduler.consolidation_history.append(self.model.training_step if hasattr(self.model, 'training_step') else 0)
    
    def evaluate(self, test_inputs: torch.Tensor, test_targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        
        with torch.no_grad():
            outputs = self.model(test_inputs)
            
            if 'actions' in outputs:
                # Classification accuracy
                predictions = torch.argmax(outputs['actions'], dim=-1)
                accuracy = (predictions == test_targets).float().mean().item()
                loss = F.cross_entropy(outputs['actions'], test_targets).item()
            else:
                # Regression metrics
                predictions = outputs.get('motor', torch.zeros_like(test_targets))
                mse = F.mse_loss(predictions, test_targets).item()
                accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like metric
                loss = mse
        
        return {
            'accuracy': accuracy,
            'loss': loss
        }


def create_trainer(model: nn.Module, learning_params: Optional[LearningParameters] = None) -> BrainInspiredTrainer:
    """Factory function to create a trainer for brain-inspired models"""
    
    if learning_params is None:
        learning_params = LearningParameters()
    
    return BrainInspiredTrainer(model, learning_params)


if __name__ == "__main__":
    # Example usage
    from core_architecture import create_brain_model
    
    # Create model and trainer
    model = create_brain_model(input_size=784, output_size=10, scale="small")
    learning_params = LearningParameters(
        hebbian_learning_rate=0.01,
        dopamine_learning_rate=0.001,
        cerebellar_learning_rate=0.1
    )
    trainer = create_trainer(model, learning_params)
    
    # Simulate training
    print("Brain-Inspired AI Training Simulation")
    print("=" * 50)
    
    for epoch in range(10):
        # Generate random data
        batch_size = 32
        inputs = torch.randn(batch_size, 784)
        targets = torch.randint(0, 10, (batch_size,))
        rewards = torch.randn(batch_size)
        errors = torch.randn(batch_size, 10)
        
        # Training step
        metrics = trainer.train_step(inputs, targets, reward=rewards, error=errors)
        
        print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.4f}, "
              f"LR = {metrics['adaptive_lr']:.4f}, "
              f"Dopamine = {metrics['dopamine']:.4f}")
    
    # Evaluation
    test_inputs = torch.randn(100, 784)
    test_targets = torch.randint(0, 10, (100,))
    eval_metrics = trainer.evaluate(test_inputs, test_targets)
    
    print(f"\nFinal Evaluation: Accuracy = {eval_metrics['accuracy']:.4f}, Loss = {eval_metrics['loss']:.4f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
