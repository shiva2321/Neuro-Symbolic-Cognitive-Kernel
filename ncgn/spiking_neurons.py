"""
Spiking Neural Network Components
Implements neuromorphic computing with event-driven dynamics.

Agent Prompt 2 Implementation:
- Leaky Integrate-and-Fire (LIF) neurons
- Poisson spike encoding
- Event-driven computation
- Energy-efficient operations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model.

    Dynamics:
        tau * dv/dt = -(v - v_rest) + R*I(t)

    Where:
        v: membrane potential
        tau: membrane time constant
        v_rest: resting potential
        R: membrane resistance
        I(t): input current
    """

    def __init__(self, tau: float = 10.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, v_rest: float = 0.0,
                 refractory_period: int = 2):
        """
        Initialize LIF neuron parameters.

        Args:
            tau: Membrane time constant (ms)
            v_threshold: Spike threshold voltage
            v_reset: Reset voltage after spike
            v_rest: Resting membrane potential
            refractory_period: Refractory period in timesteps
        """
        super().__init__()

        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.refractory_period = refractory_period

        # Decay factor (for discrete time)
        self.beta = np.exp(-1.0 / tau)

        # State variables (will be initialized during forward pass)
        self.membrane_potential = None
        self.refractory_counter = None

    def reset_state(self, batch_size: int, num_neurons: int, device: str = 'cpu'):
        """
        Reset neuron state.

        Args:
            batch_size: Batch size
            num_neurons: Number of neurons
            device: Computation device
        """
        self.membrane_potential = torch.ones(batch_size, num_neurons,
                                             device=device) * self.v_rest
        self.refractory_counter = torch.zeros(batch_size, num_neurons,
                                              device=device, dtype=torch.long)

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one timestep.

        Args:
            input_current: Input current (batch_size, num_neurons)

        Returns:
            Tuple of (spikes, membrane_potential)
        """
        batch_size, num_neurons = input_current.shape
        device = input_current.device

        # Initialize state if needed
        if self.membrane_potential is None:
            self.reset_state(batch_size, num_neurons, device)

        # Check refractory period
        not_refractory = (self.refractory_counter == 0)

        # Update membrane potential (leaky integration)
        # v(t+1) = beta * v(t) + (1 - beta) * (v_rest + R*I)
        self.membrane_potential = (
            self.beta * self.membrane_potential +
            (1 - self.beta) * (self.v_rest + input_current)
        ) * not_refractory.float() + self.membrane_potential * (~not_refractory).float()

        # Generate spikes where threshold is exceeded
        spikes = (self.membrane_potential >= self.v_threshold).float()

        # Reset neurons that spiked
        self.membrane_potential = torch.where(
            spikes.bool(),
            torch.ones_like(self.membrane_potential) * self.v_reset,
            self.membrane_potential
        )

        # Set refractory period for neurons that spiked
        self.refractory_counter = torch.where(
            spikes.bool(),
            torch.ones_like(self.refractory_counter) * self.refractory_period,
            torch.maximum(self.refractory_counter - 1,
                         torch.zeros_like(self.refractory_counter))
        )

        return spikes, self.membrane_potential


class IzhikevichNeuron(nn.Module):
    """
    Izhikevich Neuron Model - more biologically realistic.

    Dynamics:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)

    If v >= 30mV:
        v <- c
        u <- u + d
    """

    def __init__(self, a: float = 0.02, b: float = 0.2,
                 c: float = -65.0, d: float = 8.0):
        """
        Initialize Izhikevich neuron.

        Args:
            a: Recovery time scale
            b: Sensitivity of recovery variable
            c: Reset voltage
            d: Reset of recovery variable
        """
        super().__init__()

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.v = None  # Membrane potential
        self.u = None  # Recovery variable

    def reset_state(self, batch_size: int, num_neurons: int, device: str = 'cpu'):
        """Reset neuron state"""
        self.v = torch.ones(batch_size, num_neurons, device=device) * self.c
        self.u = self.b * self.v

    def forward(self, input_current: torch.Tensor, dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one timestep.

        Args:
            input_current: Input current
            dt: Time step

        Returns:
            Tuple of (spikes, membrane_potential)
        """
        batch_size, num_neurons = input_current.shape
        device = input_current.device

        if self.v is None:
            self.reset_state(batch_size, num_neurons, device)

        # Update dynamics
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + input_current) * dt
        du = self.a * (self.b * self.v - self.u) * dt

        self.v = self.v + dv
        self.u = self.u + du

        # Check for spikes
        spikes = (self.v >= 30.0).float()

        # Reset spiked neurons
        self.v = torch.where(spikes.bool(),
                            torch.ones_like(self.v) * self.c,
                            self.v)
        self.u = torch.where(spikes.bool(),
                            self.u + self.d,
                            self.u)

        return spikes, self.v


class PoissonEncoder:
    """
    Encodes continuous values into spike trains using Poisson process.
    """

    @staticmethod
    def encode(values: torch.Tensor, time_steps: int,
               max_rate: float = 100.0) -> torch.Tensor:
        """
        Encode continuous values into Poisson spike trains.

        Args:
            values: Input values (batch_size, num_features)
            time_steps: Number of time steps
            max_rate: Maximum firing rate (Hz)

        Returns:
            Spike trains (time_steps, batch_size, num_features)
        """
        # Normalize values to [0, 1]
        normalized = torch.sigmoid(values)

        # Calculate firing probabilities
        # rate = normalized * max_rate
        # probability per time step = rate * dt (assuming dt=1ms)
        prob = normalized * (max_rate / 1000.0)

        # Generate spikes
        spikes = []
        for _ in range(time_steps):
            spike = torch.bernoulli(prob)
            spikes.append(spike)

        return torch.stack(spikes, dim=0)

    @staticmethod
    def decode(spike_train: torch.Tensor) -> torch.Tensor:
        """
        Decode spike trains back to continuous values.

        Args:
            spike_train: Spike trains (time_steps, batch_size, num_features)

        Returns:
            Decoded values (batch_size, num_features)
        """
        # Sum spikes over time and normalize by time steps
        return spike_train.sum(dim=0) / spike_train.shape[0]


class RateEncoder:
    """
    Encodes continuous values using rate coding.
    Higher values produce higher firing rates.
    """

    @staticmethod
    def encode(values: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        Encode using rate coding.

        Args:
            values: Input values (batch_size, num_features)
            time_steps: Number of time steps

        Returns:
            Spike trains (time_steps, batch_size, num_features)
        """
        # Normalize to [0, 1]
        normalized = torch.sigmoid(values)

        # Create spike pattern
        spikes = []
        for t in range(time_steps):
            # Probability decreases with time
            prob = normalized * (1.0 - t / time_steps)
            spike = (torch.rand_like(prob) < prob).float()
            spikes.append(spike)

        return torch.stack(spikes, dim=0)


class TemporalEncoder:
    """
    Encodes continuous values using temporal coding (latency).
    Higher values spike earlier.
    """

    @staticmethod
    def encode(values: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        Encode using temporal/latency coding.

        Args:
            values: Input values (batch_size, num_features)
            time_steps: Number of time steps

        Returns:
            Spike trains (time_steps, batch_size, num_features)
        """
        # Normalize to [0, 1]
        normalized = torch.sigmoid(values)

        # Calculate spike times (higher value -> earlier spike)
        spike_times = ((1.0 - normalized) * (time_steps - 1)).long()

        # Create spike trains
        batch_size, num_features = values.shape
        spikes = torch.zeros(time_steps, batch_size, num_features,
                            device=values.device)

        for b in range(batch_size):
            for f in range(num_features):
                t = spike_times[b, f].item()
                if 0 <= t < time_steps:
                    spikes[t, b, f] = 1.0

        return spikes


class SpikingLayer(nn.Module):
    """
    A layer of spiking neurons with synaptic connections.
    """

    def __init__(self, in_features: int, out_features: int,
                 neuron_model: str = 'lif', bias: bool = True,
                 synaptic_delay: Optional[torch.Tensor] = None):
        """
        Initialize spiking layer.

        Args:
            in_features: Number of input features
            out_features: Number of output neurons
            neuron_model: Type of neuron ('lif' or 'izhikevich')
            bias: Whether to include bias
            synaptic_delay: Delays for each synapse (in_features, out_features)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Synaptic weights
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Neuron model
        if neuron_model.lower() == 'lif':
            self.neurons = LIFNeuron()
        elif neuron_model.lower() == 'izhikevich':
            self.neurons = IzhikevichNeuron()
        else:
            raise ValueError(f"Unknown neuron model: {neuron_model}")

        # Synaptic delays (in timesteps)
        if synaptic_delay is None:
            # Random delays between 1-5 timesteps
            self.synaptic_delay = torch.randint(1, 6, (in_features, out_features))
        else:
            self.synaptic_delay = synaptic_delay

        # Delay buffer
        self.max_delay = self.synaptic_delay.max().item()
        self.spike_buffer = []

    def reset_state(self, batch_size: int, device: str = 'cpu'):
        """Reset layer state"""
        self.neurons.reset_state(batch_size, self.out_features, device)
        self.spike_buffer = []

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one timestep.

        Args:
            input_spikes: Input spike train (batch_size, in_features)

        Returns:
            Output spikes (batch_size, out_features)
        """
        device = input_spikes.device
        batch_size = input_spikes.shape[0]

        # Add to buffer
        self.spike_buffer.append(input_spikes.cpu())

        # Keep only necessary history
        if len(self.spike_buffer) > self.max_delay:
            self.spike_buffer.pop(0)

        # Compute input current with delays
        input_current = torch.zeros(batch_size, self.out_features, device=device)

        for delay in range(1, min(len(self.spike_buffer) + 1, self.max_delay + 1)):
            if delay <= len(self.spike_buffer):
                delayed_spikes = self.spike_buffer[-delay].to(device)

                # Apply weights only for synapses with this delay
                delay_mask = (self.synaptic_delay == delay).float().to(device)
                masked_weights = self.weight * delay_mask

                input_current += torch.matmul(delayed_spikes, masked_weights)

        # Add bias
        if self.bias is not None:
            input_current = input_current + self.bias

        # Process through neurons
        output_spikes, _ = self.neurons(input_current)

        return output_spikes


class SpikingGraphConvolution(nn.Module):
    """
    Graph convolution with spiking neurons.
    Combines message passing with neuromorphic dynamics.
    """

    def __init__(self, in_features: int, out_features: int,
                 neuron_model: str = 'lif'):
        """
        Initialize spiking graph convolution.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            neuron_model: Type of spiking neuron
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Transformation weights
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

        # Spiking neurons
        if neuron_model.lower() == 'lif':
            self.neurons = LIFNeuron()
        elif neuron_model.lower() == 'izhikevich':
            self.neurons = IzhikevichNeuron()
        else:
            raise ValueError(f"Unknown neuron model: {neuron_model}")

    def forward(self, input_spikes: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with graph structure.

        Args:
            input_spikes: Input spikes (batch_size, num_nodes, in_features)
            adjacency: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Output spikes (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = input_spikes.shape
        device = input_spikes.device

        # Transform features
        transformed = torch.matmul(input_spikes, self.weight)

        # Aggregate messages from neighbors
        # Normalize by degree
        degree = adjacency.sum(dim=1, keepdim=True) + 1e-6
        norm_adj = adjacency / degree

        # Message passing: sum of neighbor features
        aggregated = torch.matmul(norm_adj.unsqueeze(0), transformed)

        # Reshape for neuron processing
        aggregated_flat = aggregated.view(batch_size * num_nodes, self.out_features)

        # Process through spiking neurons
        output_spikes, _ = self.neurons(aggregated_flat)

        # Reshape back
        output_spikes = output_spikes.view(batch_size, num_nodes, self.out_features)

        return output_spikes


if __name__ == "__main__":
    # Test LIF neuron
    logger.info("Testing LIF Neuron...")

    lif = LIFNeuron(tau=10.0, v_threshold=1.0)

    # Simulate for 100 timesteps
    batch_size, num_neurons = 2, 5
    spike_times = []

    lif.reset_state(batch_size, num_neurons)

    for t in range(100):
        # Constant input current
        input_current = torch.ones(batch_size, num_neurons) * 0.15
        spikes, membrane = lif(input_current)

        if spikes.any():
            spike_times.append(t)

    print(f"Spikes occurred at timesteps: {spike_times[:10]}...")

    # Test Poisson encoder
    logger.info("\nTesting Poisson Encoder...")

    values = torch.randn(2, 10)
    spike_train = PoissonEncoder.encode(values, time_steps=50)
    decoded = PoissonEncoder.decode(spike_train)

    print(f"Original values: {values[0, :5]}")
    print(f"Spike counts: {spike_train[:, 0, :5].sum(dim=0)}")
    print(f"Decoded values: {decoded[0, :5]}")

    logger.info("\nSpiking neural network tests complete!")

