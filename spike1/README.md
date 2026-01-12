# Spike1 - Simple Spiking Neural Network

A basic implementation of a Spiking Neural Network (SNN) using the Leaky Integrate-and-Fire (LIF) neuron model. This is the simplest version in a series of increasingly complex SNN implementations.

## Overview

This implementation provides:
- **LIF Neuron Model**: Classic Leaky Integrate-and-Fire neurons with configurable parameters
- **Synaptic Connections**: Weighted connections between neurons
- **Network Simulation**: Time-stepped simulation with external input support
- **Simple Architecture**: Easy to understand and extend

## Features

### Leaky Integrate-and-Fire (LIF) Neuron

The LIF neuron model follows the differential equation:

```
dV/dt = -(V - V_rest)/tau_m + I/C
```

Where:
- `V` is the membrane potential
- `V_rest` is the resting potential
- `tau_m` is the membrane time constant
- `I` is the input current
- `C` is the membrane capacitance (normalized to 1)

When `V >= V_thresh`, the neuron fires a spike and `V` is reset to `V_reset`.

### Key Parameters

- `tau_m`: Membrane time constant (default: 10 ms)
- `v_rest`: Resting potential (default: -65 mV)
- `v_reset`: Reset potential after spike (default: -70 mV)
- `v_thresh`: Threshold for spiking (default: -50 mV)
- `refractory_period`: Time after spike when neuron cannot fire (default: 2 ms)

## Usage

### Basic Example

```python
from spike1 import SpikingNetwork

# Create a network with 0.1 ms time step
net = SpikingNetwork(dt=0.1)

# Add neurons
net.add_neuron('n1')
net.add_neuron('n2')

# Connect them with a synapse (weight=20.0)
net.add_synapse('n1', 'n2', weight=20.0)

# Run simulation with external input to n1
results = net.run(
    duration=100,  # ms
    external_inputs={'n1': 15.0},
    record_v=True  # Record membrane potentials
)

# Access results
spike_times = results['spikes']
voltages = results['voltages']
```

### Running Examples

The `example.py` script includes three demonstrations:

1. **Single Neuron**: Shows how a neuron responds to constant input
2. **Connected Neurons**: Demonstrates spike propagation through a synapse
3. **Small Network**: A 4-neuron network with multiple connections

Run the examples:

```bash
cd /home/user/SpikeSpike
python -m spike1.example
```

This will generate visualization plots:
- `example1_single_neuron.png`
- `example2_connected_neurons.png`
- `example3_small_network.png`

## Architecture

### Components

1. **neuron.py**: Implements the `LIFNeuron` class
   - Maintains neuron state (membrane potential, refractory period)
   - Updates state based on input current
   - Generates spikes when threshold is reached

2. **synapse.py**: Implements the `Synapse` class
   - Represents weighted connections between neurons
   - Supports synaptic delay

3. **network.py**: Implements the `SpikingNetwork` class
   - Manages collections of neurons and synapses
   - Runs time-stepped simulations
   - Propagates spikes through the network
   - Records network activity

### Simulation Loop

The simulation follows these steps each time step:

1. Apply external inputs to neurons
2. Update each neuron's membrane potential
3. Check for threshold crossings and generate spikes
4. Propagate spikes through synapses to postsynaptic neurons
5. Record activity
6. Advance simulation time

## Limitations

This is a **simple** implementation designed for learning and basic experiments:

- No synaptic delays (spikes propagate instantaneously)
- No plasticity (weights are fixed)
- Simple Euler integration (not the most accurate)
- No complex neuron models
- Limited scalability

For more advanced features, see the `spike2` and `spike3` directories.

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib (for examples/visualization)

## Future Enhancements

Potential improvements for future versions:
- Synaptic delays with spike queues
- Spike-Timing-Dependent Plasticity (STDP)
- More accurate integration methods (Runge-Kutta)
- Different neuron models (Izhikevich, Hodgkin-Huxley)
- Network analysis tools
- GPU acceleration
- More complex network topologies

## References

- Gerstner, W., & Kistler, W. M. (2002). Spiking Neuron Models. Cambridge University Press.
- Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience. MIT Press.
