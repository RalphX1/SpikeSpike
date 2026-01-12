"""
Example usage of the simple spiking neural network.

This script demonstrates:
1. Creating a small network of LIF neurons
2. Connecting them with synapses
3. Running a simulation with external input
4. Visualizing the results
"""
import numpy as np
import matplotlib.pyplot as plt
from spike1 import SpikingNetwork


def example_single_neuron():
    """Example 1: Single neuron responding to constant input."""
    print("=" * 60)
    print("Example 1: Single Neuron with Constant Input")
    print("=" * 60)

    # Create network
    net = SpikingNetwork(dt=0.1)

    # Add a single neuron
    net.add_neuron('n1')

    # Run with constant input current
    duration = 100  # ms
    input_current = 15.0  # Input current strength
    results = net.run(
        duration=duration,
        external_inputs={'n1': input_current},
        record_v=True
    )

    # Print results
    print(f"Neuron spiked {len(results['spikes']['n1'])} times")
    print(f"Spike times: {results['spikes']['n1'][:10]}... (first 10)")

    # Plot membrane potential
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    time = results['time_points']
    voltage = results['voltages']['n1']

    ax1.plot(time, voltage, 'b-', linewidth=0.5)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Single Neuron Response to Constant Input')
    ax1.grid(True, alpha=0.3)

    # Raster plot
    if results['spikes']['n1']:
        ax2.eventplot(results['spikes']['n1'], colors='red', linewidths=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Spike')
    ax2.set_title('Spike Raster')
    ax2.set_xlim(0, duration)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/SpikeSpike/spike1/example1_single_neuron.png', dpi=150)
    print("Plot saved to: spike1/example1_single_neuron.png\n")


def example_connected_neurons():
    """Example 2: Two connected neurons with feed-forward connection."""
    print("=" * 60)
    print("Example 2: Two Connected Neurons")
    print("=" * 60)

    # Create network
    net = SpikingNetwork(dt=0.1)

    # Add two neurons
    net.add_neuron('input')
    net.add_neuron('output')

    # Connect them with a strong excitatory synapse
    net.add_synapse('input', 'output', weight=20.0)

    # Run with input to first neuron only
    duration = 100
    results = net.run(
        duration=duration,
        external_inputs={'input': 15.0},  # Only stimulate input neuron
        record_v=True
    )

    # Print results
    print(f"Input neuron spiked {len(results['spikes'].get('input', []))} times")
    print(f"Output neuron spiked {len(results['spikes'].get('output', []))} times")

    # Plot both neurons
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    time = results['time_points']

    # Plot membrane potentials
    axes[0].plot(time, results['voltages']['input'], 'b-', linewidth=0.5, label='Input')
    axes[0].set_ylabel('Input V (mV)')
    axes[0].set_title('Two Connected Neurons (Input -> Output)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, results['voltages']['output'], 'g-', linewidth=0.5, label='Output')
    axes[1].set_ylabel('Output V (mV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Raster plot
    spike_data = []
    labels = []
    for idx, neuron_id in enumerate(['input', 'output']):
        if neuron_id in results['spikes'] and results['spikes'][neuron_id]:
            spike_data.append(results['spikes'][neuron_id])
            labels.append(neuron_id)

    if spike_data:
        axes[2].eventplot(spike_data, linewidths=2, colors=['blue', 'green'])
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(labels)
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Neuron')
    axes[2].set_title('Spike Raster')
    axes[2].set_xlim(0, duration)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/SpikeSpike/spike1/example2_connected_neurons.png', dpi=150)
    print("Plot saved to: spike1/example2_connected_neurons.png\n")


def example_small_network():
    """Example 3: Small network with multiple connections."""
    print("=" * 60)
    print("Example 3: Small Network (4 neurons)")
    print("=" * 60)

    # Create network
    net = SpikingNetwork(dt=0.1)

    # Add 4 neurons
    for i in range(4):
        net.add_neuron(f'n{i}')

    # Create connections:
    # n0 -> n1, n2
    # n1 -> n3
    # n2 -> n3
    net.add_synapse('n0', 'n1', weight=18.0)
    net.add_synapse('n0', 'n2', weight=18.0)
    net.add_synapse('n1', 'n3', weight=12.0)
    net.add_synapse('n2', 'n3', weight=12.0)

    print(f"Network: {net}")

    # Run with input to first neuron only
    duration = 150
    results = net.run(
        duration=duration,
        external_inputs={'n0': 15.0},  # Only stimulate first neuron
        record_v=True
    )

    # Print results
    for neuron_id in ['n0', 'n1', 'n2', 'n3']:
        num_spikes = len(results['spikes'].get(neuron_id, []))
        print(f"Neuron {neuron_id} spiked {num_spikes} times")

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))

    time = results['time_points']

    # Plot membrane potentials
    for idx, neuron_id in enumerate(['n0', 'n1', 'n2', 'n3']):
        axes[idx].plot(time, results['voltages'][neuron_id], linewidth=0.5)
        axes[idx].set_ylabel(f'{neuron_id} V (mV)')
        axes[idx].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx].set_title('Small Network Activity (n0 -> {n1, n2} -> n3)')

    # Raster plot
    spike_data = []
    colors = []
    color_map = {'n0': 'red', 'n1': 'blue', 'n2': 'green', 'n3': 'orange'}
    for neuron_id in ['n0', 'n1', 'n2', 'n3']:
        if neuron_id in results['spikes'] and results['spikes'][neuron_id]:
            spike_data.append(results['spikes'][neuron_id])
            colors.append(color_map[neuron_id])

    if spike_data:
        axes[4].eventplot(spike_data, linewidths=2, colors=colors)
        axes[4].set_yticks(range(len(spike_data)))
        axes[4].set_yticklabels(['n0', 'n1', 'n2', 'n3'][:len(spike_data)])
    axes[4].set_xlabel('Time (ms)')
    axes[4].set_ylabel('Neuron')
    axes[4].set_title('Spike Raster')
    axes[4].set_xlim(0, duration)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/SpikeSpike/spike1/example3_small_network.png', dpi=150)
    print("Plot saved to: spike1/example3_small_network.png\n")


if __name__ == '__main__':
    # Run all examples
    example_single_neuron()
    example_connected_neurons()
    example_small_network()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
