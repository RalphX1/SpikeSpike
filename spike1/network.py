"""
Spiking Neural Network implementation
"""
import numpy as np
from collections import defaultdict
from .neuron import LIFNeuron
from .synapse import Synapse


class SpikingNetwork:
    """
    A simple spiking neural network.

    Manages a collection of LIF neurons connected by synapses.
    """

    def __init__(self, dt=0.1):
        """
        Initialize the network.

        Args:
            dt: Time step for simulation (ms)
        """
        self.dt = dt
        self.neurons = {}  # neuron_id -> LIFNeuron
        self.synapses = []  # List of Synapse objects
        self.time = 0.0  # Current simulation time

        # For tracking connectivity
        self.outgoing = defaultdict(list)  # pre_id -> [(post_id, weight)]
        self.incoming = defaultdict(list)  # post_id -> [(pre_id, weight)]

    def add_neuron(self, neuron_id, **kwargs):
        """
        Add a neuron to the network.

        Args:
            neuron_id: Unique identifier for the neuron
            **kwargs: Parameters for LIFNeuron constructor

        Returns:
            The created LIFNeuron object
        """
        if neuron_id in self.neurons:
            raise ValueError(f"Neuron {neuron_id} already exists")

        neuron = LIFNeuron(neuron_id, **kwargs)
        self.neurons[neuron_id] = neuron
        return neuron

    def add_synapse(self, pre_id, post_id, weight, delay=1.0):
        """
        Add a synaptic connection between two neurons.

        Args:
            pre_id: ID of presynaptic neuron
            post_id: ID of postsynaptic neuron
            weight: Synaptic weight
            delay: Synaptic delay (ms)

        Returns:
            The created Synapse object
        """
        if pre_id not in self.neurons:
            raise ValueError(f"Presynaptic neuron {pre_id} does not exist")
        if post_id not in self.neurons:
            raise ValueError(f"Postsynaptic neuron {post_id} does not exist")

        synapse = Synapse(pre_id, post_id, weight, delay)
        self.synapses.append(synapse)

        # Update connectivity maps
        self.outgoing[pre_id].append((post_id, weight))
        self.incoming[post_id].append((pre_id, weight))

        return synapse

    def step(self, external_inputs=None):
        """
        Simulate one time step.

        Args:
            external_inputs: Dict mapping neuron_id -> input current

        Returns:
            Dict mapping neuron_id -> bool (True if spiked)
        """
        if external_inputs is None:
            external_inputs = {}

        spikes = {}

        # Update each neuron
        for neuron_id, neuron in self.neurons.items():
            # Get external input for this neuron
            current_input = external_inputs.get(neuron_id, 0.0)

            # Update neuron and record if it spiked
            spiked = neuron.step(self.dt, current_input)
            spikes[neuron_id] = spiked

            if spiked:
                neuron.spike_times.append(self.time)
                neuron.last_spike_time = self.time

        # Propagate spikes through synapses
        synaptic_inputs = defaultdict(float)
        for neuron_id, spiked in spikes.items():
            if spiked:
                # Send current to all postsynaptic neurons
                for post_id, weight in self.outgoing[neuron_id]:
                    synaptic_inputs[post_id] += weight

        # Apply synaptic inputs (will affect next time step)
        # In this simple version, we apply immediately without delay
        for post_id, current in synaptic_inputs.items():
            self.neurons[post_id].i_input = current

        self.time += self.dt
        return spikes

    def run(self, duration, external_inputs=None, record_v=False):
        """
        Run simulation for a specified duration.

        Args:
            duration: Simulation duration (ms)
            external_inputs: Function that takes time and returns dict of inputs,
                           or a static dict of inputs
            record_v: If True, record membrane potentials

        Returns:
            Dict with 'spikes' and optionally 'voltages'
        """
        num_steps = int(duration / self.dt)
        spike_trains = defaultdict(list)
        voltages = defaultdict(list) if record_v else None

        for step in range(num_steps):
            # Get inputs for this time step
            if callable(external_inputs):
                inputs = external_inputs(self.time)
            elif external_inputs is not None:
                inputs = external_inputs
            else:
                inputs = {}

            # Simulate one step
            spikes = self.step(inputs)

            # Record spikes
            for neuron_id, spiked in spikes.items():
                if spiked:
                    spike_trains[neuron_id].append(self.time)

            # Record voltages if requested
            if record_v:
                for neuron_id, neuron in self.neurons.items():
                    voltages[neuron_id].append(neuron.v)

        result = {'spikes': dict(spike_trains)}
        if record_v:
            result['voltages'] = dict(voltages)
            result['time_points'] = np.arange(0, duration, self.dt)

        return result

    def reset(self):
        """Reset all neurons and simulation time."""
        for neuron in self.neurons.values():
            neuron.reset()
        self.time = 0.0

    def __repr__(self):
        return f"SpikingNetwork({len(self.neurons)} neurons, {len(self.synapses)} synapses)"
