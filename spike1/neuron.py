"""
Simple Leaky Integrate-and-Fire (LIF) Neuron Model
"""
import numpy as np


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model.

    The membrane potential follows:
    dV/dt = -(V - V_rest)/tau_m + I/C

    When V >= V_thresh, the neuron spikes and V is reset to V_reset.
    """

    def __init__(self, neuron_id, tau_m=10.0, v_rest=-65.0, v_reset=-70.0,
                 v_thresh=-50.0, refractory_period=2.0):
        """
        Initialize a LIF neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau_m: Membrane time constant (ms)
            v_rest: Resting potential (mV)
            v_reset: Reset potential after spike (mV)
            v_thresh: Threshold potential for spiking (mV)
            refractory_period: Refractory period duration (ms)
        """
        self.id = neuron_id
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.refractory_period = refractory_period

        # State variables
        self.v = v_rest  # Current membrane potential
        self.i_input = 0.0  # Input current
        self.refractory_time = 0.0  # Time remaining in refractory period
        self.last_spike_time = -np.inf  # Time of last spike
        self.spike_times = []  # History of spike times

    def step(self, dt, current_input=0.0):
        """
        Update neuron state for one time step.

        Args:
            dt: Time step (ms)
            current_input: External input current

        Returns:
            bool: True if neuron spiked, False otherwise
        """
        spiked = False

        # Update refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False

        # Update membrane potential using Euler method
        dv = (-(self.v - self.v_rest) + current_input) / self.tau_m
        self.v += dv * dt

        # Check for spike
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refractory_time = self.refractory_period
            spiked = True

        return spiked

    def reset(self):
        """Reset neuron to initial state."""
        self.v = self.v_rest
        self.i_input = 0.0
        self.refractory_time = 0.0
        self.last_spike_time = -np.inf
        self.spike_times = []
