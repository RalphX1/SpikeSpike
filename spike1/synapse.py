"""
Synaptic connections between neurons
"""


class Synapse:
    """
    Represents a synaptic connection between two neurons.

    Implements a simple weighted connection with optional delay.
    """

    def __init__(self, pre_neuron_id, post_neuron_id, weight, delay=1.0):
        """
        Initialize a synapse.

        Args:
            pre_neuron_id: ID of the presynaptic (source) neuron
            post_neuron_id: ID of the postsynaptic (target) neuron
            weight: Synaptic weight (strength of connection)
            delay: Synaptic delay in ms
        """
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = weight
        self.delay = delay

    def __repr__(self):
        return f"Synapse({self.pre_neuron_id} -> {self.post_neuron_id}, w={self.weight:.2f})"
