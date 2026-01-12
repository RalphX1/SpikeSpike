"""
Spike1 - Simple Spiking Neural Network

A basic implementation of a spiking neural network using
Leaky Integrate-and-Fire (LIF) neurons.
"""

from .neuron import LIFNeuron
from .synapse import Synapse
from .network import SpikingNetwork

__all__ = ['LIFNeuron', 'Synapse', 'SpikingNetwork']
__version__ = '0.1.0'
