#!/usr/bin/env python3
Module to create a forward propagation graph for a neural network.
"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Parameters:
    - x: placeholder for the input data
    - layer_sizes: list containing the number of nodes in each layer
    - activations: list containing the activation functions for each layer

    Returns:
    - The prediction of the network in tensor form
    """
    # Import the create_layer function as specified in the prompt
    create_layer = __import__('1-create_layer').create_layer

    # Initialize the current layer output with the input data
    current_output = x

    # Loop through all layers to connect them sequentially
    for i in range(len(layer_sizes)):
        n_nodes = layer_sizes[i]
        activation = activations[i]
        
        # Pass the output of the previous layer into the next layer
        current_output = create_layer(current_output, n_nodes, activation)

    return current_output
