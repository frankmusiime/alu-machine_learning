#!/usr/bin/env python3
"""Module that contains the create_layer function"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a fully connected neural network layer.

    Parameters:
    - prev: tensor output of the previous layer
    - n: number of nodes in the layer to create
    - activation: activation function that the layer should use

    Returns:
    - The tensor output of the layer
    """
    # Define the He et al. initialization for weights
    init_weights = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    # Create a fully connected layer with the specified parameters
    layer = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_initializer=init_weights,
        name="layer"
    )

    return layer
