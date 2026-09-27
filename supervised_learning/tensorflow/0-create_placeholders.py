#!/usr/bin/env python3
"""Creates placeholders for a neural network."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Creates placeholders for the neural network.

    Args:
        nx: Number of feature columns.
        classes: Number of classes in the classifier.

    Returns:
        x: Placeholder for the input data.
        y: Placeholder for the one-hot labels.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y
