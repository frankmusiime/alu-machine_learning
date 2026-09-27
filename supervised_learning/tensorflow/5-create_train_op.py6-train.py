#!/usr/bin/env python3
"""Module to create the training operation"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Parameters:
        loss: the loss of the network's prediction
        alpha: the learning rate

    Returns:
        An operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
