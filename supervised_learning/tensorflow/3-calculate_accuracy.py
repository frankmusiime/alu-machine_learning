#!/usr/bin/env python3
"""Module that calculates the accuracy of a prediction using TensorFlow."""
import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.
    
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network's predictions
    
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    # Get the index of the highest probability (the predicted class)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    
    # Cast the boolean values to floats and calculate the mean accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return accuracy
