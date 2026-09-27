#!/usr/bin/env python3
"""Module to evaluate a neural network performance using a saved meta graph."""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Parameters:
    - X: numpy.ndarray containing the input data to evaluate.
    - Y: numpy.ndarray containing the one-hot labels for X.
    - save_path: str, the location to load the model from.

    Returns:
    - prediction: numpy.ndarray, the network's prediction.
    - accuracy: float, the accuracy of the network.
    - loss: float, the loss of the network.
    """
    with tf.Session() as sess:
        # Import the meta graph
        saver = tf.train.import_meta_graph(save_path + '.meta')
        
        # Restore the variables/weights into the session
        saver.restore(sess, save_path)
        
        # Get tensors from the graph's collection
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        
        # Get the input and label placeholders from the graph by name or collection.
        # Since the main script feeds X and Y, we retrieve the default graph's operations.
        # Commonly, these are named 'x:0' and 'y:0' or similar in TensorFlow 1.x projects.
        x_placeholder = tf.get_default_graph().get_tensor_by_name('x:0')
        y_placeholder = tf.get_default_graph().get_tensor_by_name('y:0')
        
        # Run the evaluation in the session
        prediction, acc_val, loss_val = sess.run(
            [y_pred, accuracy, loss],
            feed_dict={x_placeholder: X, y_placeholder: Y}
        )
        
    return prediction, acc_val, loss_val

