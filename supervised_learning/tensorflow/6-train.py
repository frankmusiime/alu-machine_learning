#!/usr/bin/env python3
"Module a training function for the neural network"
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.
    """
    # 1. Placeholders
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    
    # 2. Build Network Architecture
    # Import helper functions from previous tasks as permitted
    create_layer = __import__('1-create_layer').create_layer
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    
    # 3. Optimization Operation
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    
    # 4. Add Tensors and Operations to Graph Collections
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    
    # Saver object to handle model checkpoint storage
    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()
    
    # 5. Training Session
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(iterations + 1):
            # Evaluate current performance stats
            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            
            # Print conditions: 0th iteration, every 100 iterations, and final iteration
            if i == 0 or i == iterations or i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_accuracy))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_accuracy))
                
            # Perform training step optimization step if not at maximum iteration boundary
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
                
        # Save structural checkpoint graph variables after optimization steps wrap up
        return saver.save(sess, save_path)

