import tensorflow as tf

def fc(x, num_in, units, name, relu=True):
    """Create a fully connected layer.
    Args:
        x: A `Tensor`. It holds the input samples.
        num_in: An integer. It specifies the dimensionality of the input
        units: An integer. It specifies the dimensionality of the output
    Returns:
    Output tensor the same shape as `x` except the last dimension is of size `units`.    
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable(name='weights',
                               shape=[num_in, units],
                               initializer=tf.glorot_uniform_initializer(),
                               regularizer=tf.nn.l2_loss,
                               trainable=True)
        biases = tf.get_variable(name='biases',
                               shape=[units],
                               initializer=tf.zeros_initializer(),
                               trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non-linearity
        relu = tf.nn.relu(act)
        return relu, weights, biases
    else:
        return act, weights, biases

def conv():
    """Create a convolutional layer"""
    raise NotImplementedError
