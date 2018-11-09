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

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    Adapted from: https://github.com/kratzert/finetune_alexnet_with_tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu, weights, biases
