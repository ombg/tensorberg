import tensorflow as tf

def fc(x, num_in, units, name, relu=True, force_alloc=False, log_weights=True):
    """Create a fully connected layer.
    Args:
        x: A `Tensor`. It holds the input samples.
        num_in: An integer. It specifies the dimensionality of the input
        units: An integer. It specifies the dimensionality of the output
    Returns:
    Output tensor the same shape as `x` except the last dimension is of size `units`.    
    """
    max_p = 3e8
    if num_in * units > max_p and force_alloc == False:
        raise MemoryError(('You are trying to allocate more than {} bytes' 
                           ' for a single weight matrix.'
                           ' If this is a good idea'
                           ' use `force_alloc=True`').format(int(max_p * 4)))
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable(name='weights',
                               shape=[num_in, units],
                               initializer=tf.random_normal_initializer(stddev=1e-3),
                               regularizer=None,
                               trainable=True)
        biases = tf.get_variable(name='biases',
                               shape=[units],
                               initializer=tf.zeros_initializer(),
                               trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if log_weights == True:
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)
    if relu:
        # Apply ReLu non-linearity
        relu = tf.nn.relu(act)
        return relu, weights, biases
    else:
        return act, weights, biases

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', trainable=True, log_weights=False):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    Adapted from: https://github.com/kratzert/finetune_alexnet_with_tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution

    with tf.variable_scope(name) as scope:

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        weights = tf.get_variable(name='weights',
                                  shape=[filter_height, filter_width,
                                         input_channels, num_filters],
                                  initializer=tf.glorot_uniform_initializer(),
                                  regularizer=regularizer,
                                  trainable=trainable)

        biases = tf.get_variable(name='biases',
                                 shape=[num_filters],
                                 initializer=tf.zeros_initializer(),
                                 trainable=trainable)

        out = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1],
                                        padding=padding)
        # Add biases
        out = tf.nn.bias_add(out, biases)

        # Apply relu function
        out = tf.nn.relu(out, name=scope.name)

        if log_weights == True:
            tf.summary.image('weights', weights[tf.newaxis,:,:,0,0,tf.newaxis])
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)

    return out, weights, biases
