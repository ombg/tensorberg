import tensorflow as tf
import numpy as np

def complex_model(X,y,is_training):
    # Initialize dimensions and shapes
    F = 7
    D = 32
    S = 1
    P = 0
    S_pool = 2
    pool_height = 2
    pool_width = 2
    conv1_out_rows = 1 + (X.shape[1] - F + 2*P) // S
    conv1_out_cols = 1 + (X.shape[2] - F + 2*P) // S
    pool_output_height = 1 + (conv1_out_rows - pool_height) // S_pool
    pool_output_width  = 1 + (conv1_out_cols - pool_width) // S_pool
    hidden_dim_1 = pool_output_height * pool_output_width * D
    hidden_dim_2 = 1024
    num_classes = 10

    # Setup variables
    # Conv1
    Wconv1 = tf.get_variable("Wconv1", shape=[F,F,X.shape[3],D])
    bconv1 = tf.get_variable("bconv1", shape=[D])
    # FC layer 1
    W1 = tf.get_variable("W1", shape=[hidden_dim_1, hidden_dim_2])
    b1 = tf.get_variable("b1", shape=[hidden_dim_2])
    # FC layer 2
    W2 = tf.get_variable("W2", shape=[hidden_dim_2, num_classes])
    b2 = tf.get_variable("b2", shape=[num_classes])

    # Spatial Batchnorm
    gamma = tf.get_variable("gamma", shape=[32])
    beta = tf.get_variable("beta", shape=[32])
    
    # define our graph
    # default data_format='NHWC'  !!!

    # Conv1
    conv1 = tf.nn.conv2d(X, Wconv1, strides=[1,S,S,1], padding='VALID') + bconv1
    conv_z1 = tf.nn.relu(conv1)
    
    # Spatial Batchnorm
    # Note: Make sure you know the default inputs of this function
    bn_conv_z1 = tf.layers.batch_normalization(conv_z1,
                                               training=is_training,
                                              name='batchnorm1')
    # Max pooling
    pool_out = tf.layers.max_pooling2d(inputs=bn_conv_z1,
                        pool_size=[2,2],
                        strides=S_pool, 
                        padding='VALID',
                        name='pool1')
    
    # output volume after pooling and flattened.
    dim_pool_out_flat = pool_out.get_shape()[1] * pool_out.get_shape()[2] * pool_out.get_shape()[3]
    
    # FC layer 1
    pool_out_flat = tf.reshape(pool_out,[-1, int(dim_pool_out_flat)])
    a1 = tf.matmul(pool_out_flat,W1) + b1
    h1 = tf.nn.relu(a1)

    # FC layer 2
    y_out = tf.matmul(h1, W2) + b2
    return y_out
