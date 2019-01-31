import sys
import pprint

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from models import layers
from ompy import plotml, ml

def main():

    d =  np.load('/tmp/spiral_data.npy')
    train_x = d.item().get('features')
    train_y = d.item().get('labels')
    train_y_scalar = d.item().get('labels_scalar')
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')

    d =  np.load('/tmp/spiral_testdata.npy')
    test_x = d.item().get('features')
    test_y = d.item().get('labels')
    test_y_scalar = d.item().get('labels_scalar')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('float32')

    # Create datasets
    trainset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    trainset = trainset.repeat(-1)
    trainset = trainset.batch(900)
    testset = tf.data.Dataset.from_tensor_slices((test_x,test_y))
    testset = testset.repeat(-1)
    testset = testset.batch(300) # Set to size of testset
    handle = tf.placeholder(tf.string, shape=[])
    # Create iterator
    iterator = tf.data.Iterator.from_string_handle(handle, trainset.output_types,
                                                           trainset.output_shapes)
    next_element = iterator.get_next()

    train_iter = trainset.make_one_shot_iterator()
    test_iter = testset.make_one_shot_iterator()

    data = next_element[0]
    label = next_element[1]

    num_samples, num_classes = train_y.shape
    num_features= train_x.shape[1]

    # create instance of the model 
    hidden1, fc1w, fc1b = layers.fc(data, num_features, 100, name='hidden1',log_weights=False)
    logits, fc2w, fc2b = layers.fc(hidden1, 100, num_classes, name='logits', relu=False, log_weights=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=label,
                        logits=logits)
    data_loss = tf.reduce_mean(cross_entropy)
    reg_loss = 1e-3 * tf.losses.get_regularization_loss()
    loss = tf.add(data_loss, reg_loss, name='data_and_reg_loss')
    global_step=tf.train.get_or_create_global_step()
    optimize = tf.train.RMSPropOptimizer(0.03).minimize(loss, global_step=global_step)
    prediction = tf.nn.softmax(logits)
    mistakes = tf.not_equal(
        tf.argmax(label, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_handle = sess.run(train_iter.string_handle())
        test_handle = sess.run(test_iter.string_handle())
        global_step=tf.train.get_or_create_global_step()
        for _ in range(1000):
            fetches = [ error,
                        loss,
                        optimize,
                        global_step]
            error_vl, loss_vl, _, global_step_vl = sess.run(fetches, feed_dict={handle: train_handle})
            if global_step_vl % 50 == 0:
                print('{}#: Training error {:6.2f}% - Training set loss: {:6.2f}'.format(
                      global_step_vl, 100 * error_vl, loss_vl ))
                error_vl, loss_vl, global_step_vl = sess.run([error,loss, global_step], feed_dict={handle: test_handle})
                print('{}#: Test error {:6.2f}% - Test set loss: {:6.2f}'.format(
                      global_step_vl, 100 * error_vl, loss_vl ))

        # Plot final classifier boundaries
        fetches = [fc1w, fc1b,fc2w,fc2b]
        W1, b1, W2, b2 = sess.run(fetches, feed_dict={handle: train_handle})
        plotml.plot2Dclassifier(W1, b1, W2, b2, test_x, test_y_scalar)

    #Alternatively, run without TF and plot for comparison with TF version
    w = ml.train_twolayernet(train_x, train_y_scalar)
    plotml.plot2Dclassifier(w['W1'], w['b1'], w['W2'], w['b2'],
                            test_x, test_y_scalar,
                            save_to='/tmp/toy_spiral_w_boundaries.png')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
