import sys
import pprint

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from models import layers
from ompy import plotml, ml

def main():

    d =  np.load('/tmp/spiral_data.npy')
    features = d.item().get('features')
    labels = d.item().get('labels')
    labels_scalar = d.item().get('labels_scalar')
    num_samples, num_classes = labels.shape
    num_features=features.shape[1]
    dset_train = tf.data.Dataset.from_tensor_slices((features, labels))
    #dset_train = dset_train.shuffle(num_samples)
    dset_train = dset_train.repeat(-1)
    dset_train = dset_train.batch(features.shape[0]) #Use whole X, as in ml.train_twolayernet()
    iter_train = dset_train.make_one_shot_iterator()
    next_element = iter_train.get_next()
    data = next_element[0]
    label = next_element[1]
    #data = tf.placeholder(tf.float32, shape=[None, num_features], name='data')
    #label = tf.placeholder(tf.float32, shape=[None, num_classes], name='label')
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
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        global_step=tf.train.get_or_create_global_step()
        for _ in range(10000):
            #sess.run(next_element)
            fetches = [ error,
                        loss,
                        optimize,
                        global_step]
            error_vl, loss_vl, _, global_step_vl = sess.run(fetches)
            #error_vl, loss_vl, _, global_step_vl = sess.run(fetches, feed_dict={ data: features, label: labels})
            if global_step_vl % 1000 == 0:
                print('{}#: Training error {:6.2f}% - Training set loss: {:6.2f}'.format(
                      global_step_vl, 100 * error_vl, loss_vl ))

        # Plot final classifier boundaries
        fetches = [fc1w, fc1b,fc2w,fc2b]
        #W1, b1, W2, b2 = sess.run(fetches, feed_dict={ data: features, label: labels})
        W1, b1, W2, b2 = sess.run(fetches)
        plotml.plot2Dclassifier(W1, b1, W2, b2, features, labels_scalar)
        plotml.plot2layerNetClassifier(features, W1, W2, b1, b2, labels_scalar)

    # Alternatively, run without TF and plot for comparison with TF version
    w = ml.train_twolayernet(features, labels_scalar)
    plotml.plot2Dclassifier(w['W1'], w['b1'], w['W2'], w['b2'],
                            features, labels_scalar,
                            save_to='/tmp/toy_spiral_w_boundaries.png')
    plotml.plot2layerNetClassifier(features, w['W1'], w['W2'], w['b1'], w['b2'],
                                   labels_scalar,
                                   save_to='/tmp/toy_spiral_w_boundaries_B.png')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
