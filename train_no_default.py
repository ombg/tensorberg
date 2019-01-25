import sys
import pprint

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from models.classification import ToyModel
from ompy import plotml, ml

def main():

    d =  np.load('/tmp/spiral_data_dict.npy')
    features = d.item().get('features')
    labels = d.item().get('labels')
    
    num_samples, num_classes = labels.shape
    num_features=features.shape[1]
    #dset_train = tf.data.Dataset.from_tensor_slices((features, labels))
    #dset_train = dset_train.shuffle(num_samples)
    #dset_train = dset_train.batch(20)
    #iter_train = dset_train.make_one_shot_iterator()
    #next_element = iter_train.get_next()

    data = tf.placeholder(tf.float32, shape=[None, num_features], name='data')
    label = tf.placeholder(tf.float32, shape=[None, num_classes], name='label')
    # create instance of the model 
    model = ToyModel(data,
                     label,
                     num_features=num_features,
                     num_classes=num_classes)

    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        global_step=tf.train.get_or_create_global_step()
        bs = 5
        i=0
        while i < 20:
            #sess.run(next_element)
            fetches = [ model.error,
                        model.loss,
                        model.optimize,
                        global_step]
            error, loss, _, global_step_vl = sess.run(fetches, feed_dict={ data: features[i:i+bs], label: labels[i:i+bs]})
            print('{}#: Training error {:6.2f}% - Training set loss: {:6.2f}'.format(
                     global_step_vl, 100 * error, loss ))
            i+=bs
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
