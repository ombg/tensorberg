import sys
import pprint

import numpy as np
import tensorflow as tf

from models.classification import ToyModel
from ompy import plotml, ml

def main():

    d =  np.load('/tmp/spiral_data_dict.npy')
    features = d.item().get('features')
    labels = d.item().get('labels')

    num_samples, num_classes = labels.shape
    # create tensorflow session
    sess = tf.Session()
    dset_train = tf.data.Dataset.from_tensor_slices((features, labels))
    dset_train = dset_train.shuffle(num_samples)
    dset_train = dset_train.batch(20)
    iter_train = dset_train.make_one_shot_iterator()
    next_element = iter_train.get_next()

    # create instance of the model 
    model = ToyModel(next_element[0],
                     next_element[1],
                     num_features=features.shape[1],
                     num_classes=num_classes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        global_step=tf.train.get_or_create_global_step()
        for _ in range(15):
            sess.run(next_element)
            fetches = [ model.error,
                        model.loss,
                        model.optimize,
                        global_step]
            error, loss, _, global_step_vl = sess.run(fetches)
            print('{}#: Training error {:6.2f}% - Training set loss: {:6.2f}'.format(
                     global_step_vl, 100 * error, loss ))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
