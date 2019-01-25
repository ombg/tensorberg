import sys
import pprint

import numpy as np
import tensorflow as tf

from models.classification import ToyModel
from ompy import plotml, ml

def main():

    # create tensorflow session
    sess = tf.Session()
    num_classes = 3
    num_features = 2
    samples_per_class = 200

    # Loads data into a tf.dataset
    features, labels = plotml.spiral(samples_per_class, num_features, num_classes, plot_me=True)
    features = features.astype(np.float32)
    labels = ml.makeonehot(labels, num_classes=num_classes)
    assert features.shape[0] == labels.shape[0]

    dset_train = tf.data.Dataset.from_tensor_slices((features, labels))
    dset_train = dset_train.shuffle(num_classes * samples_per_class)
    dset_train = dset_train.batch(20)
    iter_train = dset_train.make_one_shot_iterator()
    features_tensor, labels_tensor = iter_train.get_next()

    # create instance of the model 
    model = ToyModel(features_tensor,
                     labels_tensor,
                     num_features=num_features,
                     num_classes=num_classes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        global_step=tf.train.get_or_create_global_step()
        for _ in range(15):
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
