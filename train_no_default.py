import sys
import pprint

import numpy as np
import tensorflow as tf

from models.classification import ToyModel

from ompy import plotml, ml
from utils import datahandler
from utils.config import process_config
from utils.general_utils import get_args
from utils.data_utils import parse_txt

def main():
    # capture the config path from the run arguments
    # then process the json configration file and print it
    args = get_args()
    config = process_config(args.config)

    # create tensorflow session
    sess = tf.Session()
    num_classes = 4
    num_features = 2
    samples_per_class = 100

    # Loads data into a tf.dataset
    features, labels = plotml.spiral(samples_per_class, num_features, num_classes)
    features = features.astype(np.float32)
    labels = ml.makeonehot(labels, num_classes=num_classes)
    assert features.shape[0] == labels.shape[0]

    dset_train = tf.data.Dataset.from_tensor_slices((features, labels))
    dset_train = dset_train.shuffle(num_classes * samples_per_class)
    dset_train = dset_train.batch(1)
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
        for _ in range(20):
            error, global_step_vl = sess.run([model.error, global_step])
            print('{}#: Training error {:6.2f}%'.format(global_step_vl, 100 * error ))
            for i in range(60):
                if i == 0:
                    global_step_vl, loss, _ = sess.run([global_step, model.loss, model.optimize])
                    print('{}#: Training set loss: {:6.2f}'.format(global_step_vl, loss))
                else:
                    sess.run(model.optimize)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
