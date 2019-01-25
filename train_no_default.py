import sys
import pprint

import tensorflow as tf

from models.classification import ToyModel

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

    # Loads data into a tf.dataset
    dset_train = datahandler.dset_from_tfrecord('/tmp/cifar_tfrecord/train.tfrecords',
                                   do_shuffle=False,
                                   use_distortion=False)
    dset_val = datahandler.dset_from_tfrecord('/tmp/cifar_tfrecord/validation.tfrecords',
                                   do_shuffle=False,
                                   use_distortion=False)

    iter_train = dset_train.make_one_shot_iterator()
    next_train_sample = iter_train.get_next()

    iter_val = dset_val.make_one_shot_iterator()
    next_val_sample = iter_val.get_next()
    # create instance of the model 
    model = ToyModel(data_loader=next_train_sample)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        global_step=tf.train.get_or_create_global_step()
        for _ in range(20):
            sess.run(next_val_sample)
            error, global_step_vl = sess.run([model.error, global_step])
            print('{}#: Test error {:6.2f}%'.format(global_step_vl, 100 * error ))
            sess.run(next_train_sample)
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
