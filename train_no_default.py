import sys
import pprint

import tensorflow as tf

#from models.classification import ToyModel
from models.classification import FullyConnectedNet

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

    iterator = tf.data.Iterator.from_structure(dset_train.output_types,
                                               dset_train.output_shapes)

    training_init_op = iterator.make_initializer(dset_train)
    validation_init_op = iterator.make_initializer(dset_val)
    next_element = iterator.get_next()
    # create instance of the model 
    #model = ToyModel(data_loader=data_loader)
    model = FullyConnectedNet(config, data_loader=next_element)
    model.build_graph()
    # TODO There is a fancy way to get rid of this using decorators:
    # https://danijar.com/structuring-your-tensorflow-models/

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(validation_init_op)
        global_step=tf.train.get_or_create_global_step()
        for _ in range(20):
            accuracy, global_step_vl = sess.run([model.accuracy, global_step])
            print('{}#: Test accuracy {:6.2f}%'.format(global_step_vl, 100 * accuracy ))
            sess.run(training_init_op)
            for i in range(60):
                if i == 0:
                    global_step_vl, loss, _ = sess.run([global_step, model.loss, model.optimize])
                    print('{}#: Training set loss: {:6.2f}'.format(global_step_vl, loss))
                else:
                    sess.run(model.optimize)

    tf.logging.info('==== Configuration ====')
    tf.logging.info(pprint.pprint(config))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
