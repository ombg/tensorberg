import sys
import pprint

import tensorflow as tf

#from models.classification import ToyModel
from models.classification import FullyConnectedNet

from utils.datahandler import TFRecordDatasetLoader
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
    data_loader = TFRecordDatasetLoader(config)

    data_loader.load_datasets(do_shuffle=True,
                              train_repetitions=-1)
    # create instance of the model 
    #model = ToyModel(data_loader=data_loader)
    model = FullyConnectedNet(config, data_loader=data_loader)
    model.build_graph()
    next_element = data_loader.get_input()
    # TODO There is a fancy way to get rid of this using decorators:
    # https://danijar.com/structuring-your-tensorflow-models/

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(20):
            data_loader.initialize_val(sess)
            accuracy = sess.run(model.accuracy)
            print('Test accuracy {:6.2f}%'.format(100 * accuracy))
            data_loader.initialize_train(sess)
            for i in range(60):
                if i == 0:
                    loss, _,_ = sess.run([model.loss, model.optimize, next_element])
                    print('Training set loss: {:6.2f}'.format(loss))
                else:
                    sess.run(next_element)
                    sess.run(model.optimize)

    tf.logging.info('==== Configuration ====')
    tf.logging.info(pprint.pprint(config))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
