import sys
import os
import pprint

import tensorflow as tf

from models.regression import VggMod
from trainers.default_trainer import RegressionTrainer

from utils.datahandler import RegressionDatasetLoader
from utils.data_utils import load_image_and_blur, parse_png
from utils.config import process_config
from utils.general_utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    args = get_args()
    config = process_config(args.config)

    # create tensorflow session
    sess = tf.Session()

    # Loads data into a tf.dataset
    tf.logging.info('Loading dataset...')
    image_data = RegressionDatasetLoader(config, parse_png, load_image_and_blur)
    image_data.load_datasets(do_shuffle=True,
                             train_repetitions=-1)
    # create instance of the model 
    tf.logging.info('Setting up the model graph...')
    model = VggMod(config, data_loader=image_data)

    # TODO There is a fancy way to get rid of this using decorators:
    # https://danijar.com/structuring-your-tensorflow-models/
    model.build_graph()

    # Trainer loops over the data using the model
    tf.logging.info('Initialize model...')
    trainer = RegressionTrainer(sess, model, config, data_loader=image_data)

    keys = ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b',
            'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b',
            'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b',
            'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b',
            'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3_b']

    tf.logging.info('Loading model parameters...')
    model.load_weights_from_numpy(config.weights_file,
                                  sess,
                                  weights_to_load=keys)

    tf.logging.info('Starting training now...')
    trainer.train()
    tf.logging.info('Starting testing now...')
    trainer.test()
    tf.logging.info('==== Configuration summary====')
    print(pprint.pprint(config))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
