import sys
import pprint

import tensorflow as tf
from models.regression import VggMod
from trainers.default_trainer import Trainer

from utils.datahandler import DirectoryDatasetLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

from utils.data_utils import parse_png

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    sess = tf.Session()

    # Loads data into a tf.dataset
    tf.logging.info('Loading dataset...')
    image_data = DirectoryDatasetLoader(config)
    image_data.load_datasets(parse_png,
                             do_shuffle=True,
                             train_repetitions=-1)
    # create instance of the model 
    tf.logging.info('Setting up the model graph...')
    model = VggMod(config, data_loader=image_data)

    # TODO There is a fancy way to get rid of this using decorators:
    # https://danijar.com/structuring-your-tensorflow-models/
    model.build_graph()

    # Trainer loops over the data using the model
    tf.logging.info('Initialize model...')
    trainer = Trainer(sess, model, config, data_loader=image_data)

    tf.logging.info('Loading model parameters...')
    weights_to_load = ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b']
    model.load_weights_from_numpy(config.weights_file,
                                  sess,
                                  weights_to_load=weights_to_load)

    #tf.logging.info('Starting training now...')
    trainer.train()
    tf.logging.info('Starting testing now...')
    trainer.test()
    tf.logging.info('==== Configuration summary====')
    print(pprint.pprint(config))
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
