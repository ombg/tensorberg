import sys
import pprint

import tensorflow as tf

from models.classification import FullyConnectedNet
from trainers.default_trainer import ClassificationTrainer

from utils.datahandler import DirectoryDatasetLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

from utils.data_utils import parse_txt

def main():
    # capture the config path from the run arguments
    # then process the json configration file and print it
    args = get_args()
    config = process_config(args.config)

    # create tensorflow session
    sess = tf.Session()

    # Loads data into a tf.dataset
    bottlenecks_loader = DirectoryDatasetLoader(config)

    bottlenecks_loader.load_datasets(parse_txt,
                                     do_shuffle=True,
                                     train_repetitions=-1)
    # create instance of the model 
    model = FullyConnectedNet(config, data_loader=bottlenecks_loader)

    # TODO There is a fancy way to get rid of this using decorators:
    # https://danijar.com/structuring-your-tensorflow-models/
    model.build_graph()

    # Trainer loops over the data using the model
    trainer = ClassificationTrainer(sess,
                                    model,
                                    config,
                                    data_loader=bottlenecks_loader)

    trainer.train()
    trainer.test()
    tf.logging.info('==== Configuration ====')
    tf.logging.info(pprint.pprint(config))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
