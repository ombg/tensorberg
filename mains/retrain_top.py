import sys

sys.path.extend(['..'])

import tensorflow as tf

from models.fcn import FullyConnectedNet
from trainers.default_trainer import Trainer

from utils.datahandler import ImageDirLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


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
    bottlenecks_tensor = ImageDirLoader(config,
                                        do_shuffle=True,
                                        is_jpg=False,
                                        train_repetitions=-1)

    # create instance of the model 
    model = FullyConnectedNet(config, data_loader=bottlenecks_tensor)

    # Trainer loops over the data using the model
    trainer = Trainer(sess, model, config, data_loader=bottlenecks_tensor)

    trainer.train()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
