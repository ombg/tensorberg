import sys
import pprint

sys.path.extend(['..'])

import tensorflow as tf

from models.fcn import OutputLayer
from trainers.default_trainer import Trainer

from utils.datahandler import DirectoryDatasetLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file and print it
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        tf.logging.error("missing or invalid arguments %s" % e)
        raise SystemExit

    # create tensorflow session
    sess = tf.Session()

    # Loads data into a tf.dataset
    bottlenecks_loader = DirectoryDatasetLoader(config)

    bottlenecks_loader.load_datasets(do_shuffle=True,
                                     is_png=False,
                                     train_repetitions=-1)
    # create instance of the model 
    model = OutputLayer(config, data_loader=bottlenecks_loader)

    # TODO There is a fancy way to get rid of this using decorators:
    # https://danijar.com/structuring-your-tensorflow-models/
    model.build_graph()

    # Trainer loops over the data using the model
    trainer = Trainer(sess, model, config, data_loader=bottlenecks_loader)

    #trainer.train()
    trainer.test(config.checkpoint_dir_restore)
    trainer.predict('/home/oliver/data/misc/keukenhof-tulpe.jpg')
    tf.logging.info('==== Configuration ====')
    tf.logging.info(pprint.pprint(config))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
