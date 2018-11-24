import sys
import pprint

sys.path.extend(['..'])

import tensorflow as tf

from models.vggnet import Vgg16
from trainers.default_trainer import Trainer

from utils.datahandler import DirectoryDatasetLoader
#from utils.datahandler import FileListDatasetLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

from utils.data_utils import parse_png

def main():
    # capture the config path from the run arguments
    # then process the json configration file and print it
    try:
        args = get_args()
        config = process_config(args.config)
        tf.logging.info('==== Configuration ====')
        tf.logging.info(pprint.pprint(config))
    except Exception as e:
        tf.logging.error("missing or invalid arguments %s" % e)
        raise SystemExit

    # create the experiments dirs
    create_dirs([config.bottleneck_dir, config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    sess = tf.Session()

    # Loads data into a tf.dataset
    image_data = DirectoryDatasetLoader(config)

    image_data.load_datasets(parse_png,
                             do_shuffle=False,
                             train_repetitions=1)

    # create instance of the model 
    model = Vgg16(config, data_loader=image_data)

    # Trainer loops over the data using the model
    trainer = Trainer(sess, model, config, data_loader=image_data)

    tf.logging.info('Loading pre-trained weights...')
    model.load_weights_from_numpy(config.weights_file, sess)

    trainer.create_bottlenecks(subset='training')
    trainer.create_bottlenecks(subset='validation')
    
    tf.logging.info('==== Configuration ====')
    tf.logging.info(pprint.pprint(config))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
