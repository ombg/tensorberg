import sys
import pprint

import tensorflow as tf

from models.vggnet import Vgg16
from trainers.default_trainer import ClassificationTrainer

from utils.datahandler import DirectoryDatasetLoader
#from utils.datahandler import FileListDatasetLoader
from utils.config import process_config
from utils.utils import get_args
from utils.data_utils import parse_png

def main():
    # capture the config path from the run arguments
    args = get_args()
    config = process_config(args.config)

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
    trainer = ClassificationTrainer(sess, model, config, data_loader=image_data)

    tf.logging.info('Loading pre-trained weights...')
    model.load_weights_from_numpy(config.weights_file, sess)

    trainer.create_bottlenecks(subset='training')
    trainer.create_bottlenecks(subset='validation')
    
    tf.logging.info('==== Configuration ====')
    print(pprint.pprint(config))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    print("Done!")
