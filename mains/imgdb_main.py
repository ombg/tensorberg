import sys

sys.path.extend(['..'])

import tensorflow as tf
from models.vggnet import Vgg16
from trainers.default_trainer import Trainer

#from utils.datahandler import ImgdbLoader
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
    #data_loader = ImgdbLoader(config)
    data_loader = ImageDirLoader(config)

    # create instance of the model 
    model = Vgg16(data_loader, config)

    # Trainer loops over the data using the model
    trainer = Trainer(sess, model, data_loader, config)

    model.load_weights_from_numpy(config.weights_file, sess)
    # here you train your model
    #trainer.train()


if __name__ == '__main__':
    main()
