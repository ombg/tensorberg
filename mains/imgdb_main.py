import sys

sys.path.extend(['..'])

import tensorflow as tf
from data_loader.cifar_imgdb import ImgdbLoader
from models.vggnet import Vgg16
from trainers.default_trainer import Trainer

from utils.config import process_config
from utils.dirs import create_dirs
#from utils.logger import DefinedSummarizer
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

    # create your data generator
    data_loader = ImgdbLoader(config)

    # create instance of the model you want
    model = Vgg16(data_loader, config)

    # create tensorboard logger
#    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir, 
#                               config=config,
#                               scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
#                                            'test/loss_per_epoch','test/acc_per_epoch'])

    # create trainer and path all previous components to it
    #trainer = Trainer(sess, model, data_loader, config, logger)
    trainer = Trainer(sess, model, data_loader, config)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
