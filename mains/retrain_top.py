import sys

sys.path.extend(['..'])

import tensorflow as tf

from models.vggnet import Vgg16
from trainers.default_trainer import Trainer

from utils.datahandler import ImageDirLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

bottleneck_path = '/home/oliver/projects/crowdnet/tensorberg/no_sync/experiments/vgg16_finetuning_flowers/bottlenecks/daisy/10140303196_b88d3d6cec.txt'

def float_string_to_list(filename):

    try:
        with open(filename, 'r') as the_file:
            bottleneck_string = the_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        print('Invalid float found')
    except FileNotFoundError:
        print('Cannot find file: %s'.format(bottleneck_path))
    return bottleneck_values

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    print(float_string_to_list(bottleneck_path))
    print("Done!")
