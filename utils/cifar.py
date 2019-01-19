import os.path

NUM_SAMPLES_TRAINING = 45000
NUM_SAMPLES_VALIDATION = 5000 
NUM_SAMPLES_TESTING = 10000

NUM_CLASSES = 10

HEIGHT = 32
WIDTH = 32
DEPTH = 3

"""Size of distorted image"""
RESIZE_TARGET_HEIGHT = 40
RESIZE_TARGET_WIDTH = 40
def get_data_path(config, subset):
    if subset == 'training':
        return os.path.join(config.data_path,'train.tfrecords')
    elif subset == 'validation':
        return os.path.join(config.data_path,'validation.tfrecords')
    elif subset == 'testing':
        return os.path.join(config.data_path,'eval.tfrecords')
    else:
        raise ValueError('Invalid data subset "%s"' % subset)
