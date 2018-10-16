"""
This class will contain different loaders for cifar 100 dataset
# Techniques
# FIT in ram
# - load numpys in the graph - Cifar100DataLoaderNumpy
# - generator python - BaselineCifar100Loader

# Doesn't fit in ram
# - load files but in tfrecords format - Cifar100TFRecord
# - load files from disk using dataset api - Cifar100IMGLoader

Supports IMGDB4 and CIFAR dataset
"""
from data_loader import data_utils
from tqdm import tqdm

import tensorflow as tf
import numpy as np

class ImgdbLoader:
    """
    Loads IMGDB dataset using TensorFlow-Dataset API
    """
    def __init__(self,config):
        print('Loading data...')
        self.config = config
        # Load data as numpy array
        data = data_utils.get_some_data(
            self.config.input_path,
            input_path_imgdb_test=self.config.input_path_test,
            dataset_name=self.config.dataset_name,
            subtract_mean=True,
            channels_first=False,
            reshape_data=True)

        data_utils.print_shape(data)
        X_train, y_train, X_val, y_val, X_test, y_test = data
        num_classes = len(np.unique(y_train))
        train_dataset_x = tf.data.Dataset.from_tensor_slices(X_train)
        train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train).map(
            lambda z: tf.one_hot(z, num_classes))
        self.train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=len(X_train))
        self.train_dataset = self.train_dataset.repeat().batch(self.config.batch_size)

        # Create an uninitializaed iterator which can be reused with
        # different tf.data.Datasets as long as they have the same shape and type
        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                   self.train_dataset.output_shapes)

        # Until now the iterator is not bound to a dataset and is uninitialized.
        # Therefore, we now create init_ops. Later, a session runs these init_ops.
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)

    def initialize(self, sess):
        sess.run(self.training_init_op)
    def get_input(self):
        return self.iterator.get_next()
