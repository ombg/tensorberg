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
from utils import data_utils

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
            normalize_data=False,
            subtract_mean=False,
            channels_first=False,
            flatten_imgs=False)

        data_utils.print_shape(data)
        X_train, y_train, X_val, y_val, X_test, y_test = data
#        #DEBUG - small subset to provoke overfitting
#        idx_overfit=np.random.choice(len(X_train),size=256,replace=False)
#        X_train= X_train[idx_overfit]
#        y_train= y_train[idx_overfit]
        self.num_batches = len(X_train) // self.config.batch_size

        self.train_dataset = self._from_numpy(X_train, y_train)
        self.val_dataset = self._from_numpy(X_val, y_val)
        self.test_dataset = self._from_numpy(X_test, y_test)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                   self.train_dataset.output_shapes)

        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        self.val_init_op = self.iterator.make_initializer(self.val_dataset)
        self.test_init_op = self.iterator.make_initializer(self.test_dataset)

    def _from_numpy(self, X, y):
        num_classes = len(np.unique(y))
        dset_x = tf.data.Dataset.from_tensor_slices(X)
        dset_y = tf.data.Dataset.from_tensor_slices(y).map(
            lambda z: tf.one_hot(z, num_classes))
        dset = tf.data.Dataset.zip((dset_x, dset_y))
        dset = dset.shuffle(buffer_size=len(X))
        dset = dset.repeat().batch(self.config.batch_size)
        return dset


    def initialize_train(self, sess):
        sess.run(self.training_init_op)
    def initialize_val(self, sess):
        sess.run(self.val_init_op)
    def initialize_test(self, sess):
        sess.run(self.test_init_op)
    def get_input(self):
        return self.iterator.get_next()
